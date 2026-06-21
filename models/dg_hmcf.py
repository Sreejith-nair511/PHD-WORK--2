"""
DG-HMCF: Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion

Main model class that orchestrates:
  1. Per-modality branch encoders
  2. Missing modality handler + optional dropout augmentation
  3. Multi-scale temporal fusion per modality
  4. Dynamic reliability gating
  5. Hierarchical cross-modal transformer
  6. Adaptive fusion layer
  7. Multi-task classifier
  8. Explainability module
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from models.branches.speech_branch import SpeechBranch
from models.branches.text_branch import TextBranch
from models.branches.face_branch import FaceBranch
from models.branches.eeg_branch import EEGBranch
from models.modules.dynamic_gating import DynamicReliabilityGating
from models.modules.multiscale_temporal import MultiScaleTemporalFusion
from models.modules.hierarchical_cross_modal import HierarchicalCrossModalTransformer
from models.modules.adaptive_fusion import AdaptiveFusionLayer
from models.modules.missing_modality import MissingModalityHandler, ModalityDropout
from models.modules.explainability import ExplainabilityModule
from models.classifier import DepressionClassifier


class DGHMCF(nn.Module):
    """
    Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion model.

    Parameters
    ----------
    config : dict
        Model configuration dict (from base_config.yaml ``model`` section).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        embed_dim = config.get("speech_embed_dim", 256)  # same for all modalities
        fusion_dim = config.get("fusion_dim", 512)

        # ---- Branch encoders ---------------------------------------------
        sp_cfg = config.get("speech", {})
        self.speech_branch = SpeechBranch(
            wav2vec2_model=sp_cfg.get("wav2vec2_model", "facebook/wav2vec2-base-960h"),
            behavioral_feat_dim=sp_cfg.get("behavioral_features", 6),
            embed_dim=embed_dim,
            bilstm_hidden=sp_cfg.get("bilstm_hidden", 128),
            bilstm_layers=sp_cfg.get("bilstm_layers", 2),
            dropout=sp_cfg.get("dropout", 0.1),
        )

        tx_cfg = config.get("text", {})
        self.text_branch = TextBranch(
            roberta_model=tx_cfg.get("roberta_model", "roberta-base"),
            linguistic_feat_dim=tx_cfg.get("linguistic_features", 5),
            embed_dim=embed_dim,
            bilstm_hidden=tx_cfg.get("bilstm_hidden", 128),
            bilstm_layers=tx_cfg.get("bilstm_layers", 2),
            dropout=tx_cfg.get("dropout", 0.1),
        )

        fa_cfg = config.get("face", {})
        self.face_branch = FaceBranch(
            vit_model=fa_cfg.get("vit_model", "google/vit-base-patch16-224"),
            behavioral_feat_dim=fa_cfg.get("behavioral_features", 7),
            embed_dim=embed_dim,
            dropout=fa_cfg.get("dropout", 0.1),
        )

        eg_cfg = config.get("eeg", {})
        self.eeg_branch = EEGBranch(
            n_channels=eg_cfg.get("n_channels", 64),
            segment_length=eg_cfg.get("segment_length", 256),
            cnn_channels=eg_cfg.get("cnn_channels", [64, 128, 256]),
            embed_dim=embed_dim,
            bilstm_hidden=eg_cfg.get("bilstm_hidden", 128),
            bilstm_layers=eg_cfg.get("bilstm_layers", 2),
            dropout=eg_cfg.get("dropout", 0.1),
        )

        # ---- Missing modality handler ------------------------------------
        self.missing_handler = MissingModalityHandler(
            embed_dim=embed_dim,
            n_modalities=4,
        )
        self.modality_dropout = ModalityDropout(drop_prob=0.2, min_modalities=1)

        # ---- Multi-scale temporal fusion (per modality) -----------------
        ms_cfg = config.get("multiscale", {})
        ms_out_ch = ms_cfg.get("out_channels", 64)
        ms_kernels = ms_cfg.get("kernel_sizes", [3, 5, 7])
        ms_total_out = ms_out_ch * len(ms_kernels)

        self.multiscale_speech = MultiScaleTemporalFusion(
            in_channels=embed_dim, out_channels=ms_out_ch, kernel_sizes=ms_kernels
        )
        self.multiscale_text = MultiScaleTemporalFusion(
            in_channels=embed_dim, out_channels=ms_out_ch, kernel_sizes=ms_kernels
        )
        self.multiscale_face = MultiScaleTemporalFusion(
            in_channels=embed_dim, out_channels=ms_out_ch, kernel_sizes=ms_kernels
        )
        self.multiscale_eeg = MultiScaleTemporalFusion(
            in_channels=embed_dim, out_channels=ms_out_ch, kernel_sizes=ms_kernels
        )

        # Project multi-scale output back to embed_dim
        self.ms_proj = nn.Linear(ms_total_out, embed_dim)

        # ---- Dynamic reliability gating ---------------------------------
        gate_cfg = config.get("gating", {})
        self.gating = DynamicReliabilityGating(
            embed_dim=embed_dim,
            n_modalities=4,
            hidden_dim=gate_cfg.get("hidden_dim", 128),
            temperature=gate_cfg.get("temperature", 1.0),
        )

        # ---- Hierarchical cross-modal transformer -----------------------
        cm_cfg = config.get("cross_modal", {})
        self.cross_modal = HierarchicalCrossModalTransformer(
            embed_dim=embed_dim,
            n_heads=cm_cfg.get("n_heads", 8),
            n_layers=cm_cfg.get("n_layers", 2),
            dropout=cm_cfg.get("dropout", 0.1),
            ffn_dim=cm_cfg.get("ffn_dim", 1024),
        )

        # ---- Adaptive fusion --------------------------------------------
        self.fusion = AdaptiveFusionLayer(
            embed_dim=embed_dim,
            fusion_dim=fusion_dim,
            n_modalities=4,
        )

        # ---- Classifier -------------------------------------------------
        cls_cfg = config.get("classifier", {})
        self.classifier = DepressionClassifier(
            fusion_dim=fusion_dim,
            hidden_dim=cls_cfg.get("hidden_dim", 256),
            n_classes=cls_cfg.get("n_classes", 2),
            dropout=cls_cfg.get("dropout", 0.3),
        )

        # ---- Explainability module --------------------------------------
        self.explainability = ExplainabilityModule(
            embed_dim=embed_dim,
            fusion_dim=fusion_dim,
            n_modalities=4,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full model forward pass.

        Parameters
        ----------
        batch : dict
            Keys expected:
              ``speech``        – dict with waveform, attention_mask, behavioral_features
              ``text``          – dict with input_ids, attention_mask, linguistic_features
              ``face``          – dict with pixel_values, frame_mask, behavioral_features
              ``eeg``           – dict with segments, segment_mask
              ``modality_mask`` – (B, 4) float tensor

        Returns
        -------
        dict with keys:
            ``classification_logits`` – (B, 2)
            ``phq8_score``            – (B,)
            ``reliability_weights``   – (B, 4)
            ``attention_maps``        – dict of pair → attn tensor
            ``modality_importance``   – (B, 4)
            ``fused_features``        – (B, fusion_dim)
        """
        modality_mask = batch["modality_mask"]  # (B, 4)
        device = modality_mask.device

        # ---- Step 1: Extract per-modality embeddings --------------------
        embeddings: Dict[str, Optional[torch.Tensor]] = {
            "speech": None, "text": None, "face": None, "eeg": None
        }

        if batch.get("speech") is not None and modality_mask[:, 0].sum() > 0:
            sp = batch["speech"]
            embeddings["speech"] = self.speech_branch(
                input_values=sp["waveform"],
                attention_mask=sp.get("attention_mask"),
                behavioral_features=sp.get("behavioral_features"),
            )

        if batch.get("text") is not None and modality_mask[:, 1].sum() > 0:
            tx = batch["text"]
            embeddings["text"] = self.text_branch(
                input_ids=tx["input_ids"],
                attention_mask=tx.get("attention_mask"),
                linguistic_features=tx.get("linguistic_features"),
            )

        if batch.get("face") is not None and modality_mask[:, 2].sum() > 0:
            fa = batch["face"]
            embeddings["face"] = self.face_branch(
                pixel_values=fa["pixel_values"],
                frame_mask=fa.get("frame_mask"),
                behavioral_features=fa.get("behavioral_features"),
            )

        if batch.get("eeg") is not None and modality_mask[:, 3].sum() > 0:
            eg = batch["eeg"]
            embeddings["eeg"] = self.eeg_branch(
                eeg_segments=eg["segments"],
                segment_mask=eg.get("segment_mask"),
            )

        # ---- Step 2: Modality dropout (training augmentation) ----------
        embeddings, modality_mask = self.modality_dropout(embeddings, modality_mask)

        # ---- Step 3: Handle missing modalities -------------------------
        complete_embeddings = self.missing_handler(embeddings, modality_mask)

        # ---- Step 4: Multi-scale temporal fusion (on each modality) ----
        def apply_multiscale(emb: torch.Tensor, ms_module) -> torch.Tensor:
            # emb: (B, embed_dim) → expand to (B, 1, embed_dim) for temporal module
            x = emb.unsqueeze(1)           # (B, 1, embed_dim)
            x = ms_module(x)               # (B, 1, ms_total_out)
            x = self.ms_proj(x)            # (B, 1, embed_dim)
            return x.squeeze(1)            # (B, embed_dim)

        ms_embeddings = {
            "speech": apply_multiscale(complete_embeddings["speech"], self.multiscale_speech),
            "text":   apply_multiscale(complete_embeddings["text"],   self.multiscale_text),
            "face":   apply_multiscale(complete_embeddings["face"],   self.multiscale_face),
            "eeg":    apply_multiscale(complete_embeddings["eeg"],    self.multiscale_eeg),
        }

        # ---- Step 5: Dynamic reliability gating -----------------------
        emb_list = [
            ms_embeddings.get("speech"),
            ms_embeddings.get("text"),
            ms_embeddings.get("face"),
            ms_embeddings.get("eeg"),
        ]
        reliability_weights = self.gating(emb_list, modality_mask)  # (B, 4)

        # ---- Step 6: Hierarchical cross-modal transformer --------------
        cross_modal_embs, attention_maps = self.cross_modal(
            speech_emb=ms_embeddings["speech"],
            text_emb=ms_embeddings["text"],
            face_emb=ms_embeddings["face"],
            eeg_emb=ms_embeddings["eeg"],
            modality_mask=modality_mask,
        )

        # ---- Step 7: Adaptive fusion -----------------------------------
        fused = self.fusion(
            embeddings=ms_embeddings,
            reliability_weights=reliability_weights,
            cross_modal_embeddings=cross_modal_embs,
        )  # (B, fusion_dim)

        # ---- Step 8: Classification ------------------------------------
        cls_logits, phq8_score = self.classifier(fused)

        # ---- Step 9: Explainability ------------------------------------
        explain_out = self.explainability(
            reliability_weights=reliability_weights,
            cross_modal_attention_weights=attention_maps,
            embeddings=ms_embeddings,
        )

        return {
            "classification_logits": cls_logits,
            "phq8_score": phq8_score,
            "reliability_weights": reliability_weights,
            "attention_maps": attention_maps,
            "modality_importance": explain_out["modality_importance"],
            "fused_features": fused,
        }

    def predict(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference-mode forward with probability outputs and report.

        Returns classification probabilities, predicted label, estimated PHQ-8,
        and explainability report for each sample in the batch.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)

        probs = torch.softmax(outputs["classification_logits"], dim=-1)
        pred_labels = probs.argmax(dim=-1)

        phq8_raw = outputs["phq8_score"] * 24.0  # denormalize

        reports = []
        for b in range(outputs["classification_logits"].size(0)):
            report = self.explainability.generate_report(
                explainability_output={
                    "modality_importance": outputs["modality_importance"],
                    "reliability_weights": outputs["reliability_weights"],
                    "embedding_norms": outputs.get("embedding_norms", outputs["reliability_weights"]),
                    "attention_maps": outputs["attention_maps"],
                },
                prediction={
                    "classification_logits": outputs["classification_logits"],
                    "phq8_score": outputs["phq8_score"],
                },
                sample_idx=b,
            )
            reports.append(report)

        return {
            **outputs,
            "probabilities": probs,
            "predicted_labels": pred_labels,
            "phq8_score_raw": phq8_raw,
            "reports": reports,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Return trainable parameter counts per sub-module."""
        counts = {}
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = n
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
