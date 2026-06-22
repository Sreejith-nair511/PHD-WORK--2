"""
run_experiment.py — DG-HMCF DAIC-WOZ Quick Experiment Runner
=============================================================
One-command execution:
    python run_experiment.py --data_root data/raw/daic_woz --output_dir outputs/experiment_1

Produces:
  outputs/experiment_1/
    ├── best_model.pt
    ├── val_metrics.json
    ├── test_metrics.json
    ├── classification_report.txt
    ├── predictions.csv
    ├── confusion_matrix.png
    └── training_curves.png
"""

import os, sys, json, logging, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("DG-HMCF")

# ── optional imports (checked at runtime) ────────────────────────────────────
def _require(pkg):
    try:
        return __import__(pkg)
    except ImportError:
        log.error("Missing package: %s — run: pip install %s", pkg, pkg)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DG-HMCF DAIC-WOZ Experiment Runner")
    p.add_argument("--data_root",   default="data/raw/daic_woz",
                   help="Path to DAIC-WOZ root folder")
    p.add_argument("--output_dir",  default="outputs/experiment_1",
                   help="Where to save all outputs")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--max_audio_s", type=float, default=30.0,
                   help="Max audio seconds per utterance to process")
    p.add_argument("--text_model",  default="roberta-base")
    p.add_argument("--speech_model",default="facebook/wav2vec2-base-960h")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no_speech",   action="store_true",
                   help="Skip speech modality (text-only, faster)")
    p.add_argument("--device",      default=None,
                   help="cuda / cpu (auto-detected if omitted)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_labels(data_root: Path) -> dict:
    """
    Load train / dev / test splits from DAIC-WOZ CSV files.
    Returns dict: split → DataFrame with columns [Participant_ID, PHQ8_Score, PHQ8_Binary]
    """
    splits = {}
    label_files = {
        "train": "train_split_Depression_AVEC2017.csv",
        "dev":   "dev_split_Depression_AVEC2017.csv",
        "test":  "test_split_Depression_AVEC2017.csv",
    }
    for split, fname in label_files.items():
        fpath = data_root / fname
        if not fpath.exists():
            log.warning("Label file not found: %s — will skip split '%s'", fpath, split)
            continue
        df = pd.read_csv(fpath)
        # Normalise column names (DAIC-WOZ uses varying capitalisation)
        df.columns = [c.strip() for c in df.columns]
        # Map common variants
        rename_map = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "_").replace("-", "_")
            if "participant" in lc:        rename_map[col] = "Participant_ID"
            elif "phq8_score" in lc or "phq_score" in lc: rename_map[col] = "PHQ8_Score"
            elif "phq8_binary" in lc or "phq_binary" in lc: rename_map[col] = "PHQ8_Binary"
        df = df.rename(columns=rename_map)
        # Derive binary label if missing
        if "PHQ8_Binary" not in df.columns and "PHQ8_Score" in df.columns:
            df["PHQ8_Binary"] = (df["PHQ8_Score"] >= 10).astype(int)
        splits[split] = df
        log.info("Loaded %s split: %d participants, %d depressed",
                 split, len(df),
                 int(df["PHQ8_Binary"].sum()) if "PHQ8_Binary" in df.columns else -1)
    return splits


def load_transcript(data_root: Path, pid: int) -> str:
    """
    Load participant-only transcript text for a given participant ID.
    Filters out interviewer (Ellie) turns.
    Returns concatenated participant utterances as a single string.
    """
    transcript_path = data_root / str(pid) / f"{pid}_TRANSCRIPT.csv"
    if not transcript_path.exists():
        return ""
    try:
        df = pd.read_csv(transcript_path, sep="\t", header=None,
                         names=["start", "stop", "speaker", "value"],
                         on_bad_lines="skip")
        # Filter participant only (not Ellie)
        participant = df[~df["speaker"].str.strip().str.lower().isin(
            ["ellie", "interviewer", "e"]
        )]["value"].dropna()
        text = " ".join(participant.astype(str).tolist()).strip()
        return text
    except Exception as e:
        log.debug("Transcript load error for %d: %s", pid, e)
        return ""


def load_audio_path(data_root: Path, pid: int):
    """Return audio file path if it exists, else None."""
    audio_path = data_root / str(pid) / f"{pid}_AUDIO.wav"
    if audio_path.exists():
        return audio_path
    # Some versions use _P.wav suffix
    alt = data_root / str(pid) / f"{pid}_P.wav"
    if alt.exists():
        return alt
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_embeddings(texts: list, model_name: str, device, batch_size=8) -> np.ndarray:
    """
    Extract RoBERTa [CLS] embeddings with chunking for long sequences.
    Sequences longer than 512 tokens are split into overlapping chunks;
    chunk embeddings are mean-pooled into a single patient-level vector.
    Returns: (N, hidden_size) numpy array
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    log.info("Loading text model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []
    chunk_size  = 512
    stride      = 256   # 50% overlap

    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch_texts = texts[i : i + batch_size]
        batch_embs  = []

        for text in batch_texts:
            if not text or not text.strip():
                batch_embs.append(np.zeros(model.config.hidden_size, dtype=np.float32))
                continue

            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

            if len(tokens) <= chunk_size - 2:
                # Short enough — process directly
                enc = tokenizer(
                    text, return_tensors="pt",
                    max_length=chunk_size, truncation=True, padding=True
                ).to(device)
                with torch.no_grad():
                    out = model(**enc)
                emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            else:
                # Chunk with stride and mean-pool
                chunk_embs = []
                for start in range(0, len(tokens), stride):
                    chunk = tokens[start : start + chunk_size - 2]
                    if not chunk:
                        break
                    chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
                    enc = tokenizer(
                        chunk_text, return_tensors="pt",
                        max_length=chunk_size, truncation=True, padding=True
                    ).to(device)
                    with torch.no_grad():
                        out = model(**enc)
                    chunk_embs.append(
                        out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                    )
                emb = np.mean(chunk_embs, axis=0)

            batch_embs.append(emb.astype(np.float32))

        embeddings.extend(batch_embs)

    return np.array(embeddings, dtype=np.float32)


def extract_speech_embeddings(audio_paths: list, model_name: str, device,
                               max_audio_s: float = 30.0) -> np.ndarray:
    """
    Extract Wav2Vec2 embeddings from utterance-level audio.
    Audio is trimmed to max_audio_s seconds, then mean-pooled over time frames
    to produce a single patient-level vector.
    Returns: (N, hidden_size) numpy array
    """
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    log.info("Loading speech model: %s", model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()

    TARGET_SR  = 16000
    max_frames = int(TARGET_SR * max_audio_s)
    hidden_size = model.config.hidden_size

    try:
        import librosa
        USE_LIBROSA = True
    except ImportError:
        try:
            import torchaudio
            USE_LIBROSA = False
        except ImportError:
            log.error("Install librosa or torchaudio for speech processing.")
            sys.exit(1)

    embeddings = []

    for path in tqdm(audio_paths, desc="Speech embeddings"):
        if path is None:
            embeddings.append(np.zeros(hidden_size, dtype=np.float32))
            continue
        try:
            if USE_LIBROSA:
                import librosa
                audio, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
            else:
                import torchaudio
                audio_t, sr = torchaudio.load(str(path))
                if sr != TARGET_SR:
                    audio_t = torchaudio.functional.resample(audio_t, sr, TARGET_SR)
                audio = audio_t.mean(0).numpy()

            # Trim to max length
            audio = audio[:max_frames]

            inputs = processor(
                audio, sampling_rate=TARGET_SR,
                return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)

            # Mean-pool over time → (hidden_size,)
            emb = out.last_hidden_state.squeeze(0).mean(0).cpu().numpy()
            embeddings.append(emb.astype(np.float32))

        except Exception as e:
            log.debug("Speech embedding error for %s: %s", path, e)
            embeddings.append(np.zeros(hidden_size, dtype=np.float32))

    return np.array(embeddings, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — DG-HMCF FUSION MODEL
# ═══════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGatingNetwork(nn.Module):
    """
    Computes per-sample reliability weights for text and speech modalities.
    Uses a small MLP per modality, then masked softmax.
    """
    def __init__(self, text_dim: int, speech_dim: int, hidden: int = 128):
        super().__init__()
        self.text_scorer   = nn.Sequential(
            nn.Linear(text_dim,   hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.speech_scorer = nn.Sequential(
            nn.Linear(speech_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, text_emb, speech_emb, speech_mask):
        """
        speech_mask: (B,) float — 1 if speech present, 0 if absent
        Returns: text_w (B,1), speech_w (B,1)
        """
        t_score = self.text_scorer(text_emb)        # (B,1)
        s_score = self.speech_scorer(speech_emb)    # (B,1)

        # Mask absent speech
        NEG_INF = -1e9
        s_score = s_score * speech_mask.unsqueeze(1) + \
                  NEG_INF * (1 - speech_mask.unsqueeze(1))

        scores  = torch.cat([t_score, s_score], dim=1)  # (B,2)
        weights = F.softmax(scores / self.temperature.abs().clamp(min=1e-3), dim=1)
        return weights[:, 0:1], weights[:, 1:2]         # text_w, speech_w


class HierarchicalCrossModalBlock(nn.Module):
    """Single cross-attention block: query attends to key/value."""
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )

    def forward(self, q, kv):
        # q, kv: (B, 1, dim)
        attn_out, _ = self.attn(q, kv, kv)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q


class DGHMCFClassifier(nn.Module):
    """
    Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion
    Text + Speech → Fused → Binary Depression Classifier
    """
    def __init__(self, text_dim: int, speech_dim: int,
                 fusion_dim: int = 256, n_heads: int = 8,
                 dropout: float = 0.3):
        super().__init__()

        # Project both modalities to a shared fusion_dim
        self.text_proj   = nn.Linear(text_dim,   fusion_dim)
        self.speech_proj = nn.Linear(speech_dim, fusion_dim)

        # Dynamic gating
        self.gating = DynamicGatingNetwork(fusion_dim, fusion_dim, hidden=128)

        # Hierarchical cross-modal transformer (text ↔ speech)
        self.text2speech = HierarchicalCrossModalBlock(fusion_dim, n_heads, dropout=0.1)
        self.speech2text = HierarchicalCrossModalBlock(fusion_dim, n_heads, dropout=0.1)

        # Adaptive fusion: weighted sum + cross-modal + residual
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_emb, speech_emb, speech_mask):
        """
        text_emb   : (B, text_dim)
        speech_emb : (B, speech_dim)
        speech_mask: (B,) float  — 1=present, 0=absent
        Returns logits (B, 2)
        """
        # Project to fusion_dim
        t = self.text_proj(text_emb)     # (B, fusion_dim)
        s = self.speech_proj(speech_emb) # (B, fusion_dim)

        # Dynamic reliability weights
        t_w, s_w = self.gating(t, s, speech_mask)  # (B,1) each

        # Weighted embeddings
        t_weighted = t * t_w
        s_weighted = s * s_w

        # Cross-modal transformer (add seq dim)
        t_seq = t_weighted.unsqueeze(1)   # (B,1,fusion_dim)
        s_seq = s_weighted.unsqueeze(1)

        t_enhanced = self.text2speech(t_seq, s_seq).squeeze(1)
        s_enhanced = self.speech2text(s_seq, t_seq).squeeze(1)

        # Adaptive fusion: concat + project + residual
        fused = self.fusion_proj(
            torch.cat([t_enhanced, s_enhanced], dim=-1)
        )  # (B, fusion_dim)
        fused = fused + (t_weighted + s_weighted)  # residual

        logits = self.classifier(fused)
        return logits


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATASET & DATALOADER
# ═══════════════════════════════════════════════════════════════════════════

class DepressionDataset(torch.utils.data.Dataset):
    def __init__(self, text_embs, speech_embs, speech_masks, labels, pids):
        self.text_embs    = torch.tensor(text_embs,    dtype=torch.float32)
        self.speech_embs  = torch.tensor(speech_embs,  dtype=torch.float32)
        self.speech_masks = torch.tensor(speech_masks, dtype=torch.float32)
        self.labels       = torch.tensor(labels,       dtype=torch.long)
        self.pids         = pids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text_emb":    self.text_embs[idx],
            "speech_emb":  self.speech_embs[idx],
            "speech_mask": self.speech_masks[idx],
            "label":       self.labels[idx],
            "pid":         self.pids[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        t   = batch["text_emb"].to(device)
        s   = batch["speech_emb"].to(device)
        sm  = batch["speech_mask"].to(device)
        lbl = batch["label"].to(device)

        optimizer.zero_grad()
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                logits = model(t, s, sm)
                loss   = criterion(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(t, s, sm)
            loss   = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * lbl.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == lbl).sum().item()
        total      += lbl.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_probs, all_labels, all_pids = [], [], [], []
    total_loss, total = 0.0, 0

    for batch in loader:
        t   = batch["text_emb"].to(device)
        s   = batch["speech_emb"].to(device)
        sm  = batch["speech_mask"].to(device)
        lbl = batch["label"].to(device)

        logits = model(t, s, sm)
        loss   = criterion(logits, lbl)

        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * lbl.size(0)
        total      += lbl.size(0)
        all_preds .extend(preds.cpu().tolist())
        all_probs .extend(probs.cpu().tolist())
        all_labels.extend(lbl.cpu().tolist())
        all_pids  .extend(batch["pid"])

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / total
    return metrics, all_preds, all_probs, all_labels, all_pids


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(labels, preds, probs):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        classification_report
    )
    labels = np.array(labels)
    preds  = np.array(preds)
    probs  = np.array(probs)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    cm     = confusion_matrix(labels, preds)
    report = classification_report(labels, preds,
                                   target_names=["Not Depressed", "Depressed"],
                                   zero_division=0)
    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(cm_data, out_path, title="Confusion Matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array(cm_data)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Depressed", "Depressed"],
                yticklabels=["Not Depressed", "Depressed"])
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved confusion matrix → %s", out_path)


def save_training_curves(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_f1"],  label="F1",  color="green")
    axes[1].plot(epochs, history["val_acc"], label="Acc", color="blue")
    axes[1].set_title("Val F1 & Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_auc"], label="ROC-AUC", color="purple")
    axes[2].set_title("Val ROC-AUC"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle("DG-HMCF Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved training curves → %s", out_path)


def save_roc_curve(labels, probs, out_path, title="ROC Curve"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    labels = np.array(labels)
    probs  = np.array(probs)
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved ROC curve → %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — FEATURE CACHE (avoids re-extracting every run)
# ═══════════════════════════════════════════════════════════════════════════

def get_cache_path(output_dir: Path, split: str, modality: str) -> Path:
    return output_dir / "feature_cache" / f"{split}_{modality}.npy"


def save_cache(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)
    log.info("Cached %s features → %s", path.stem, path)


def load_cache(path: Path):
    if path.exists():
        log.info("Loading cached features from %s", path)
        return np.load(str(path), allow_pickle=False)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def prepare_split_data(split_name, split_df, data_root, args, device, output_dir):
    """
    For a given split DataFrame, extract/load all features.
    Returns (text_embs, speech_embs, speech_masks, labels, pids).
    """
    pids   = split_df["Participant_ID"].tolist()
    labels = split_df["PHQ8_Binary"].tolist()

    # ── Text features ─────────────────────────────────────────────────────
    text_cache = get_cache_path(output_dir, split_name, "text")
    text_embs  = load_cache(text_cache)

    if text_embs is None:
        log.info("[%s] Extracting text embeddings for %d participants...", split_name, len(pids))
        texts     = [load_transcript(data_root, pid) for pid in tqdm(pids, desc="Load transcripts")]
        empty     = sum(1 for t in texts if not t.strip())
        log.info("[%s] %d / %d transcripts loaded (empty: %d)", split_name, len(texts), len(pids), empty)
        text_embs = extract_text_embeddings(texts, args.text_model, device)
        save_cache(text_embs, text_cache)

    # ── Speech features ───────────────────────────────────────────────────
    speech_cache = get_cache_path(output_dir, split_name, "speech")
    speech_embs  = load_cache(speech_cache)
    speech_masks = np.ones(len(pids), dtype=np.float32)

    if speech_embs is None and not args.no_speech:
        log.info("[%s] Extracting speech embeddings for %d participants...", split_name, len(pids))
        audio_paths = [load_audio_path(data_root, pid) for pid in pids]
        missing     = sum(1 for p in audio_paths if p is None)
        if missing:
            log.warning("[%s] %d / %d audio files missing — using zero vectors", split_name, missing, len(pids))
        speech_masks = np.array([1.0 if p is not None else 0.0 for p in audio_paths], dtype=np.float32)
        speech_embs  = extract_speech_embeddings(audio_paths, args.speech_model, device, args.max_audio_s)
        save_cache(speech_embs,  speech_cache)
        save_cache(speech_masks, get_cache_path(output_dir, split_name, "speech_mask"))
    elif args.no_speech:
        log.info("[%s] Speech modality disabled — using zero vectors", split_name)
        speech_dim  = 768  # wav2vec2-base hidden size
        speech_embs = np.zeros((len(pids), speech_dim), dtype=np.float32)
        speech_masks = np.zeros(len(pids), dtype=np.float32)
    else:
        mask_cache   = get_cache_path(output_dir, split_name, "speech_mask")
        cached_masks = load_cache(mask_cache)
        speech_masks = cached_masks if cached_masks is not None else speech_masks

    return text_embs, speech_embs, speech_masks, np.array(labels), pids


def print_metrics(metrics: dict, split: str):
    log.info("=" * 55)
    log.info("  %s METRICS", split.upper())
    log.info("=" * 55)
    log.info("  Accuracy  : %.4f", metrics["accuracy"])
    log.info("  Precision : %.4f", metrics["precision"])
    log.info("  Recall    : %.4f", metrics["recall"])
    log.info("  F1-Score  : %.4f", metrics["f1"])
    log.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
    log.info("  Loss      : %.4f", metrics.get("loss", float("nan")))
    log.info("=" * 55)
    log.info("\n%s", metrics["classification_report"])


def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── Paths ─────────────────────────────────────────────────────────────
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        log.error("data_root does not exist: %s", data_root)
        log.error("Please place your DAIC-WOZ files in that directory.")
        sys.exit(1)

    # ── Load label splits ─────────────────────────────────────────────────
    splits = load_labels(data_root)
    if not splits:
        log.error("No label CSV files found in %s", data_root)
        sys.exit(1)

    required = ["train", "dev"]
    for sp in required:
        if sp not in splits:
            log.error("Missing required split: '%s'", sp)
            sys.exit(1)

    # ── Extract / load features ───────────────────────────────────────────
    log.info("\n── Preparing TRAIN split ──")
    tr_text, tr_speech, tr_mask, tr_labels, tr_pids = prepare_split_data(
        "train", splits["train"], data_root, args, device, output_dir
    )

    log.info("\n── Preparing DEV (val) split ──")
    va_text, va_speech, va_mask, va_labels, va_pids = prepare_split_data(
        "dev", splits["dev"], data_root, args, device, output_dir
    )

    has_test = "test" in splits
    if has_test:
        log.info("\n── Preparing TEST split ──")
        te_text, te_speech, te_mask, te_labels, te_pids = prepare_split_data(
            "test", splits["test"], data_root, args, device, output_dir
        )

    # ── Build datasets and loaders ────────────────────────────────────────
    train_ds = DepressionDataset(tr_text, tr_speech, tr_mask, tr_labels, tr_pids)
    val_ds   = DepressionDataset(va_text, va_speech, va_mask, va_labels, va_pids)

    # Class weights for imbalanced data
    class_counts = np.bincount(tr_labels)
    class_weights = torch.tensor(
        1.0 / (class_counts + 1e-6), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0
    )
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    if has_test:
        test_ds     = DepressionDataset(te_text, te_speech, te_mask, te_labels, te_pids)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

    # ── Build model ───────────────────────────────────────────────────────
    text_dim   = tr_text.shape[1]
    speech_dim = tr_speech.shape[1]
    log.info("Text embedding dim: %d | Speech embedding dim: %d", text_dim, speech_dim)

    model = DGHMCFClassifier(
        text_dim=text_dim,
        speech_dim=speech_dim,
        fusion_dim=256,
        n_heads=8,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("DG-HMCF Classifier — trainable parameters: %d (%.2fM)",
             total_params, total_params / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Training loop ─────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [], "val_auc": []
    }
    best_val_f1  = -1.0
    best_metrics = {}
    patience     = 7
    no_improve   = 0

    log.info("\n── Starting training: %d epochs ──", args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_metrics, _, val_probs, val_lbls, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["roc_auc"])

        log.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.3f | "
            "val_loss=%.4f acc=%.3f f1=%.3f auc=%.3f | lr=%.2e",
            epoch, args.epochs,
            train_loss, train_acc,
            val_metrics["loss"], val_metrics["accuracy"],
            val_metrics["f1"], val_metrics["roc_auc"],
            optimizer.param_groups[0]["lr"]
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_metrics = val_metrics
            no_improve   = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, output_dir / "best_model.pt")
            log.info("  ✓ New best val F1=%.4f — checkpoint saved", best_val_f1)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d (no val F1 improvement for %d epochs)",
                         epoch, patience)
                break

    # ── Load best model for final evaluation ──────────────────────────────
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("\nLoaded best model from epoch %d (val F1=%.4f)",
             ckpt["epoch"], best_val_f1)

    # ── Final validation metrics ──────────────────────────────────────────
    val_metrics, val_preds, val_probs, val_lbls, val_pid_list = evaluate(
        model, val_loader, criterion, device
    )
    print_metrics(val_metrics, "VALIDATION")

    with open(output_dir / "val_metrics.json", "w") as f:
        json.dump({k: v for k, v in val_metrics.items()
                   if k != "classification_report"}, f, indent=2)
    with open(output_dir / "val_classification_report.txt", "w") as f:
        f.write(val_metrics["classification_report"])

    pd.DataFrame({
        "participant_id": val_pid_list,
        "true_label":     val_lbls,
        "predicted_label": val_preds,
        "prob_depressed":  val_probs,
    }).to_csv(output_dir / "val_predictions.csv", index=False)

    save_confusion_matrix(val_metrics["confusion_matrix"],
                          output_dir / "val_confusion_matrix.png",
                          "Validation Confusion Matrix")
    save_roc_curve(val_lbls, val_probs,
                   output_dir / "val_roc_curve.png",
                   "Validation ROC Curve")

    # ── Test evaluation ───────────────────────────────────────────────────
    if has_test:
        te_metrics, te_preds, te_probs, te_lbls, te_pid_list = evaluate(
            model, test_loader, criterion, device
        )
        print_metrics(te_metrics, "TEST")

        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump({k: v for k, v in te_metrics.items()
                       if k != "classification_report"}, f, indent=2)
        with open(output_dir / "test_classification_report.txt", "w") as f:
            f.write(te_metrics["classification_report"])

        pd.DataFrame({
            "participant_id": te_pid_list,
            "true_label":     te_lbls,
            "predicted_label": te_preds,
            "prob_depressed":  te_probs,
        }).to_csv(output_dir / "test_predictions.csv", index=False)

        save_confusion_matrix(te_metrics["confusion_matrix"],
                              output_dir / "test_confusion_matrix.png",
                              "Test Confusion Matrix")
        save_roc_curve(te_lbls, te_probs,
                       output_dir / "test_roc_curve.png",
                       "Test ROC Curve")
    else:
        log.warning("No test split found — skipping test evaluation.")

    # ── Training curves ───────────────────────────────────────────────────
    save_training_curves(history, output_dir / "training_curves.png")

    # ── Summary report ────────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("  EXPERIMENT COMPLETE")
    log.info("=" * 55)
    log.info("  Output directory : %s", output_dir.resolve())
    log.info("  Best model       : best_model.pt")
    log.info("  Val F1           : %.4f", val_metrics["f1"])
    log.info("  Val AUC          : %.4f", val_metrics["roc_auc"])
    if has_test:
        log.info("  Test F1          : %.4f", te_metrics["f1"])
        log.info("  Test AUC         : %.4f", te_metrics["roc_auc"])
    log.info("=" * 55)


if __name__ == "__main__":
    main()
