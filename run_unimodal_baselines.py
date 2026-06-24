"""
run_unimodal_baselines.py — Unimodal Baseline Experiments for DG-HMCF
======================================================================
Trains two standalone unimodal baselines on DAIC-WOZ:
  1. RoBERTa + BiLSTM  (Text Only)
  2. Wav2Vec2 + BiLSTM (Speech Only)

Identical hyperparameters to run_experiment.py (multimodal DG-HMCF):
  batch_size=8, lr=2e-5, AdamW, CosineAnnealingLR, dropout=0.3,
  early stopping patience=7, max 50 epochs, seed=42

One-command execution:
    python run_unimodal_baselines.py --data_root data/raw/daic_woz --output_dir outputs/baselines

All feature caches from run_experiment.py are reused automatically.

Outputs (all in --output_dir):
  roberta_bilstm_training_log.csv
  wav2vec2_bilstm_training_log.csv
  roberta_bilstm_results.csv
  wav2vec2_bilstm_results.csv
  roberta_bilstm_training_curves.png
  wav2vec2_bilstm_training_curves.png
  roberta_bilstm_confusion_matrix.png
  wav2vec2_bilstm_confusion_matrix.png
  roberta_bilstm_roc_curve.png
  wav2vec2_bilstm_roc_curve.png
  baseline_comparison_unimodal.csv
  baseline_comparison_unimodal.png
"""

import os, sys, json, logging, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Baselines")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — ARGS
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Unimodal Baseline Experiments — DAIC-WOZ")
    p.add_argument("--data_root",    default="data/raw/daic_woz")
    p.add_argument("--output_dir",   default="outputs/baselines")
    p.add_argument("--cache_dir",    default=None,
                   help="Path to existing feature cache (default: output_dir/feature_cache). "
                        "Point to outputs/experiment_1/feature_cache to reuse DG-HMCF caches.")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--max_audio_s",  type=float, default=30.0)
    p.add_argument("--text_model",   default="roberta-base")
    p.add_argument("--speech_model", default="facebook/wav2vec2-base-960h")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default=None)
    p.add_argument("--dghmcf_results", default=None,
                   help="Path to DG-HMCF test_metrics.json for comparison table. "
                        "E.g. outputs/experiment_1/test_metrics.json")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING  (identical to run_experiment.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_labels(data_root: Path) -> dict:
    splits = {}
    label_files = {
        "train": "train_split_Depression_AVEC2017.csv",
        "dev":   "dev_split_Depression_AVEC2017.csv",
        "test":  "test_split_Depression_AVEC2017.csv",
    }
    for split, fname in label_files.items():
        fpath = data_root / fname
        if not fpath.exists():
            log.warning("Label file not found: %s", fpath)
            continue
        df = pd.read_csv(fpath)
        df.columns = [c.strip() for c in df.columns]
        rename_map = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "_").replace("-", "_")
            if "participant" in lc:
                rename_map[col] = "Participant_ID"
            elif "phq8_score" in lc or "phq_score" in lc:
                rename_map[col] = "PHQ8_Score"
            elif "phq8_binary" in lc or "phq_binary" in lc:
                rename_map[col] = "PHQ8_Binary"
        df = df.rename(columns=rename_map)
        if "PHQ8_Binary" not in df.columns and "PHQ8_Score" in df.columns:
            df["PHQ8_Binary"] = (df["PHQ8_Score"] >= 10).astype(int)
        splits[split] = df
        dep = int(df["PHQ8_Binary"].sum()) if "PHQ8_Binary" in df.columns else -1
        log.info("Loaded %s: %d participants (%d depressed)", split, len(df), dep)
    return splits


def load_transcript(data_root: Path, pid: int) -> str:
    path = data_root / str(pid) / f"{pid}_TRANSCRIPT.csv"
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["start", "stop", "speaker", "value"],
                         on_bad_lines="skip")
        rows = df[~df["speaker"].str.strip().str.lower().isin(
            ["ellie", "interviewer", "e"])]["value"].dropna()
        return " ".join(rows.astype(str).tolist()).strip()
    except Exception as e:
        log.debug("Transcript error %d: %s", pid, e)
        return ""


def load_audio_path(data_root: Path, pid: int):
    for suffix in [f"{pid}_AUDIO.wav", f"{pid}_P.wav"]:
        p = data_root / str(pid) / suffix
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE EXTRACTION  (identical to run_experiment.py)
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_embeddings(texts, model_name, device, batch_size=8):
    from transformers import AutoTokenizer, AutoModel
    log.info("Loading text model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device).eval()
    chunk_size, stride = 512, 256
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        for text in texts[i: i + batch_size]:
            if not text or not text.strip():
                embeddings.append(np.zeros(model.config.hidden_size, dtype=np.float32))
                continue
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(tokens) <= chunk_size - 2:
                enc = tokenizer(text, return_tensors="pt",
                                max_length=chunk_size, truncation=True,
                                padding=True).to(device)
                with torch.no_grad():
                    out = model(**enc)
                emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            else:
                chunk_embs = []
                for start in range(0, len(tokens), stride):
                    chunk = tokens[start: start + chunk_size - 2]
                    if not chunk:
                        break
                    ct  = tokenizer.decode(chunk, skip_special_tokens=True)
                    enc = tokenizer(ct, return_tensors="pt",
                                    max_length=chunk_size, truncation=True,
                                    padding=True).to(device)
                    with torch.no_grad():
                        out = model(**enc)
                    chunk_embs.append(
                        out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy())
                emb = np.mean(chunk_embs, axis=0)
            embeddings.append(emb.astype(np.float32))
    return np.array(embeddings, dtype=np.float32)


def extract_speech_embeddings(audio_paths, model_name, device, max_audio_s=30.0):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    log.info("Loading speech model: %s", model_name)
    processor   = Wav2Vec2Processor.from_pretrained(model_name)
    model       = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()
    TARGET_SR   = 16000
    max_frames  = int(TARGET_SR * max_audio_s)
    hidden_size = model.config.hidden_size
    try:
        import librosa
        USE_LIBROSA = True
    except ImportError:
        import torchaudio
        USE_LIBROSA = False
    embeddings = []
    for path in tqdm(audio_paths, desc="Speech embeddings"):
        if path is None:
            embeddings.append(np.zeros(hidden_size, dtype=np.float32))
            continue
        try:
            if USE_LIBROSA:
                import librosa
                audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
            else:
                import torchaudio
                wv, sr = torchaudio.load(str(path))
                if sr != TARGET_SR:
                    wv = torchaudio.functional.resample(wv, sr, TARGET_SR)
                audio = wv.mean(0).numpy()
            audio  = audio[:max_frames]
            inputs = processor(audio, sampling_rate=TARGET_SR,
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            emb = out.last_hidden_state.squeeze(0).mean(0).cpu().numpy()
            embeddings.append(emb.astype(np.float32))
        except Exception as e:
            log.debug("Speech error %s: %s", path, e)
            embeddings.append(np.zeros(hidden_size, dtype=np.float32))
    return np.array(embeddings, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE CACHE
# ═══════════════════════════════════════════════════════════════════════════

def cache_path(cache_dir: Path, split: str, modality: str) -> Path:
    return cache_dir / f"{split}_{modality}.npy"

def load_cache(path: Path):
    if path.exists():
        log.info("  Using cache: %s", path.name)
        return np.load(str(path), allow_pickle=False)
    return None

def save_cache(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)
    log.info("  Saved cache: %s", path.name)


def prepare_features(split_name, split_df, data_root, args, device, cache_dir):
    """Load or extract text + speech features for one split."""
    pids   = split_df["Participant_ID"].tolist()
    labels = np.array(split_df["PHQ8_Binary"].tolist(), dtype=np.int64)

    # ── Text ──────────────────────────────────────────────────────────────
    tcp = cache_path(cache_dir, split_name, "text")
    text_embs = load_cache(tcp)
    if text_embs is None:
        log.info("[%s] Extracting text features...", split_name)
        texts     = [load_transcript(data_root, pid)
                     for pid in tqdm(pids, desc="  Transcripts")]
        text_embs = extract_text_embeddings(texts, args.text_model, device)
        save_cache(text_embs, tcp)

    # ── Speech ────────────────────────────────────────────────────────────
    scp  = cache_path(cache_dir, split_name, "speech")
    smcp = cache_path(cache_dir, split_name, "speech_mask")
    speech_embs  = load_cache(scp)
    speech_masks = load_cache(smcp)

    if speech_embs is None:
        log.info("[%s] Extracting speech features...", split_name)
        audio_paths  = [load_audio_path(data_root, pid) for pid in pids]
        missing      = sum(1 for p in audio_paths if p is None)
        if missing:
            log.warning("[%s] %d/%d audio files missing", split_name, missing, len(pids))
        speech_masks = np.array(
            [1.0 if p is not None else 0.0 for p in audio_paths], dtype=np.float32)
        speech_embs  = extract_speech_embeddings(
            audio_paths, args.speech_model, device, args.max_audio_s)
        save_cache(speech_embs,  scp)
        save_cache(speech_masks, smcp)
    elif speech_masks is None:
        speech_masks = np.ones(len(pids), dtype=np.float32)

    return text_embs, speech_embs, speech_masks, labels, pids


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — UNIMODAL MODELS
# ═══════════════════════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    """
    Unimodal baseline: Linear projection → BiLSTM → mean-pool → Classifier.

    Treats the single (B, embed_dim) patient-level embedding as a
    1-step sequence so BiLSTM produces a contextualised hidden state,
    then adds a skip-connection back to the projected embedding before
    the classifier head.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 bilstm_hidden: int = 128, bilstm_layers: int = 2,
                 dropout: float = 0.3, n_classes: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        bilstm_out_dim = bilstm_hidden * 2  # bidirectional

        # Residual projection to match dims
        self.residual_proj = nn.Linear(hidden_dim, bilstm_out_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(bilstm_out_dim),
            nn.Linear(bilstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif isinstance(self.classifier[-1], nn.Linear) and "weight" in name:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, input_dim)  — patient-level embedding
        Returns logits (B, n_classes)
        """
        proj  = self.input_proj(x)               # (B, hidden_dim)
        seq   = proj.unsqueeze(1)                 # (B, 1, hidden_dim) — 1-step seq
        lstm_out, _ = self.bilstm(seq)            # (B, 1, bilstm_out_dim)
        lstm_out    = lstm_out.squeeze(1)         # (B, bilstm_out_dim)

        # Residual connection
        out = lstm_out + self.residual_proj(proj) # (B, bilstm_out_dim)
        return self.classifier(out)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATASET
# ═══════════════════════════════════════════════════════════════════════════

class UnimodalDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, pids: list):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels     = torch.tensor(labels,     dtype=torch.long)
        self.pids       = pids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label":     self.labels[idx],
            "pid":       self.pids[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRAINING & EVALUATION LOOPS
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        emb = batch["embedding"].to(device)
        lbl = batch["label"].to(device)
        optimizer.zero_grad()
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                logits = model(emb)
                loss   = criterion(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(emb)
            loss   = criterion(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * lbl.size(0)
        correct    += (logits.argmax(1) == lbl).sum().item()
        total      += lbl.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    all_preds, all_probs, all_labels, all_pids = [], [], [], []
    total_loss, total = 0.0, 0
    for batch in loader:
        emb = batch["embedding"].to(device)
        lbl = batch["label"].to(device)
        logits = model(emb)
        loss   = criterion(logits, lbl)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * lbl.size(0)
        total      += lbl.size(0)
        all_preds .extend(logits.argmax(1).cpu().tolist())
        all_probs .extend(probs.cpu().tolist())
        all_labels.extend(lbl.cpu().tolist())
        all_pids  .extend(batch["pid"])
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / total
    return metrics, all_preds, all_probs, all_labels, all_pids


def compute_metrics(labels, preds, probs):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report
    )
    labels = np.array(labels);  preds = np.array(preds);  probs = np.array(probs)
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    cm     = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=["Not Depressed", "Depressed"], zero_division=0)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc,
                confusion_matrix=cm.tolist(), classification_report=report)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — PLOTTING  (publication-ready IEEE style)
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# IEEE-compatible style defaults
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

COLORS = {
    "train":   "#2166ac",
    "val":     "#d6604d",
    "f1":      "#1a9641",
    "acc":     "#756bb1",
    "auc":     "#e08214",
}


def plot_training_curves(history: dict, model_name: str, out_path: Path):
    """
    4-panel training curves figure:
    Loss | Accuracy | F1 | ROC-AUC
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.suptitle(f"{model_name} — Training Curves (DAIC-WOZ)",
                 fontsize=12, fontweight="bold", y=1.01)

    # Loss
    axes[0].plot(epochs, history["train_loss"], color=COLORS["train"],
                 lw=1.8, label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   color=COLORS["val"],
                 lw=1.8, label="Val",   marker="s", markersize=3)
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["val_accuracy"], color=COLORS["acc"],
                 lw=1.8, marker="s", markersize=3)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # F1
    axes[2].plot(epochs, history["val_f1"], color=COLORS["f1"],
                 lw=1.8, marker="^", markersize=3)
    axes[2].set_title("Validation F1 Score")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1")
    axes[2].set_ylim(0, 1)

    # AUC
    axes[3].plot(epochs, history["val_roc_auc"], color=COLORS["auc"],
                 lw=1.8, marker="D", markersize=3)
    axes[3].set_title("Validation ROC-AUC")
    axes[3].set_xlabel("Epoch"); axes[3].set_ylabel("AUC")
    axes[3].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()
    log.info("Saved training curves → %s", out_path)


def plot_confusion_matrix(cm_data, model_name, out_path):
    import seaborn as sns
    cm = np.array(cm_data)
    # Compute percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=["Not Dep.", "Depressed"],
                yticklabels=["Not Dep.", "Depressed"],
                linewidths=0.5, linecolor="gray", ax=ax)
    # Annotate with count + percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_title(f"{model_name}\nConfusion Matrix", fontweight="bold")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()
    log.info("Saved confusion matrix → %s", out_path)


def plot_roc_curve(labels, probs, model_name, out_path):
    from sklearn.metrics import roc_curve, auc
    labels = np.array(labels); probs = np.array(probs)
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color="#d6604d", lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#d6604d")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name}\nROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()
    log.info("Saved ROC curve → %s", out_path)


def plot_comparison_bar(comparison_df: pd.DataFrame, out_path: Path):
    """
    Publication-ready grouped bar chart comparing models on Accuracy, F1, ROC-AUC.
    """
    metrics  = ["Accuracy", "F1", "ROC_AUC"]
    n_models = len(comparison_df)
    n_metrics = len(metrics)
    x       = np.arange(n_metrics)
    width   = 0.22
    colors  = ["#4393c3", "#d6604d", "#1a9641", "#756bb1"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        offset = (i - n_models / 2 + 0.5) * width
        vals   = [row[m] for m in metrics]
        bars   = ax.bar(x + offset, vals, width, label=row["Model"],
                        color=colors[i % len(colors)], edgecolor="white",
                        linewidth=0.6, zorder=3)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "F1 Score", "ROC-AUC"], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Unimodal Baselines vs. DG-HMCF — DAIC-WOZ Test Set",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    # Caption note
    fig.text(0.5, -0.04,
             "DAIC-WOZ test set. No oversampling or augmentation. "
             "Best val F1 checkpoint. Seed=42.",
             ha="center", fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()
    log.info("Saved comparison chart → %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — CORE TRAINING RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_baseline(
    model_tag: str,           # "roberta_bilstm" or "wav2vec2_bilstm"
    display_name: str,        # "RoBERTa-BiLSTM" etc.
    train_embs:    np.ndarray,
    val_embs:      np.ndarray,
    test_embs:     np.ndarray,
    train_labels:  np.ndarray,
    val_labels:    np.ndarray,
    test_labels:   np.ndarray,
    train_pids:    list,
    val_pids:      list,
    test_pids:     list,
    args,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """
    Full train → validate → test cycle for one unimodal baseline.
    Returns dict with val and test metrics.
    """
    log.info("\n" + "═" * 60)
    log.info("  BASELINE: %s", display_name)
    log.info("  Input dim: %d | Train: %d | Val: %d | Test: %d",
             train_embs.shape[1], len(train_labels),
             len(val_labels), len(test_labels))
    log.info("═" * 60)

    # ── Class weights ──────────────────────────────────────────────────────
    counts       = np.bincount(train_labels)
    class_weights = torch.tensor(1.0 / (counts + 1e-6),
                                  dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = UnimodalDataset(train_embs, train_labels, train_pids)
    val_ds   = UnimodalDataset(val_embs,   val_labels,   val_pids)
    test_ds  = UnimodalDataset(test_embs,  test_labels,  test_pids)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────
    input_dim = train_embs.shape[1]
    model = BiLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        bilstm_hidden=128,
        bilstm_layers=2,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Trainable params: %d (%.2fM)", n_params, n_params / 1e6)

    # ── Optimizer & scheduler (same as DG-HMCF) ────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Training loop ─────────────────────────────────────────────────────
    history = {k: [] for k in [
        "train_loss", "val_loss", "val_accuracy",
        "val_precision", "val_recall", "val_f1", "val_roc_auc"
    ]}
    best_val_f1   = -1.0
    best_epoch    = 0
    best_ckpt     = output_dir / f"{model_tag}_best.pt"
    no_improve    = 0
    patience      = 7  # identical to run_experiment.py

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_m, _, val_probs, val_lbls, _ = eval_model(
            model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"]    .append(tr_loss)
        history["val_loss"]      .append(val_m["loss"])
        history["val_accuracy"]  .append(val_m["accuracy"])
        history["val_precision"] .append(val_m["precision"])
        history["val_recall"]    .append(val_m["recall"])
        history["val_f1"]        .append(val_m["f1"])
        history["val_roc_auc"]   .append(val_m["roc_auc"])

        is_best = val_m["f1"] > best_val_f1
        marker  = " ✓" if is_best else ""
        log.info(
            "  Ep %3d/%d | tr_loss=%.4f | val_loss=%.4f "
            "acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f%s",
            epoch, args.epochs, tr_loss, val_m["loss"],
            val_m["accuracy"], val_m["precision"], val_m["recall"],
            val_m["f1"], val_m["roc_auc"], marker)

        if is_best:
            best_val_f1 = val_m["f1"]
            best_epoch  = epoch
            no_improve  = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": val_m}, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stopping at epoch %d (patience=%d)",
                         epoch, patience)
                break

    # ── Load best and evaluate ─────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("  Best checkpoint: epoch %d (val F1=%.4f)", best_epoch, best_val_f1)

    val_m,  val_preds,  val_probs,  val_lbls,  val_pid_list  = eval_model(
        model, val_loader,  criterion, device)
    test_m, test_preds, test_probs, test_lbls, test_pid_list = eval_model(
        model, test_loader, criterion, device)

    # ── Save training log CSV ──────────────────────────────────────────────
    log_df = pd.DataFrame({
        "epoch":      range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss":   history["val_loss"],
        "accuracy":   history["val_accuracy"],
        "precision":  history["val_precision"],
        "recall":     history["val_recall"],
        "f1":         history["val_f1"],
        "roc_auc":    history["val_roc_auc"],
    })
    log_csv = output_dir / f"{model_tag}_training_log.csv"
    log_df.to_csv(log_csv, index=False, float_format="%.6f")
    log.info("  Saved training log → %s", log_csv)

    # ── Save results CSV ───────────────────────────────────────────────────
    for split_name, m in [("val", val_m), ("test", test_m)]:
        res_df = pd.DataFrame([{
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "roc_auc":   m["roc_auc"],
        }])
        res_csv = output_dir / f"{model_tag}_{split_name}_results.csv"
        res_df.to_csv(res_csv, index=False, float_format="%.6f")

        pred_df = pd.DataFrame({
            "participant_id":  test_pid_list if split_name == "test" else val_pid_list,
            "true_label":      test_lbls    if split_name == "test" else val_lbls,
            "predicted_label": test_preds   if split_name == "test" else val_preds,
            "prob_depressed":  test_probs   if split_name == "test" else val_probs,
        })
        pred_df.to_csv(
            output_dir / f"{model_tag}_{split_name}_predictions.csv",
            index=False, float_format="%.6f")

        with open(output_dir / f"{model_tag}_{split_name}_classification_report.txt",
                  "w") as f:
            f.write(f"=== {display_name} — {split_name.upper()} ===\n\n")
            f.write(m["classification_report"])

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_training_curves(history, display_name,
                         output_dir / f"{model_tag}_training_curves.png")
    plot_confusion_matrix(test_m["confusion_matrix"], display_name,
                          output_dir / f"{model_tag}_confusion_matrix.png")
    plot_roc_curve(test_lbls, test_probs, display_name,
                   output_dir / f"{model_tag}_roc_curve.png")

    # ── Print summary ──────────────────────────────────────────────────────
    log.info("\n  ── %s TEST RESULTS ──", display_name)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        log.info("    %-12s: %.4f", k.capitalize(), test_m[k])

    return {"val": val_m, "test": test_m}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s (%.1f GB)",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Paths ───────────────────────────────────────────────────────────────
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature cache: use shared cache if specified, otherwise local
    cache_dir = Path(args.cache_dir) if args.cache_dir \
                else output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        log.error("data_root not found: %s", data_root)
        sys.exit(1)

    # ── Labels ──────────────────────────────────────────────────────────────
    splits = load_labels(data_root)
    for sp in ["train", "dev"]:
        if sp not in splits:
            log.error("Missing required split: '%s'", sp)
            sys.exit(1)
    has_test = "test" in splits

    # ── Feature extraction (shared across both baselines) ───────────────────
    log.info("\n── Loading / extracting features ──")
    tr_text, tr_speech, tr_smask, tr_labels, tr_pids = prepare_features(
        "train", splits["train"], data_root, args, device, cache_dir)
    va_text, va_speech, va_smask, va_labels, va_pids = prepare_features(
        "dev",   splits["dev"],   data_root, args, device, cache_dir)

    if has_test:
        te_text, te_speech, te_smask, te_labels, te_pids = prepare_features(
            "test", splits["test"], data_root, args, device, cache_dir)
    else:
        log.warning("No test split found — using dev set as test proxy.")
        te_text, te_speech, te_smask  = va_text, va_speech, va_smask
        te_labels, te_pids            = va_labels, va_pids

    # ══════════════════════════════════════════════════════════════════════
    # BASELINE 1 — RoBERTa + BiLSTM  (Text Only)
    # ══════════════════════════════════════════════════════════════════════
    roberta_results = run_baseline(
        model_tag    = "roberta_bilstm",
        display_name = "RoBERTa-BiLSTM",
        train_embs   = tr_text,  val_embs   = va_text,  test_embs   = te_text,
        train_labels = tr_labels, val_labels = va_labels, test_labels = te_labels,
        train_pids   = tr_pids,  val_pids   = va_pids,  test_pids   = te_pids,
        args=args, device=device, output_dir=output_dir,
    )

    # ══════════════════════════════════════════════════════════════════════
    # BASELINE 2 — Wav2Vec2 + BiLSTM  (Speech Only)
    # ══════════════════════════════════════════════════════════════════════
    wav2vec2_results = run_baseline(
        model_tag    = "wav2vec2_bilstm",
        display_name = "Wav2Vec2-BiLSTM",
        train_embs   = tr_speech, val_embs  = va_speech, test_embs  = te_speech,
        train_labels = tr_labels, val_labels = va_labels, test_labels = te_labels,
        train_pids   = tr_pids,  val_pids   = va_pids,  test_pids   = te_pids,
        args=args, device=device, output_dir=output_dir,
    )

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n── Building comparison table ──")

    # Try to load DG-HMCF results from file if provided
    dghmcf_row = {"Model": "DG-HMCF (Text+Speech)",
                  "Accuracy": float("nan"), "Precision": float("nan"),
                  "Recall": float("nan"), "F1": float("nan"),
                  "ROC_AUC": float("nan")}

    if args.dghmcf_results:
        p = Path(args.dghmcf_results)
        if p.exists():
            with open(p) as f:
                dm = json.load(f)
            dghmcf_row = {
                "Model":     "DG-HMCF (Text+Speech)",
                "Accuracy":  dm.get("accuracy",  float("nan")),
                "Precision": dm.get("precision", float("nan")),
                "Recall":    dm.get("recall",    float("nan")),
                "F1":        dm.get("f1",        float("nan")),
                "ROC_AUC":   dm.get("roc_auc",   float("nan")),
            }
            log.info("  Loaded DG-HMCF results from %s", p)
        else:
            log.warning("  DG-HMCF results file not found: %s — NaN placeholders used", p)

    def _row(name, m):
        tm = m["test"]
        return {"Model":     name,
                "Accuracy":  tm["accuracy"],
                "Precision": tm["precision"],
                "Recall":    tm["recall"],
                "F1":        tm["f1"],
                "ROC_AUC":   tm["roc_auc"]}

    rows = [
        _row("RoBERTa-BiLSTM",   roberta_results),
        _row("Wav2Vec2-BiLSTM",  wav2vec2_results),
        dghmcf_row,
    ]
    comparison_df = pd.DataFrame(rows)

    cmp_csv = output_dir / "baseline_comparison_unimodal.csv"
    comparison_df.to_csv(cmp_csv, index=False, float_format="%.4f")
    log.info("  Saved comparison table → %s", cmp_csv)

    # Print table to console
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 100)
    log.info("\n%s\n", comparison_df.to_string(index=False))

    # Comparison bar chart
    plot_comparison_bar(comparison_df, output_dir / "baseline_comparison_unimodal.png")

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  EXPERIMENT COMPLETE")
    log.info("  Output directory: %s", output_dir.resolve())
    log.info("  Files generated:")
    for f in sorted(output_dir.glob("*.csv")):
        log.info("    %s", f.name)
    for f in sorted(output_dir.glob("*.png")):
        log.info("    %s", f.name)
    log.info("═" * 60)


if __name__ == "__main__":
    main()
