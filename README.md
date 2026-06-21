# DG-HMCF: Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion for Depression Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

DG-HMCF is a multi-modal deep learning framework for automated depression detection that fuses **speech**, **text**, **facial expression**, and **EEG** signals using a hierarchical cross-modal transformer with dynamic reliability gating.

### Key Contributions

- **Dynamic Reliability Gating** — learns per-sample, per-modality reliability weights to handle noisy or missing modalities automatically.
- **Multi-Scale Temporal Fusion** — parallel dilated convolutions at 3 timescales capture short-term prosodic and long-term contextual patterns.
- **Hierarchical Cross-Modal Transformer** — 6 pairwise cross-attention stages progressively transfer information between all modality pairs.
- **Adaptive Fusion Layer** — combines reliability-weighted, cross-modal-enhanced, and residual streams through learned gates.
- **Missing Modality Handler** — synthesises proxy embeddings for absent modalities so the full pipeline remains active regardless of data availability.
- **Multi-Task Head** — jointly predicts the binary depression label and the PHQ-8 continuous score.

---

## Architecture

```
Speech  ──► SpeechBranch (Wav2Vec2 + BiLSTM)  ──┐
Text    ──► TextBranch (RoBERTa + BiLSTM)      ──┤
Face    ──► FaceBranch (ViT + Temporal Attn)   ──┤─► Missing Modality Handler
EEG     ──► EEGBranch (CNN + BiLSTM)           ──┘        │
                                                           │
                              Multi-Scale Temporal Fusion (per modality)
                                                           │
                              Dynamic Reliability Gating ──┘
                                                           │
                        Hierarchical Cross-Modal Transformer
                        (speech↔text, speech↔face, ...)   │
                                                           │
                              Adaptive Fusion Layer        │
                                                           │
                    ┌──────────────────────────────────────┘
                    │
             DepressionClassifier
                    ├─► Binary Label (depressed / non-depressed)
                    └─► PHQ-8 Score Regression
```

---

## Installation

```bash
git clone https://github.com/Sreejith-nair511/PHD-WORK--2.git
cd PHD-WORK--2

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .
```

---

## Dataset Setup

> **Important:** Raw datasets are NOT included in this repository (file sizes, licensing).  
> You must download each dataset separately and place it in the exact folder structure shown below.  
> The `data/raw/` directory is git-ignored — nothing you put there will be committed.

---

### Directory Layout (create these folders first)

```
DG-HMCF/
└── data/
    ├── raw/                   ← place all downloaded datasets here
    │   ├── daic_woz/          ← DAIC-WOZ files go here
    │   ├── modma/             ← MODMA files go here
    │   └── pdch/              ← PDCH files go here
    └── processed/             ← auto-created by preprocessing script
        ├── daic_woz/
        ├── modma/
        └── pdch/
```

Create the folders manually:

```bash
# Windows
mkdir data\raw\daic_woz
mkdir data\raw\modma
mkdir data\raw\pdch

# Linux / macOS
mkdir -p data/raw/daic_woz data/raw/modma data/raw/pdch
```

---

### Dataset 1 — DAIC-WOZ

**Modalities used:** Speech · Text · Face  
**Access:** Requires a signed license agreement (free for research)

#### How to get it
1. Go to → **https://dcapswoz.ict.usc.edu/**
2. Fill in the data request form with your university/institution details
3. You will receive a download link by email (usually within a few days)
4. Download the full corpus ZIP and extract it

#### Expected folder structure after extraction

```
data/raw/daic_woz/
├── train_split_Depression_AVEC2017.csv     ← train labels (PHQ-8 scores)
├── dev_split_Depression_AVEC2017.csv       ← dev/val labels
├── test_split_Depression_AVEC2017.csv      ← test labels
│
├── 300/                    ← one folder per participant (numbered 300–492)
│   ├── 300_AUDIO.wav       ← raw interview audio
│   ├── 300_TRANSCRIPT.csv  ← word-level transcript with speaker labels and timestamps
│   └── 300_CLNF_features3D.txt  ← OpenFace facial landmarks
│
├── 301/
│   ├── 301_AUDIO.wav
│   ├── 301_TRANSCRIPT.csv
│   └── 301_CLNF_features3D.txt
│
└── ...  (189 participant folders total)
```

#### Key files explained

| File | What it contains |
|------|-----------------|
| `*_AUDIO.wav` | Full interview audio (participant + interviewer "Ellie") |
| `*_TRANSCRIPT.csv` | Columns: `start_time`, `stop_time`, `speaker`, `value` — filter `speaker != Ellie` |
| `*_CLNF_features3D.txt` | Per-frame facial action units and head pose from OpenFace |
| `*_split_Depression_AVEC2017.csv` | Columns: `Participant_ID`, `PHQ8_Score`, `PHQ8_Binary` |

#### Config file to use
```bash
--config configs/daic_woz_config.yaml
--data_root data/raw/daic_woz
```

---

### Dataset 2 — MODMA

**Modalities used:** Speech · EEG  
**Access:** Publicly available, no license required

#### How to get it
1. Go to → **http://modma.lzu.edu.cn/data/index/**
2. Register a free account
3. Download the dataset (EEG `.mat` files + speech `.wav` files)

#### Expected folder structure after extraction

```
data/raw/modma/
├── labels.csv              ← subject ID, depression label, clinical scores
│
├── EEG/
│   ├── subject_001.mat     ← EEG recording (128 channels, 250 Hz, MATLAB format)
│   ├── subject_002.mat
│   └── ...  (55 subjects total)
│
└── Speech/
    ├── subject_001/
    │   ├── task1.wav       ← speech recording for task 1
    │   └── task2.wav
    ├── subject_002/
    └── ...
```

#### Key files explained

| File | What it contains |
|------|-----------------|
| `labels.csv` | Columns: `subject_id`, `label` (0=control, 1=depressed), `HAMD_score` |
| `EEG/*.mat` | Variable `data`: shape `(n_channels, n_timepoints)`, variable `label` |
| `Speech/subject_*/` | WAV files per subject per task |

#### Config file to use
```bash
--config configs/modma_config.yaml
--data_root data/raw/modma
```

---

### Dataset 3 — PDCH

**Modalities used:** Speech · Text · Face · EEG (all 4)  
**Access:** Confirm access with your institution / dataset authors

#### Expected folder structure

```
data/raw/pdch/
├── metadata.csv            ← subject ID, PHQ-8 score, binary label
│
├── audio/
│   └── subject_001.wav
│
├── transcripts/
│   └── subject_001.txt
│
├── video/
│   └── subject_001.mp4
│
└── eeg/
    └── subject_001.mat     ← or .csv / .edf depending on format
```

#### Config file to use
```bash
--config configs/pdch_config.yaml
--data_root data/raw/pdch
```

---

### Verify Your Data Setup

After placing files, run the check script to confirm everything is found:

```bash
python scripts/preprocess_data.py \
    --dataset daic_woz \
    --data_root data/raw/daic_woz \
    --dry_run
```

A `--dry_run` flag scans the folder structure and reports how many valid participants were found without doing any processing.

---

## Preprocessing

Once datasets are in place, run the preprocessing pipeline. This creates cleaned, feature-extracted `.pkl` files in `data/processed/` that the training script reads directly.

### DAIC-WOZ

```bash
python scripts/preprocess_data.py \
    --dataset daic_woz \
    --data_root data/raw/daic_woz \
    --output_dir data/processed/daic_woz \
    --modalities speech text face \
    --n_jobs 4
```

### MODMA

```bash
python scripts/preprocess_data.py \
    --dataset modma \
    --data_root data/raw/modma \
    --output_dir data/processed/modma \
    --modalities speech eeg \
    --n_jobs 4
```

### PDCH

```bash
python scripts/preprocess_data.py \
    --dataset pdch \
    --data_root data/raw/pdch \
    --output_dir data/processed/pdch \
    --modalities speech text face eeg \
    --n_jobs 4
```

**What preprocessing does per modality:**

| Modality | Processing Steps |
|----------|-----------------|
| **Speech** | Resample to 16 kHz → remove interviewer segments → normalize amplitude → extract 6 behavioral features (speech rate, pause duration, silence ratio, pitch variance, energy variance, response latency) |
| **Text** | Filter interviewer turns → RoBERTa tokenization → extract 5 psycholinguistic features (sentiment, neg-word ratio, uncertainty, self-reference, emotional polarity) |
| **Face** | Extract 1 fps frames → detect & crop face (224×224) → normalize with ImageNet stats → extract 7 behavioral features (smile freq, gaze stability, blink freq, AU1/2/4/6) |
| **EEG** | Bandpass filter 0.5–50 Hz → notch filter 50/60 Hz → z-score normalize → segment into 1-second windows with 50% overlap |

---

## Training

```bash
# DAIC-WOZ — full quad-modal (speech + text + face)
python scripts/train.py \
    --config configs/daic_woz_config.yaml \
    --dataset daic_woz \
    --data_root data/raw/daic_woz \
    --output_dir outputs/daic_woz_run1 \
    --device cuda

# MODMA — speech + EEG
python scripts/train.py \
    --config configs/modma_config.yaml \
    --dataset modma \
    --data_root data/raw/modma \
    --output_dir outputs/modma_run1 \
    --device cuda

# Resume from checkpoint
python scripts/train.py \
    --config configs/daic_woz_config.yaml \
    --dataset daic_woz \
    --data_root data/raw/daic_woz \
    --output_dir outputs/daic_woz_run1 \
    --resume outputs/daic_woz_run1/checkpoints/best_model.pt \
    --device cuda
```

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/daic_woz_run1/checkpoints/best_model.pt \
    --config configs/daic_woz_config.yaml \
    --dataset daic_woz \
    --data_root data/raw/daic_woz \
    --output_dir outputs/eval_daic_woz \
    --per_modality
```

The `--per_modality` flag evaluates performance across all 8 valid modality combinations and prints a comparison table.

---

## Ablation Study

```bash
python scripts/run_ablation.py \
    --config configs/daic_woz_config.yaml \
    --dataset daic_woz \
    --checkpoint outputs/daic_woz_run1/checkpoints/best_model.pt \
    --output_dir outputs/ablation_daic_woz
```

Runs all 4 baselines + 8 ablation variants and saves a results comparison CSV + bar charts.

---

## Configuration

All hyperparameters live in `configs/`. Dataset-specific YAMLs extend `base_config.yaml`.

```yaml
# base_config.yaml — key settings
model:
  fusion_dim: 512
  cross_modal:
    n_heads: 8
    n_layers: 2
  gating:
    temperature: 1.0

training:
  epochs: 50
  learning_rate: 1e-4
  loss:
    classification_weight: 0.5
    regression_weight: 0.5
```

---

## Project Structure

```
DG-HMCF/
├── configs/                    YAML configuration files
│   ├── base_config.yaml
│   ├── daic_woz_config.yaml
│   ├── modma_config.yaml
│   └── pdch_config.yaml
│
├── data/                       Data pipeline
│   ├── raw/                    ← PUT YOUR DATASETS HERE (git-ignored)
│   │   ├── daic_woz/
│   │   ├── modma/
│   │   └── pdch/
│   ├── processed/              ← auto-generated by preprocessing
│   ├── preprocessing/          Per-modality preprocessors
│   ├── datasets/               PyTorch Dataset classes
│   └── utils/                  Collation, masking, split utilities
│
├── models/                     Model code
│   ├── branches/               Modality-specific encoders
│   │   ├── speech_branch.py    Wav2Vec2 + BiLSTM + behavioral features
│   │   ├── text_branch.py      RoBERTa + BiLSTM + linguistic features
│   │   ├── face_branch.py      ViT + temporal attention + behavioral features
│   │   └── eeg_branch.py       1D CNN + BiLSTM
│   ├── modules/                Core novel modules
│   │   ├── dynamic_gating.py   Novel Module 1: Dynamic Reliability Gating
│   │   ├── multiscale_temporal.py  Novel Module 2: Multi-Scale Temporal Fusion
│   │   ├── hierarchical_cross_modal.py  Novel Module 3: HCMT (6 pairs)
│   │   ├── adaptive_fusion.py  Novel Module 4: Adaptive Fusion Layer
│   │   ├── missing_modality.py Novel Module 5: Missing Modality Handler
│   │   └── explainability.py   Novel Module 6: Explainability
│   ├── classifier.py           Multi-task output head
│   └── dg_hmcf.py              Main model orchestrator
│
├── training/
│   ├── trainer.py              Training loop, early stopping, checkpointing
│   ├── losses.py               MultiTaskDepressionLoss + FocalLoss
│   └── metrics.py              F1, AUC, MAE, RMSE, Pearson-r
│
├── evaluation/
│   ├── evaluator.py            Full evaluation + per-modality combination
│   └── ablation.py             4 baselines + 8 ablation variants
│
├── utils/
│   ├── logger.py               Logger with WandB support
│   └── visualization.py        Plots: attention maps, reliability weights, curves
│
├── scripts/
│   ├── train.py                Training entry point
│   ├── evaluate.py             Evaluation entry point
│   ├── run_ablation.py         Ablation study runner
│   └── preprocess_data.py      Data preprocessing pipeline
│
├── notebooks/
│   └── demo.ipynb              Interactive demo with synthetic data
│
├── RESEARCH_NOTES.md           PhD tracking: experiments, publication plan, TODO
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Supported Datasets

| Dataset | Modalities | Subjects | Labels | Access |
|---------|-----------|----------|--------|--------|
| DAIC-WOZ | Speech, Text, Face | 189 | PHQ-8 + binary | License required |
| MODMA | Speech, EEG | 55 | Binary + clinical scales | Free (registration) |
| PDCH | Speech, Text, Face, EEG | Custom | PHQ-8 + binary | Contact authors |

---

## Explainability

DG-HMCF includes a built-in explainability module that outputs:
- **Modality importance scores** per sample
- **Cross-modal attention heatmaps** for all 6 modality pairs
- **Reliability weight visualisations**
- **Natural language summary reports**

```python
from models.dg_hmcf import DGHMCF
import yaml

config = yaml.safe_load(open("configs/base_config.yaml"))["model"]
model = DGHMCF(config)

outputs = model.predict(batch)
print(outputs["reports"][0]["summary"])
# → "The model predicts 'Depressed' (confidence: 84.3%) with estimated PHQ-8
#    score 14.2 (Moderate depression). Most influential modality: speech
#    (weight: 0.48). Speech reliability notably high; EEG reliability low."
```

---

## Citation

```bibtex
@article{dghmcf2026,
  title   = {DG-HMCF: Dynamic Gated Hierarchical Multi-Scale Cross-Modal
             Fusion for Automated Depression Detection},
  author  = {Sreejith Nair},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about the code, open a GitHub Issue.  
For dataset access issues, contact the respective dataset providers directly (links above).
