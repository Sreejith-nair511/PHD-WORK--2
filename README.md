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
git clone https://github.com/your-lab/dg-hmcf.git
cd dg-hmcf

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install as package (editable mode)
pip install -e .
```

---

## Supported Datasets

| Dataset   | Modalities             | Subjects | Labels             |
|-----------|------------------------|----------|--------------------|
| DAIC-WOZ  | Speech, Text, Face     | 189      | PHQ-8, binary      |
| MODMA     | Speech, EEG            | 55       | Binary + PHQ-8 est.|
| PDCH      | Speech, Text, Face, EEG| Custom   | PHQ-8, binary      |

Access to DAIC-WOZ requires a [license agreement](https://dcapswoz.ict.usc.edu/).

---

## Quick Start

### 1. Preprocess Data

```bash
python scripts/preprocess_data.py \
    --dataset daic_woz \
    --data_root data/raw/daic_woz \
    --output_dir data/processed/daic_woz \
    --modalities speech text face \
    --n_jobs 4
```

### 2. Train

```bash
python scripts/train.py \
    --config configs/daic_woz_config.yaml \
    --dataset daic_woz \
    --data_root data/raw/daic_woz \
    --output_dir outputs/daic_woz_run1 \
    --device cuda
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint outputs/daic_woz_run1/checkpoints/best_model.pt \
    --config configs/daic_woz_config.yaml \
    --dataset daic_woz \
    --data_root data/raw/daic_woz \
    --output_dir outputs/eval_daic_woz \
    --per_modality
```

### 4. Ablation Study

```bash
python scripts/run_ablation.py \
    --config configs/daic_woz_config.yaml \
    --dataset daic_woz \
    --checkpoint outputs/daic_woz_run1/checkpoints/best_model.pt \
    --output_dir outputs/ablation_daic_woz
```

---

## Configuration

All hyperparameters are controlled via YAML files in `configs/`.  
Dataset-specific configs extend `base_config.yaml`.

Key parameters in `base_config.yaml`:

```yaml
model:
  fusion_dim: 512           # Fused representation dimension
  cross_modal:
    n_heads: 8              # Cross-attention heads
    n_layers: 2             # Layers per cross-attention pair
  gating:
    temperature: 1.0        # Softmax temperature for reliability weights

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
├── configs/            YAML configuration files
├── data/
│   ├── preprocessing/  Per-modality preprocessors
│   ├── datasets/       PyTorch Dataset classes
│   └── utils/          Collation, masking utilities
├── models/
│   ├── branches/       Modality-specific encoders
│   ├── modules/        Core fusion modules
│   ├── classifier.py   Multi-task output head
│   └── dg_hmcf.py      Main model class
├── training/           Loss functions, metrics, trainer
├── evaluation/         Evaluator, ablation study
├── utils/              Logger, visualisation
├── scripts/            CLI entry points
└── notebooks/          Demo notebook
```

---

## Explainability

DG-HMCF includes a built-in explainability module that outputs:
- **Modality importance scores** per sample
- **Cross-modal attention heatmaps**
- **Reliability weight visualisations**
- **Natural language summary reports**

```python
report = model.predict(batch)["reports"][0]
print(report["summary"])
# → "The model predicts 'Depressed' (confidence: 84.3%) with estimated PHQ-8
#    score 14.2 (Moderate depression). Most influential modality: speech."
```

---

## Citation

```bibtex
@article{dghmcf2024,
  title   = {DG-HMCF: Dynamic Gated Hierarchical Multi-Scale Cross-Modal
             Fusion for Automated Depression Detection},
  author  = {Research Team},
  journal = {arXiv preprint},
  year    = {2024}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
