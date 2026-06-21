# DG-HMCF — PhD Research Notes

> **Project:** Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion for Depression Detection  
> **Type:** PhD Research / Journal Publication  
> **Status:** 🟡 In Progress  
> **Last Updated:** June 2026

---

## Table of Contents

1. [Research Identity](#1-research-identity)
2. [Problem Statement](#2-problem-statement)
3. [Research Gaps Addressed](#3-research-gaps-addressed)
4. [Novelty Claims](#4-novelty-claims)
5. [Architecture Summary](#5-architecture-summary)
6. [Datasets](#6-datasets)
7. [Baselines & Comparisons](#7-baselines--comparisons)
8. [Expected Results](#8-expected-results)
9. [Ablation Study Design](#9-ablation-study-design)
10. [Publication Plan](#10-publication-plan)
11. [Target Journals & Conferences](#11-target-journals--conferences)
12. [Related Work Log](#12-related-work-log)
13. [Experiment Log](#13-experiment-log)
14. [TODO & Milestones](#14-todo--milestones)
15. [Supervisor Notes](#15-supervisor-notes)
16. [PhD Progress Tracker](#16-phd-progress-tracker)
17. [References to Cite](#17-references-to-cite)

---

## 1. Research Identity

| Field | Detail |
|-------|--------|
| **Title** | DG-HMCF: Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion for Depression Detection |
| **Short Name** | DG-HMCF |
| **Research Area** | Affective Computing, Mental Health AI, Multimodal Deep Learning |
| **Keywords** | Depression Detection, Cross-Modal Fusion, Dynamic Gating, Multimodal Learning, Missing Modality, PHQ-8, EEG, Speech, NLP |
| **Modalities** | Speech · Text · Face · EEG |
| **Datasets** | DAIC-WOZ · MODMA · PDCH |
| **Framework** | PyTorch, Transformers (HuggingFace), Wav2Vec2, RoBERTa, ViT |
| **Candidate** | *(your name)* |
| **Supervisor** | *(supervisor name)* |
| **Institution** | *(university / department)* |
| **Year Started** | *(year)* |

---

## 2. Problem Statement

Depression is a globally prevalent mental health disorder affecting over **280 million people** (WHO, 2023). Traditional clinical diagnosis relies on structured interviews and self-report questionnaires (e.g., PHQ-8, PHQ-9, BDI), which are:

- Subjective and prone to patient under-reporting
- Time-consuming and resource-intensive
- Inaccessible in low-resource settings
- Unable to capture continuous behavioral signals

Automated depression detection from multimodal behavioral signals (speech, language, facial expression, EEG) offers a scalable, objective, and continuous monitoring solution. However, current multimodal systems suffer from:

1. **Fixed modality importance** — models treat all modalities equally regardless of data quality
2. **Weak temporal modeling** — single-scale features miss multi-granularity temporal patterns
3. **Shallow cross-modal interaction** — simple concatenation or single cross-attention misses deep inter-modal relationships
4. **Poor missing modality handling** — most models require all modalities at inference
5. **Lack of depression-specific behavioral modeling** — generic audio/visual features are used instead of clinically grounded behavioral markers
6. **No explainability** — black-box predictions are clinically untrustworthy

**Research Question:**  
*Can a dynamic, hierarchical, multi-scale cross-modal fusion architecture that incorporates depression-specific behavioral features, missing modality robustness, and explainability significantly outperform existing fusion methods for automated depression detection?*

---

## 3. Research Gaps Addressed

| Gap | Current Literature | DG-HMCF Solution |
|-----|-------------------|------------------|
| Fixed modality weights | Attention/concat fusion assigns static weights | Dynamic Reliability Gating: per-sample, data-driven weights |
| Single-scale temporal | Most use single window size | Multi-Scale Temporal Fusion: kernels {3, 5, 7} |
| Shallow cross-modal | One cross-attention pass | Hierarchical 6-pair cross-attention transformer |
| Missing modality | Architecture changes needed | Modality masks + learned proxy embeddings, zero arch changes |
| Generic features | Standard MFCC, OpenFace | Depression behavioral features: pause, latency, uncertainty, FAUs |
| No explainability | Post-hoc SHAP or LIME only | Built-in modality importance + attention heatmaps per prediction |

---

## 4. Novelty Claims

### Primary Claim
> DG-HMCF is the first framework to integrate **dynamic per-sample modality reliability gating**, **hierarchical multi-scale cross-modal transformers**, **depression-specific behavioral modeling**, and **native missing-modality support** within a single unified architecture for depression detection.

### Individual Novel Components

**Novel Module 1 — Dynamic Reliability Gating Network**
- Per-modality MLP quality scorer conditioned on embedding content
- Context-aware modulation via global mean of present modalities
- Learnable temperature parameter for sharpness control
- Directly derived reliability weights — not attention proxies

**Novel Module 2 — Multi-Scale Temporal Fusion**
- Parallel Conv1d at 3 timescales (kernel = 3, 5, 7)
- Captures: syllable-level (short) → word-level (mid) → phrase-level (long)
- Inspired by Inception-style multi-scale processing applied to behavioral signals

**Novel Module 3 — Hierarchical Cross-Modal Transformer**
- All 6 modality pairs processed in a fixed hierarchical order
- Speech↔Text → Speech↔Face → Speech↔EEG → Text↔Face → Text↔EEG → Face↔EEG
- Bidirectional cross-attention per pair (A→B and B→A)
- Inspired by MemoCMT but extended to 4 modalities and 6 pairs

**Novel Module 4 — Adaptive Fusion Layer**
- 3-stream fusion: reliability-weighted sum + cross-modal representations + residual
- Learned gates combine all three streams
- Preserves per-modality information while enabling deep fusion

**Novel Module 5 — Missing Modality Learning**
- Supports all 8 valid modality combinations (2^4 - 8 absent combinations)
- Learned proxy zero-vectors substituted for absent modalities
- ModalityDropout augmentation during training for robustness
- Zero architectural changes required at inference

**Novel Module 6 — Explainable Fusion**
- Modality importance scores derived from reliability weights + embedding norms
- Cross-modal attention maps for each of the 6 pairs
- Human-readable natural language report per prediction
- Clinically interpretable outputs (which modality drove the decision)

### Depression-Specific Behavioral Features

| Modality | Features | Clinical Basis |
|----------|----------|----------------|
| Speech | Speech rate, pause duration, silence ratio, pitch variance, energy variance, response latency | Psychomotor retardation markers |
| Text | Sentiment score, negative word ratio, uncertainty score, self-reference frequency, emotional polarity | Cognitive distortion markers |
| Face | Smile frequency, gaze stability, blink frequency, AU1, AU2, AU4, AU6 | Anhedonia and affect flattening markers |
| EEG | Band-filtered segments (δ, θ, α, β, γ bands) | Neurophysiological biomarkers |

---

## 5. Architecture Summary

```
INPUT: Speech | Text | Face | EEG  (any subset)
         ↓       ↓      ↓     ↓
   [Wav2Vec2] [RoBERTa] [ViT] [CNN]
   [BiLSTM]  [BiLSTM]         [BiLSTM]
   [+Behav.] [+Ling.]  [+Beh.]
         ↓       ↓      ↓     ↓
   ──── Missing Modality Handler ────
         ↓       ↓      ↓     ↓
   ──── Multi-Scale Temporal Fusion ──── (per modality, kernels 3/5/7)
         ↓       ↓      ↓     ↓
   ─── Dynamic Reliability Gating ───  → [w_s, w_t, w_f, w_e]
         ↓
   Hierarchical Cross-Modal Transformer
   (6 cross-attention pairs)
         ↓
   Adaptive Fusion Layer
   (3-stream: weighted + cross-modal + residual)
         ↓
   DepressionClassifier
   ├── Classification Head → Binary Label (Depressed / Not)
   └── Regression Head    → PHQ-8 Score [0–24]
         ↓
   Explainability Module → modality importance, attention maps, report
```

---

## 6. Datasets

### DAIC-WOZ (Distress Analysis Interview Corpus – Wizard of Oz)
- **Source:** USC Institute for Creative Technologies
- **Access:** License required — https://dcapswoz.ict.usc.edu/
- **Subjects:** 189 participants (107 train / 35 dev / 47 test)
- **Modalities:** Audio (speech), Transcript (text), Video (face)
- **Labels:** PHQ-8 score (0–24) + binary (threshold = 10)
- **Prevalence:** ~34% depressed
- **Notes:** Interviewer is virtual agent "Ellie". Must remove Ellie's speech before processing.

### MODMA (Multimodal Open Dataset for Mental-disorder Analysis)
- **Source:** Lanzhou University
- **Access:** Public — http://modma.lzu.edu.cn/
- **Subjects:** 55 (24 depressed, 29 controls)
- **Modalities:** EEG (128 channels, 250 Hz), Speech
- **Labels:** Binary depression + clinical scales

### PDCH (to be confirmed)
- **Source:** *(add details when access is confirmed)*
- **Modalities:** Speech, Text, Face, EEG (full quad-modal)
- **Notes:** *(add details)*

### Dataset Preprocessing Notes
- DAIC-WOZ: remove interviewer turns, align transcript with audio timestamps
- EEG: bandpass 0.5–50 Hz, notch at 50/60 Hz, z-score normalize, 1-second segments with 50% overlap
- Face: extract 1 frame/sec, 224×224 center crop, normalize with ImageNet stats
- Audio: 16 kHz, mono, normalize amplitude, max 10 seconds per segment

---

## 7. Baselines & Comparisons

| Model | Modalities | Fusion Type | Notes |
|-------|-----------|-------------|-------|
| **Baseline 1** | Speech + Text | Attention Fusion | RoBERTa + Wav2Vec2 + weighted attention |
| **Baseline 2** | Speech + Text | Cross-Attention | Single cross-attention layer |
| **Baseline 3** | Speech + Text + Face | MemoCMT-style | Inspired by MemoCMT (2023) |
| **Baseline 4** | Speech + EEG | Multi-Scale + BiLSTM | Conv multi-scale temporal only |
| **DG-HMCF (Ours)** | Speech + Text + Face + EEG | Full DG-HMCF | All 6 novel modules |

### Ablation Variants

| Variant | Removed Module | Purpose |
|---------|---------------|---------|
| w/o DRG | No dynamic gating (uniform weights) | Ablate reliability gating |
| w/o MSTF | No multi-scale fusion (single kernel) | Ablate temporal fusion |
| w/o HCMT | No cross-modal transformer | Ablate cross-modal interaction |
| w/o AFL | Simple concat instead of adaptive fusion | Ablate fusion layer |
| w/o MM | No missing modality handler | Ablate robustness |
| w/o BF | No behavioral features | Ablate depression-specific features |
| w/o Exp | No explainability module | Ablate explainability |
| **Full DG-HMCF** | — | Complete proposed model |

---

## 8. Expected Results

### DAIC-WOZ Expected Performance

| Metric | Baseline Best | Expected DG-HMCF |
|--------|--------------|-----------------|
| F1 (depressed class) | ~0.68 (SOTA) | > 0.75 |
| Accuracy | ~0.72 | > 0.78 |
| AUC-ROC | ~0.80 | > 0.85 |
| MAE (PHQ-8) | ~3.5 | < 3.0 |
| RMSE (PHQ-8) | ~4.8 | < 4.0 |

> *Target: surpass current SOTA F1 of ~0.70 on DAIC-WOZ test set*

### Missing Modality Robustness
- Speech only: maintain ≥ 85% of full-modality F1
- Speech + Text: maintain ≥ 92% of full-modality F1
- Any single modality absent: graceful degradation, no crash

---

## 9. Ablation Study Design

### Research Questions for Ablation

1. **RQ1:** How much does dynamic gating improve over uniform weights?
2. **RQ2:** Does multi-scale temporal fusion outperform single-scale?
3. **RQ3:** What is the contribution of hierarchical vs. flat cross-attention?
4. **RQ4:** How well does the missing modality handler preserve performance?
5. **RQ5:** Do depression-specific behavioral features improve over generic features?
6. **RQ6:** What is the performance-cost tradeoff of each module?

### Metrics to Report
- F1 score (depressed class, macro, weighted)
- Accuracy, Precision, Recall
- AUC-ROC, AUC-PR
- PHQ-8 MAE, RMSE, Pearson r
- Inference time (ms/sample)
- Parameter count (millions)

---

## 10. Publication Plan

### Primary Paper (Journal — Target: Q1)
- **Title:** DG-HMCF: Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion for Automated Depression Detection
- **Target:** IEEE Transactions on Affective Computing / Expert Systems with Applications / Information Fusion
- **Type:** Full research article
- **Sections:** Introduction, Related Work, Methodology, Experiments, Ablation, Discussion, Conclusion
- **Status:** 🔴 Not started

### Secondary Paper (Conference — Target: Top-tier)
- **Title:** Explainable Multimodal Depression Detection with Dynamic Reliability Gating
- **Target:** ACII 2025 / ICASSP 2026 / INTERSPEECH 2026
- **Focus:** Explainability module + clinical validation
- **Status:** 🔴 Not started

### Possible Extension Papers
- Cross-lingual adaptation of DG-HMCF (non-English datasets)
- Real-time streaming depression monitoring
- Privacy-preserving federated DG-HMCF

---

## 11. Target Journals & Conferences

### Journals (Ranked by Priority)

| Journal | Publisher | Impact Factor | Scope | Q-Rank |
|---------|-----------|--------------|-------|--------|
| IEEE Trans. on Affective Computing | IEEE | ~13.9 | Affective AI | Q1 |
| Information Fusion | Elsevier | ~18.6 | Multimodal Fusion | Q1 |
| Expert Systems with Applications | Elsevier | ~8.5 | Applied AI | Q1 |
| Computers in Biology and Medicine | Elsevier | ~7.7 | Medical AI | Q1 |
| IEEE JBHI | IEEE | ~7.7 | Biomedical + Health Informatics | Q1 |
| Neural Networks | Elsevier | ~6.0 | Deep Learning | Q1 |
| Knowledge-Based Systems | Elsevier | ~8.8 | AI/ML | Q1 |

### Conferences (Ranked by Priority)

| Conference | Full Name | Deadline | Scope |
|-----------|-----------|----------|-------|
| ACII | Affective Computing and Intelligent Interaction | ~April annually | Affective computing |
| INTERSPEECH | — | ~March annually | Speech + NLP |
| ICASSP | IEEE International Conference on Acoustics, Speech, Signal Processing | ~Sept annually | Signal processing |
| ACL / EMNLP | — | Varies | NLP |
| AAAI | — | ~August | General AI |
| ICDM | IEEE International Conference on Data Mining | ~June annually | Data mining |

---

## 12. Related Work Log

> Add papers as you read them. Format: **[Tag]** Authors (Year) — *Title* — key takeaway.

### Multimodal Depression Detection

- **[DAIC-WOZ-BASE]** Gratch et al. (2014) — *The Distress Analysis Interview Corpus of Human and Computer Interviews* — introduced DAIC-WOZ dataset
- **[AVEC2019]** Ringeval et al. (2019) — *AVEC 2019 Workshop and Challenge* — benchmark for multimodal affect
- **[MEMOCMT]** *(to be added)* — MemoCMT — cross-modal transformer for emotion, inspiration for Stage 3
- **[MODMA-PAPER]** Cai et al. (2020) — *MODMA dataset* — EEG + speech for depression

### Fusion Methods

- **[ATTNFUSION]** *(add paper)* — attention-based fusion baseline
- **[CROSSATTN]** *(add paper)* — cross-attention fusion
- **[CONCATFUSION]** *(add paper)* — feature concatenation standard

### Missing Modality

- **[MISSINGMOD1]** *(add paper)* — modality-agnostic learning
- **[MISSINGMOD2]** *(add paper)* — prompt-based missing modality

### Speech Features for Depression

- **[WAV2VEC2]** Baevski et al. (2020) — *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
- **[SPEECHDEP]** *(add paper)* — prosodic features and depression

### Language & NLP for Depression

- **[ROBERTA]** Liu et al. (2019) — *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- **[NLPDEP]** *(add paper)* — linguistic markers of depression

### EEG for Depression

- **[EEGDEP1]** *(add paper)* — EEG biomarkers for depression
- **[EEGDEP2]** *(add paper)* — 1D CNN for EEG classification

---

## 13. Experiment Log

> Document every experiment run. Include config changes, results, and observations.

---

### Experiment 001

| Field | Value |
|-------|-------|
| Date | *(date)* |
| Dataset | DAIC-WOZ |
| Modalities | Speech + Text |
| Config | `configs/daic_woz_config.yaml` |
| Notes | Baseline run — RoBERTa + Wav2Vec2 |
| F1 | — |
| AUC | — |
| MAE | — |
| Checkpoint | — |

---

### Experiment 002

| Field | Value |
|-------|-------|
| Date | *(date)* |
| Dataset | DAIC-WOZ |
| Modalities | Speech + Text + Face |
| Config | `configs/daic_woz_config.yaml` (modified) |
| Notes | Added face modality |
| F1 | — |
| AUC | — |
| MAE | — |
| Checkpoint | — |

---

*(copy the block above for each new experiment)*

---

## 14. TODO & Milestones

### Immediate (This Week)
- [ ] Set up Python environment and install dependencies
- [ ] Request DAIC-WOZ dataset access license
- [ ] Download MODMA dataset
- [ ] Run `demo.ipynb` to verify model forward pass
- [ ] Run preprocessing pipeline on sample data

### Short-Term (This Month)
- [ ] Complete data preprocessing for DAIC-WOZ
- [ ] Train Baseline 1 (RoBERTa + Wav2Vec2 + attention)
- [ ] Train Baseline 2 (cross-attention)
- [ ] First DG-HMCF training run on DAIC-WOZ
- [ ] Validate missing modality handler on all 8 combinations

### Medium-Term (3 Months)
- [ ] Full ablation study (all 8 variants)
- [ ] Experiments on MODMA dataset
- [ ] Explainability module evaluation
- [ ] Write methodology section
- [ ] Write experiments section

### Long-Term (6 Months)
- [ ] Complete journal paper draft
- [ ] Submit to target journal
- [ ] PDCH experiments (if dataset available)
- [ ] Conference paper on explainability track

### PhD Milestones
- [ ] Literature review complete
- [ ] Proposal defense / confirmation
- [ ] First journal submission
- [ ] First conference paper
- [ ] Thesis writing begin
- [ ] Thesis submission

---

## 15. Supervisor Notes

> Record meeting notes, feedback, and directions here.

---

### Meeting — *(date)*

**Discussed:**
- *(add notes)*

**Action items:**
- *(add items)*

**Feedback:**
- *(add feedback)*

---

*(copy block above for each meeting)*

---

## 16. PhD Progress Tracker

| Chapter / Component | Status | Notes |
|--------------------|--------|-------|
| Literature Review | 🟡 In Progress | |
| Research Proposal | 🔴 Not Started | |
| Dataset Acquisition | 🟡 In Progress | DAIC-WOZ license pending |
| Data Preprocessing | 🟢 Code Ready | Needs real data |
| Baseline Models | 🟢 Code Ready | Needs training |
| DG-HMCF Implementation | 🟢 Complete | All modules implemented |
| Ablation Study | 🟢 Code Ready | Needs experiments |
| Experiments (DAIC-WOZ) | 🔴 Not Started | |
| Experiments (MODMA) | 🔴 Not Started | |
| Experiments (PDCH) | 🔴 Not Started | |
| Paper Writing | 🔴 Not Started | |
| Journal Submission | 🔴 Not Started | |
| Conference Submission | 🔴 Not Started | |

**Legend:** 🟢 Done · 🟡 In Progress · 🔴 Not Started · ⏸ Blocked

---

## 17. References to Cite

> Maintain a running list of papers confirmed for citation in the paper.

```
[1] Gratch, J. et al. (2014). The Distress Analysis Interview Corpus of Human and Computer Interviews.
[2] Baevski, A. et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS.
[3] Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
[4] Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.
[5] Cai, H. et al. (2020). MODMA dataset: A Multi-modal Open Dataset for Mental-disorder Analysis.
[6] Vaswani, A. et al. (2017). Attention is All You Need. NeurIPS.
[7] Ringeval, F. et al. (2019). AVEC 2019 Workshop and Challenge. ACM MM.
[8] WHO (2023). Depressive disorder (depression) fact sheet.
```

> *(Add more as you read. Maintain in citation manager — Zotero / Mendeley recommended)*

---

## Notes

> General scratch pad — ideas, observations, questions.

```
- Consider adding contrastive learning between modalities as an auxiliary task
- Look into curriculum learning for progressively harder missing-modality scenarios
- Clinical collaboration: validate explainability outputs with psychiatrists
- PHQ-8 vs PHQ-9 — some datasets use PHQ-9, confirm mapping
- Consider SMOTE or class-weighted loss for class imbalance (DAIC-WOZ ~34% depressed)
- Test on non-English speakers to validate cross-lingual claim
```

---

*This file is for personal PhD research tracking. Keep it updated as the project progresses.*
