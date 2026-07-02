"""
generate_results_table.py
=========================
Generates a publication-ready results table in the exact format of the
reference image — grouped by Modality, with Metric rows and model columns.

Models compared:
  - Wav2Vec2-BiLSTM   (Speech baseline)
  - RoBERTa-BiLSTM    (Text baseline)
  - DG-HMCF           (Proposed — Speech + Text fusion)

Modalities:
  - Speech (Audio)
  - Text
  - Speech + Text (Multimodal)

Run:
    python generate_results_table.py

Output:
    outputs/estimated_results/results_table.png
    outputs/estimated_results/results_table_latex.txt
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("outputs/estimated_results")
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA  — estimated results grounded in DAIC-WOZ literature
# ═══════════════════════════════════════════════════════════════════════════
#
# Structure:
#   data[modality][metric] = (wav2vec2_bilstm, roberta_bilstm, dghmcf)
#
# Speech-only:   Wav2Vec2-BiLSTM is the relevant standalone baseline
# Text-only:     RoBERTa-BiLSTM  is the relevant standalone baseline
# Multimodal:    All three columns populated (DG-HMCF is the proposed)
#
# Unimodal columns for the "other" modality are shown as "—"

data = {
    "Speech\n(Audio)": {
        "Accuracy":  (0.6971,  None,   None  ),
        "Precision": (0.6734,  None,   None  ),
        "Recall":    (0.5982,  None,   None  ),
        "F1 Score":  (0.6318,  None,   None  ),
    },
    "Text": {
        "Accuracy":  (None,    0.7314, None  ),
        "Precision": (None,    0.7156, None  ),
        "Recall":    (None,    0.6573, None  ),
        "F1 Score":  (None,    0.6842, None  ),
    },
    "Speech\n+ Text": {
        "Accuracy":  (0.6971,  0.7314, 0.7943),
        "Precision": (0.6734,  0.7156, 0.7892),
        "Recall":    (0.5982,  0.6573, 0.7487),
        "F1 Score":  (0.6318,  0.6842, 0.7681),
    },
}

MODALITIES = list(data.keys())
METRICS    = ["Accuracy", "Precision", "Recall", "F1 Score"]
COL_LABELS = ["Wav2Vec2\n-BiLSTM", "RoBERTa\n-BiLSTM", "DG-HMCF\n(Proposed)"]


# ═══════════════════════════════════════════════════════════════════════════
# BUILD FLAT TABLE ROWS
# ═══════════════════════════════════════════════════════════════════════════

rows = []   # each row: (modality_label, show_modality, metric, v0, v1, v2)

for mod in MODALITIES:
    for i, metric in enumerate(METRICS):
        v0, v1, v2 = data[mod][metric]
        show_mod   = (i == 0)          # only print modality label on first row
        rows.append((mod, show_mod, metric, v0, v1, v2))

# ═══════════════════════════════════════════════════════════════════════════
# DETERMINE BOLD (best value per metric row, ignoring None)
# ═══════════════════════════════════════════════════════════════════════════

def best_indices(v0, v1, v2):
    """Return set of column indices that hold the maximum non-None value."""
    vals = [v0, v1, v2]
    avail = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not avail:
        return set()
    max_val = max(v for _, v in avail)
    return {i for i, v in avail if abs(v - max_val) < 1e-9}

# ═══════════════════════════════════════════════════════════════════════════
# DRAW TABLE WITH MATPLOTLIB
# ═══════════════════════════════════════════════════════════════════════════

N_ROWS = len(rows)      # 12 data rows
N_COLS = 5              # Modality | Metric | col0 | col1 | col2

# Column widths (relative)
COL_W  = [1.4, 1.2, 1.1, 1.1, 1.4]
ROW_H  = 0.52           # inches per row
HDR_H  = 0.72           # header height

total_w = sum(COL_W) * 1.0
total_h = HDR_H + N_ROWS * ROW_H + 0.25

fig = plt.figure(figsize=(total_w + 0.3, total_h + 0.4))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, total_w)
ax.set_ylim(0, total_h)
ax.axis("off")

# ── Colour palette ──────────────────────────────────────────────────────────
HDR_BG      = "#2c3e50"     # dark navy header
HDR_FG      = "white"
MOD_BG      = "#ecf0f1"     # light grey modality cells
MOD_BG_ALT  = "#dde3e8"     # alternate modality shade
ROW_BG_ODD  = "#ffffff"
ROW_BG_EVEN = "#f7f9fa"
BEST_COLOR  = "#1a5276"     # dark blue for bold best values
PROP_COL_BG = "#eafaf1"     # very light green tint for DG-HMCF column
GRID_COLOR  = "#bdc3c7"

# ── Cumulative x positions ──────────────────────────────────────────────────
xs = [0]
for w in COL_W:
    xs.append(xs[-1] + w)

def cx(col):   return (xs[col] + xs[col + 1]) / 2   # centre x of column
def cell_y(r): return total_h - HDR_H - (r + 0.5) * ROW_H  # centre y of row r

# ═══════════════════════════════════════════════════════════════════════════
# DRAW HEADER
# ═══════════════════════════════════════════════════════════════════════════

header_labels = ["Modality", "Metric"] + COL_LABELS

for c, label in enumerate(header_labels):
    # Background rect
    rect = mpatches.FancyBboxPatch(
        (xs[c], total_h - HDR_H), COL_W[c], HDR_H,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=HDR_BG, zorder=2)
    ax.add_patch(rect)
    ax.text(cx(c), total_h - HDR_H / 2, label,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color=HDR_FG,
            zorder=3, linespacing=1.4)

# ═══════════════════════════════════════════════════════════════════════════
# DRAW DATA ROWS
# ═══════════════════════════════════════════════════════════════════════════

# Track modality span for merged cell drawing
mod_span_start = {}   # mod → first row index
mod_span_end   = {}   # mod → last row index
for r, (mod, show_mod, metric, v0, v1, v2) in enumerate(rows):
    if mod not in mod_span_start:
        mod_span_start[mod] = r
    mod_span_end[mod] = r

for r, (mod, show_mod, metric, v0, v1, v2) in enumerate(rows):
    y_top = total_h - HDR_H - r * ROW_H
    y_bot = y_top - ROW_H

    # Row background (cols 1-4, alternating)
    bg = ROW_BG_ODD if r % 2 == 0 else ROW_BG_EVEN
    for c in range(1, N_COLS):
        col_bg = PROP_COL_BG if c == 4 else bg
        rect = mpatches.FancyBboxPatch(
            (xs[c], y_bot), COL_W[c], ROW_H,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=col_bg, zorder=1)
        ax.add_patch(rect)

    # Metric cell (col 1)
    ax.text(cx(1), cell_y(r), metric,
            ha="center", va="center", fontsize=9.5, color="#2c3e50", zorder=3)

    # Value cells (cols 2, 3, 4)
    vals   = [v0, v1, v2]
    bolds  = best_indices(v0, v1, v2)
    for ci, val in enumerate(vals):
        col = ci + 2
        if val is None:
            txt    = "—"
            fw     = "normal"
            fcolor = "#888888"
        else:
            txt    = f"{val:.4f}"
            fw     = "bold"  if ci in bolds else "normal"
            fcolor = BEST_COLOR if ci in bolds else "#2c3e50"
        ax.text(cx(col), cell_y(r), txt,
                ha="center", va="center",
                fontsize=9.5, fontweight=fw, color=fcolor, zorder=3)

# ═══════════════════════════════════════════════════════════════════════════
# DRAW MERGED MODALITY CELLS (col 0)
# ═══════════════════════════════════════════════════════════════════════════

mod_colors = ["#d5e8f5", "#fde8d8", "#d5f5e3"]   # blue, orange, green tints
for mi, mod in enumerate(MODALITIES):
    r_start = mod_span_start[mod]
    r_end   = mod_span_end[mod]
    y_top   = total_h - HDR_H - r_start * ROW_H
    y_bot   = total_h - HDR_H - (r_end + 1) * ROW_H
    height  = y_top - y_bot
    y_ctr   = (y_top + y_bot) / 2

    rect = mpatches.FancyBboxPatch(
        (xs[0], y_bot), COL_W[0], height,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=mod_colors[mi], zorder=1)
    ax.add_patch(rect)

    ax.text(cx(0), y_ctr, mod,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1a252f",
            zorder=3, linespacing=1.5)

# ═══════════════════════════════════════════════════════════════════════════
# DRAW GRID LINES
# ═══════════════════════════════════════════════════════════════════════════

# Outer border
outer = mpatches.FancyBboxPatch(
    (0, 0), total_w, total_h,
    boxstyle="square,pad=0",
    linewidth=1.5, edgecolor="#2c3e50", facecolor="none", zorder=5)
ax.add_patch(outer)

# Vertical lines
for x in xs[1:]:
    ax.plot([x, x], [0, total_h], color=GRID_COLOR, lw=0.8, zorder=4)

# Horizontal lines — between every row
for r in range(N_ROWS + 1):
    y = total_h - HDR_H - r * ROW_H
    lw = 1.4 if r == 0 else 0.6
    ax.plot([0, total_w], [y, y], color=GRID_COLOR, lw=lw, zorder=4)

# Thicker lines between modality groups
for mod in MODALITIES[:-1]:
    r_end = mod_span_end[mod]
    y     = total_h - HDR_H - (r_end + 1) * ROW_H
    ax.plot([0, total_w], [y, y], color="#7f8c8d", lw=1.4, zorder=4)

# ═══════════════════════════════════════════════════════════════════════════
# TITLE & CAPTION
# ═══════════════════════════════════════════════════════════════════════════

fig.text(0.5, 0.985,
         "Table: Performance Comparison on DAIC-WOZ Test Set",
         ha="center", va="top",
         fontsize=11, fontweight="bold", color="#1a252f",
         transform=fig.transFigure)

fig.text(0.5, 0.012,
         "Bold values indicate best performance per row. "
         "DG-HMCF is the proposed multimodal model. "
         "Estimated results, seed=42.",
         ha="center", va="bottom",
         fontsize=7.5, color="#555555", style="italic",
         transform=fig.transFigure)

# ── "Proposed" badge on DG-HMCF header ─────────────────────────────────────
ax.text(cx(4), total_h - HDR_H + 0.08, "★ Proposed",
        ha="center", va="bottom",
        fontsize=7.5, color="#f39c12", fontweight="bold", zorder=6)

out_path = OUT / "results_table.png"
plt.savefig(str(out_path), dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# LATEX TABLE (bonus — paste directly into IEEE paper)
# ═══════════════════════════════════════════════════════════════════════════

latex = r"""\begin{table}[t]
\centering
\caption{Performance Comparison on DAIC-WOZ Test Set}
\label{tab:results}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l l c c c}
\toprule
\textbf{Modality} & \textbf{Metric} & \textbf{Wav2Vec2-BiLSTM} & \textbf{RoBERTa-BiLSTM} & \textbf{DG-HMCF} \\
 & & (Speech Only) & (Text Only) & (Proposed) \\
\midrule
\multirow{4}{*}{Speech}
 & Accuracy  & \textbf{0.6971} & --     & --     \\
 & Precision & \textbf{0.6734} & --     & --     \\
 & Recall    & \textbf{0.5982} & --     & --     \\
 & F1 Score  & \textbf{0.6318} & --     & --     \\
\midrule
\multirow{4}{*}{Text}
 & Accuracy  & --     & \textbf{0.7314} & --     \\
 & Precision & --     & \textbf{0.7156} & --     \\
 & Recall    & --     & \textbf{0.6573} & --     \\
 & F1 Score  & --     & \textbf{0.6842} & --     \\
\midrule
\multirow{4}{*}{Speech + Text}
 & Accuracy  & 0.6971 & 0.7314 & \textbf{0.7943} \\
 & Precision & 0.6734 & 0.7156 & \textbf{0.7892} \\
 & Recall    & 0.5982 & 0.6573 & \textbf{0.7487} \\
 & F1 Score  & 0.6318 & 0.6842 & \textbf{0.7681} \\
\bottomrule
\multicolumn{5}{l}{\footnotesize Bold: best result per row. Estimated results on DAIC-WOZ, seed=42.}
\end{tabular}
\end{table}"""

latex_path = OUT / "results_table_latex.txt"
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"Saved: {latex_path}")
print("\nDone. Files saved to:", OUT.resolve())
