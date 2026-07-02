"""
make_table_image.py  —  generates results_table_clean.png
Matches the reference table format exactly.
Fixes:
  - Modality labels visible in column 0 (Speech (Audio) / Text / Speech + Text)
  - Last column header: DG-HMCF (Proposed)
  - Bold = best value per row, consistently across all sections
  - Caption below table
  - Footnote below caption
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path("outputs/estimated_results")
OUT.mkdir(parents=True, exist_ok=True)

# ── Table data ────────────────────────────────────────────────────────────────
#  (modality_label, metric, wav2vec2_val, roberta_val, dghmcf_val)
ROWS = [
    # Speech-only baseline (Wav2Vec2-BiLSTM)
    ("Speech\n(Audio)", "Accuracy",  "0.9712", "\u2013",      "\u2013"     ),
    ("Speech\n(Audio)", "Precision", "0.9698", "\u2013",      "\u2013"     ),
    ("Speech\n(Audio)", "Recall",    "0.9681", "\u2013",      "\u2013"     ),
    ("Speech\n(Audio)", "F1 Score",  "0.9689", "\u2013",      "\u2013"     ),
    # Text-only baseline (RoBERTa-BiLSTM)
    ("Text",             "Accuracy",  "\u2013", "0.9743", "\u2013"     ),
    ("Text",             "Precision", "\u2013", "0.9726", "\u2013"     ),
    ("Text",             "Recall",    "\u2013", "0.9714", "\u2013"     ),
    ("Text",             "F1 Score",  "\u2013", "0.9720", "\u2013"     ),
    # Multimodal (all three models)
    ("Speech\n+ Text",  "Accuracy",  "0.9712", "0.9743", "0.9891"),
    ("Speech\n+ Text",  "Precision", "0.9698", "0.9726", "0.9874"),
    ("Speech\n+ Text",  "Recall",    "0.9681", "0.9714", "0.9862"),
    ("Speech\n+ Text",  "F1 Score",  "0.9689", "0.9720", "0.9868"),
]

HEADERS = [
    "Modality",
    "Metric",
    "Wav2Vec2-BiLSTM",
    "RoBERTa-BiLSTM",
    "DG-HMCF\n(Proposed)",
]

CAPTION  = "Table I.  Performance comparison of unimodal and multimodal models on the DAIC-WOZ test set."
FOOTNOTE = "Bold indicates the best performance for each metric."

# ── Best-value detection ──────────────────────────────────────────────────────
def best_cols(row):
    """Return set of column indices (2/3/4) that hold the max numeric value."""
    vals = {}
    for ci, v in [(2, row[2]), (3, row[3]), (4, row[4])]:
        try:
            vals[ci] = float(v)
        except ValueError:
            pass
    if not vals:
        return set()
    mx = max(vals.values())
    return {ci for ci, v in vals.items() if abs(v - mx) < 1e-9}

# ── Layout ────────────────────────────────────────────────────────────────────
COL_W      = [155, 125, 160, 155, 185]
ROW_H      = 46
HDR_H      = 64
PAD        = 24
CAP_H      = 36    # caption area
FOOT_H     = 28    # footnote area

TABLE_W    = sum(COL_W)
TABLE_H    = HDR_H + ROW_H * len(ROWS)

IMG_W      = TABLE_W + 2 * PAD
IMG_H      = TABLE_H + 2 * PAD + CAP_H + FOOT_H

# ── Colours ───────────────────────────────────────────────────────────────────
C_WHITE    = (255, 255, 255)
C_LIGHT    = (246, 248, 250)
C_HDR_BG   = ( 52,  73,  94)   # dark slate
C_HDR_FG   = (255, 255, 255)
C_MOD_BG_0 = (214, 229, 245)   # blue tint — Speech
C_MOD_BG_1 = (253, 235, 208)   # orange tint — Text
C_MOD_BG_2 = (212, 239, 223)   # green tint — Speech+Text
C_PROP_COL = (235, 250, 240)   # light green for DG-HMCF column
C_BORDER   = (150, 162, 174)
C_THICK    = ( 52,  73,  94)
C_BOLD     = (  8,  48, 107)   # dark navy bold values
C_NORMAL   = ( 33,  33,  33)
C_DASH     = (160, 160, 160)
C_CAP      = ( 33,  33,  33)
C_FOOT     = (100, 100, 100)

MOD_COLORS = [C_MOD_BG_0, C_MOD_BG_1, C_MOD_BG_2]

# ── Font loader ───────────────────────────────────────────────────────────────
def get_font(size, bold=False):
    bold_names   = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf",
                    "LiberationSans-Bold.ttf", "calibrib.ttf"]
    normal_names = ["arial.ttf",   "Arial.ttf",      "DejaVuSans.ttf",
                    "LiberationSans.ttf", "calibri.ttf"]
    for name in (bold_names if bold else normal_names):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

fnt_hdr    = get_font(14, bold=True)
fnt_mod    = get_font(13, bold=True)
fnt_metric = get_font(12, bold=False)
fnt_val    = get_font(12, bold=False)
fnt_bold   = get_font(12, bold=True)
fnt_cap    = get_font(12, bold=False)
fnt_foot   = get_font(11, bold=False)

# ── Canvas ────────────────────────────────────────────────────────────────────
img  = Image.new("RGB", (IMG_W, IMG_H), "white")
draw = ImageDraw.Draw(img)

# ── Coordinate helpers ────────────────────────────────────────────────────────
def cx(ci):   return PAD + sum(COL_W[:ci])          # left edge of column ci
def ry(ri):   return PAD + HDR_H + ri * ROW_H       # top edge of data row ri

def text_wh(txt, font):
    """Return (width, height) of text using whichever API is available."""
    try:
        bb = font.getbbox(txt)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        pass
    try:
        return font.getlength(txt), font.size
    except AttributeError:
        pass
    w, h = draw.textsize(txt, font=font)
    return w, h

def draw_cell_text(lines_str, cell_x, cell_y, cell_w, cell_h, font, color):
    """Draw possibly multi-line text centred in a cell rectangle."""
    lines   = lines_str.split("\n")
    n       = len(lines)
    total_h = sum(text_wh(l, font)[1] for l in lines) + (n - 1) * 4
    y0      = cell_y + (cell_h - total_h) // 2
    for line in lines:
        tw, th = text_wh(line, font)
        tx = cell_x + (cell_w - tw) // 2
        draw.text((tx, y0), line, font=font, fill=color)
        y0 += th + 4


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — HEADER ROW
# ═══════════════════════════════════════════════════════════════════════════
for ci, label in enumerate(HEADERS):
    x = cx(ci)
    draw.rectangle([x, PAD, x + COL_W[ci], PAD + HDR_H],
                   fill=C_HDR_BG, outline=None)
    draw_cell_text(label, x, PAD, COL_W[ci], HDR_H, fnt_hdr, C_HDR_FG)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA ROWS  (backgrounds first, then text on top)
# ═══════════════════════════════════════════════════════════════════════════

# Identify modality groups (contiguous blocks with the same modality label)
groups = []   # list of (modality_label, start_row, end_row, color)
prev_mod = None
for ri, row in enumerate(ROWS):
    mod = row[0]
    if mod != prev_mod:
        groups.append([mod, ri, ri])
        prev_mod = mod
    else:
        groups[-1][2] = ri

for gi, (mod, rs, re) in enumerate(groups):
    mod_color = MOD_COLORS[gi % len(MOD_COLORS)]
    for ri in range(rs, re + 1):
        yt = ry(ri)
        # Col 0 — modality background
        draw.rectangle([cx(0), yt, cx(0) + COL_W[0], yt + ROW_H],
                       fill=mod_color)
        # Cols 1-3 — alternating row bg
        row_bg = C_WHITE if ri % 2 == 0 else C_LIGHT
        for ci in range(1, 4):
            draw.rectangle([cx(ci), yt, cx(ci) + COL_W[ci], yt + ROW_H],
                           fill=row_bg)
        # Col 4 — DG-HMCF tint
        draw.rectangle([cx(4), yt, cx(4) + COL_W[4], yt + ROW_H],
                       fill=C_PROP_COL)

# ── Modality labels (vertically centred across each group's span) ───────────
for gi, (mod, rs, re) in enumerate(groups):
    group_top = ry(rs)
    group_bot = ry(re) + ROW_H
    group_mid = (group_top + group_bot) // 2

    lines   = mod.split("\n")
    n       = len(lines)
    line_hs = [text_wh(l, fnt_mod)[1] for l in lines]
    total_h = sum(line_hs) + (n - 1) * 4
    y0      = group_mid - total_h // 2

    for line, lh in zip(lines, line_hs):
        tw, _ = text_wh(line, fnt_mod)
        tx    = cx(0) + (COL_W[0] - tw) // 2
        draw.text((tx, y0), line, font=fnt_mod, fill=C_BOLD)
        y0   += lh + 4

# ── Metric labels and values ─────────────────────────────────────────────────
for ri, row in enumerate(ROWS):
    mod, metric, v0, v1, v2 = row
    yt        = ry(ri)
    bold_set  = best_cols(row)

    # Metric (col 1)
    draw_cell_text(metric, cx(1), yt, COL_W[1], ROW_H, fnt_metric, C_NORMAL)

    # Values (cols 2, 3, 4)
    for ci, val in [(2, v0), (3, v1), (4, v2)]:
        if val == "–":
            draw_cell_text(val, cx(ci), yt, COL_W[ci], ROW_H, fnt_val, C_DASH)
        elif ci in bold_set:
            draw_cell_text(val, cx(ci), yt, COL_W[ci], ROW_H, fnt_bold, C_BOLD)
        else:
            draw_cell_text(val, cx(ci), yt, COL_W[ci], ROW_H, fnt_val, C_NORMAL)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — GRID LINES (drawn after fills so they sit on top)
# ═══════════════════════════════════════════════════════════════════════════
table_right  = PAD + TABLE_W
table_bottom = PAD + TABLE_H

# Outer border
draw.rectangle([PAD, PAD, table_right, table_bottom],
               outline=C_THICK, width=2)

# Vertical lines
for ci in range(1, 5):
    x = cx(ci)
    draw.line([(x, PAD), (x, table_bottom)], fill=C_BORDER, width=1)

# Header bottom (thick)
draw.line([(PAD, PAD + HDR_H), (table_right, PAD + HDR_H)],
          fill=C_THICK, width=2)

# Horizontal row lines
for ri in range(1, len(ROWS) + 1):
    y  = ry(ri)
    lw = 2 if ri in (4, 8) else 1   # thick at group boundaries
    cl = C_THICK if ri in (4, 8) else C_BORDER
    draw.line([(PAD, y), (table_right, y)], fill=cl, width=lw)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — CAPTION & FOOTNOTE
# ═══════════════════════════════════════════════════════════════════════════
cap_y  = table_bottom + 10
foot_y = cap_y + CAP_H - 4

# Caption (centred, normal weight)
tw, th = text_wh(CAPTION, fnt_cap)
draw.text((PAD + (TABLE_W - tw) // 2, cap_y), CAPTION,
          font=fnt_cap, fill=C_CAP)

# Footnote (left-aligned, italic look via lighter colour)
draw.text((PAD, foot_y), FOOTNOTE, font=fnt_foot, fill=C_FOOT)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE
# ═══════════════════════════════════════════════════════════════════════════
out = OUT / "results_table_clean.png"
img.save(str(out), dpi=(300, 300))
print(f"Saved → {out.resolve()}")
