"""
Creates TAGFC_flow_diagram.png — complete 5-step data flow diagram.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(20, 15.5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 15.5)
ax.axis("off")
fig.patch.set_facecolor("#F0F4F8")

# ── colour palette ────────────────────────────────────────────────────────────
C_NAVY    = "#003366"
C_BLUE    = "#005B96"
C_ORANGE  = "#E86A00"
C_GREEN   = "#007744"
C_PURPLE  = "#5500AA"
C_LBLUE   = "#CCE5FF"
C_LGREEN  = "#D4EDDA"
C_LPURPLE = "#F0E0FF"
C_LCREAM  = "#FFF5E6"
C_WHITE   = "#FFFFFF"
C_LGREY   = "#F0F4F8"
C_DGREY   = "#333333"
C_RED     = "#CC0000"

def rounded_box(ax, x, y, w, h, facecolor, edgecolor, lw=2, radius=0.3):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=3)
    ax.add_patch(box)

def header_box(ax, x, y, w, h, color):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0,rounding_size=0.2",
                         facecolor=color, edgecolor="none",
                         linewidth=0, zorder=4)
    ax.add_patch(box)

def arrow_down(ax, x, y_start, y_end, color=C_ORANGE, lw=2.5):
    ax.annotate("", xy=(x, y_end+0.05), xytext=(x, y_start),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=lw,
                                mutation_scale=18))

def arrow_right(ax, x_start, x_end, y, color=C_ORANGE, lw=2):
    ax.annotate("", xy=(x_end-0.05, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=lw,
                                mutation_scale=15))

def label(ax, text, x, y, size=10, color=C_DGREY, ha="center", va="center",
          bold=False, style="normal", zorder=5):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=size, color=color,
            ha=ha, va=va, fontweight=weight,
            fontstyle=style, zorder=zorder,
            fontfamily="DejaVu Sans")

# ════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — vertical step pipeline  (x: 0.3 – 6.5)
# ════════════════════════════════════════════════════════════════════════════

STEP_X   = 0.35
STEP_W   = 6.0
STEP_H   = 1.75
GAP      = 0.55
HDR_H    = 0.42

# Y positions for each step (top of box)
step_y = [13.0 - i*(STEP_H + GAP) for i in range(5)]

step_meta = [
    ("Step 1",  "Attention Rollout",
     C_BLUE,    C_LBLUE,
     [" Âᴹ = 0.5·Aᴹ + 0.5·I",
      " Rollout = Â¹ ⊗ Â² ⊗ ... ⊗ Âᴸ",
      " sₜ = column sums  ∈ [0,1]ᵀ"]),

    ("Step 2",  "Haar DWT",
     C_BLUE,    C_LBLUE,
     [" Cₕ = DWT(X[:, d])  ∈  ℝᴸ",
      " Full matrix: C  ∈  ℝᴰˣᴸ",
      " Counterfactual: X̃ = iDWT(C + δ)"]),

    ("Step 3",  "Omega Weights (ω)",
     C_GREEN,   C_LGREEN,
     [" G = Gaussian_smooth(|∂L/∂X|,  σ=1.5)",
      " Gᵂᵃᵛ = DWT(|G|)  ∈  ℝᴰˣᴸ",
      " ω = 1 / normalize(sₜ × Gᵂᵃᵛ)"]),

    ("Step 4",  "4-Term Objective",
     C_PURPLE,  C_LPURPLE,
     [" L = L₊ₗᵢₚ + λ₁·Lₛₚᵃʳₛᵉ(ω) + λ₂·Lᴄʳᵒₛₛ + λ₃·Lₘᵃⁿᴵᴸᵒᴸᵈ(C)",
      " Output: gradient  ∇L  (direction to push δ)"]),

    ("Step 5",  "Proximal GD + Soft-Thresholding",
     C_ORANGE,  C_LCREAM,
     [" δ ← δ − η·∇L            [gradient step]",
      " δᴄᵈ ← sign(δᴄᵈ)·max(|δᴄᵈ|−η·λ₁·ωᴄᵈ, 0)  [soft-thresh]",
      " δᴄᵈ ← clip(δᴄᵈ, −2.5, +2.5)          [bound clip]"]),
]

for i, (step_tag, step_name, hdr_col, bg_col, eqs) in enumerate(step_meta):
    y = step_y[i]
    # background
    rounded_box(ax, STEP_X, y - STEP_H, STEP_W, STEP_H, bg_col, hdr_col, lw=2)
    # header
    header_box(ax, STEP_X, y - HDR_H, STEP_W, HDR_H, hdr_col)
    label(ax, step_tag,   STEP_X + 0.55, y - HDR_H/2, size=13, color=C_WHITE, bold=True, ha="left")
    label(ax, step_name,  STEP_X + 1.55, y - HDR_H/2, size=12.5, color=C_WHITE, bold=True, ha="left")
    # equations
    for j, eq in enumerate(eqs):
        label(ax, eq, STEP_X + 0.15, y - HDR_H - 0.3 - j*0.4,
              size=11.5, color=C_DGREY, ha="left", va="center",
              style="normal")

# down arrows between steps
for i in range(4):
    y_start = step_y[i] - STEP_H
    y_end   = step_y[i+1]
    cx = STEP_X + STEP_W/2
    arrow_down(ax, cx, y_start, y_end + 0.05, lw=3)

# ── Query X box (top) ─────────────────────────────────────────────────────────
QY = step_y[0] + 0.35
rounded_box(ax, STEP_X+0.5, QY, STEP_W-1.0, 0.55, C_NAVY, C_NAVY, lw=0)
label(ax, "Query X   (T × D  multivariate time series)",
      STEP_X + STEP_W/2, QY + 0.27,
      size=12.5, color=C_WHITE, bold=True, ha="center")
arrow_down(ax, STEP_X + STEP_W/2, QY, step_y[0] + 0.02, lw=3)

# ── Counterfactual X* box (bottom) ────────────────────────────────────────────
last_y = step_y[4] - STEP_H - 0.55
rounded_box(ax, STEP_X+0.5, last_y, STEP_W-1.0, 0.55, C_GREEN, C_GREEN, lw=0)
label(ax, "Counterfactual X*  =  iDWT(C + δ)",
      STEP_X + STEP_W/2, last_y + 0.27,
      size=12.5, color=C_WHITE, bold=True, ha="center")
arrow_down(ax, STEP_X + STEP_W/2,
           step_y[4] - STEP_H, last_y + 0.55, lw=3)

# ════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — connection annotations  (x: 7.0 – 19.7)
# ════════════════════════════════════════════════════════════════════════════

RX   = 7.0
RW   = 12.6
RPAN_TOP = step_y[0] + 1.12
RPAN_BOT = last_y - 0.15

# panel background
rounded_box(ax, RX-0.15, RPAN_BOT, RW, RPAN_TOP - RPAN_BOT,
            C_WHITE, C_NAVY, lw=2, radius=0.4)

# panel header
rounded_box(ax, RX-0.15, RPAN_TOP-0.52, RW, 0.52, C_NAVY, C_NAVY, lw=0, radius=0.4)
label(ax, "What Each Step Outputs  →  Where It Goes",
      RX + RW/2 - 0.15, RPAN_TOP - 0.26,
      size=14, color=C_WHITE, bold=True, ha="center")

# ── Row definitions ───────────────────────────────────────────────────────────
rows = [
    # (step_y_center, hdr_color, output_text, arrow_note, to_text)
    (step_y[0] - STEP_H/2,
     C_BLUE,
     "Output:  sₜ = [0.21, 0.24, 0.26, 0.29]    (temporal saliency)",
     "SKIPS Step 2   →   used directly in Step 3",
     "Step 3:   M = sₜ × Gᵂᵃᵛ      (joint time-frequency importance)"),

    (step_y[1] - STEP_H/2,
     C_BLUE,
     "Output:  C ∈ ℝᴰˣᴸ    (wavelet coefficient matrix)",
     "Used in Steps 4 AND 5",
     "Step 4:  X̃ = iDWT(C+δ)  for loss\n"
     "Step 5:  X* = iDWT(C + δ_final)"),

    (step_y[2] - STEP_H/2,
     C_GREEN,
     "Output:  ω = [1.00, 2.04, 6.25, 2.44]    (perturbation cost per coefficient)",
     "Used in Steps 4 AND 5",
     "Step 4:  L_sparse = Σ ω·|δ|   (weighted sparsity loss)\n"
     "Step 5:  threshold = η·λ₁·ω   (high ω → zeroed out)"),

    (step_y[3] - STEP_H/2,
     C_PURPLE,
     "Output:  ∇L    (gradient of total loss w.r.t. δ)",
     "Used in Step 5 only",
     "Step 5:  δ ← δ − η·∇L     (gradient step direction)"),

    (step_y[4] - STEP_H/2,
     C_ORANGE,
     "Output:  δ    (optimised sparse wavelet perturbation)",
     "Feeds BACK to Step 4   (loop repeats up to 300 iterations)",
     "Step 4 recomputes loss with new X̃ = iDWT(C+δ)\n"
     "EXIT when P(target | X̃) > 0.5   →   return X*"),
]

ROW_H = (RPAN_TOP - RPAN_BOT - 1.0) / 5

for i, (yc, hcol, out_txt, arrow_txt, to_txt) in enumerate(rows):
    ry = RPAN_TOP - 0.90 - i * ROW_H

    # coloured header strip
    rounded_box(ax, RX, ry - 0.34, RW - 0.2, 0.34, hcol, "none", lw=0, radius=0.15)
    label(ax, out_txt, RX + 0.15, ry - 0.17,
          size=12, color=C_WHITE, bold=True, ha="left")

    # arrow note
    label(ax, "↓  " + arrow_txt, RX + 0.25, ry - 0.60,
          size=11.5, color=C_ORANGE, bold=True, ha="left")

    # "to" text (may be 2 lines)
    lines = to_txt.split("\n")
    for k, ln in enumerate(lines):
        label(ax, ln, RX + 0.35, ry - 0.95 - k*0.32,
              size=11.5, color=C_DGREY, ha="left")

    # horizontal separator
    if i < 4:
        sep_y = ry - ROW_H + 0.08
        ax.plot([RX, RX + RW - 0.2], [sep_y, sep_y],
                color="#CCCCCC", lw=1.0, zorder=5)

    # connector dot on left edge → links to step box
    ax.plot(RX - 0.12, yc, "o", markersize=7,
            color=hcol, zorder=6)
    ax.plot([STEP_X + STEP_W, RX - 0.12], [yc, yc],
            color=hcol, lw=1.5, linestyle="--", zorder=5, alpha=0.6)


# ════════════════════════════════════════════════════════════════════════════
# ONE-TIME vs LOOP label strip at bottom
# ════════════════════════════════════════════════════════════════════════════
strip_y = last_y - 0.62
rounded_box(ax, 0.2, strip_y, 19.6, 0.48, C_NAVY, C_NAVY, lw=0, radius=0.2)
label(ax,
      "Steps 1, 2, 3  run ONCE  (before optimisation)     ···     "
      "Steps 4 ⇔ 5  loop until P(target | X̃) > 0.5  or  max 300 iterations",
      10.0, strip_y + 0.24,
      size=12, color=C_WHITE, bold=True, ha="center")

# ── Title ─────────────────────────────────────────────────────────────────────
rounded_box(ax, 0.2, 14.75, 19.6, 0.52, C_NAVY, C_NAVY, lw=0, radius=0.2)
label(ax, "TAGFC — Complete 5-Step Data Flow  (How Each Step’s Output Feeds the Next)",
      10.0, 15.01, size=15, color=C_WHITE, bold=True, ha="center")


# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)
out = r"c:\Users\abhia\Desktop\counterfactual_basis_kernel-main\Report\TAGFC_flow_diagram.png"
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved:", out)
