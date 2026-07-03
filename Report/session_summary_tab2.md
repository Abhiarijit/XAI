# TAGFC Session Summary — Tab 2 (June 2026)
# Use this file as context in any new chat tab

---

## Files Created / Modified This Session

| File | What it is |
|---|---|
| `make_flow_diagram.py` | Python script to generate the flow diagram PNG |
| `Report/TAGFC_flow_diagram.png` | Standalone PNG of complete 5-step TAGFC data flow |
| `Report/TAGFC_Presentation.pptx` | 29-slide PowerPoint (flow diagram at slide 11) |

---

## 1. Flow Diagram Image (TAGFC_flow_diagram.png)

**Script:** `make_flow_diagram.py`  
**Output:** `Report/TAGFC_flow_diagram.png` (180 dpi, ~500 KB)  
**Library:** matplotlib, Agg backend, figsize=(20, 15.5), ylim=(0, 15.5)

**Layout:**
- Title bar at top (y=14.75): "TAGFC — Complete 5-Step Data Flow"
- Left column: vertical pipeline → Query X → Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → X*
- Right column: panel "What Each Step Outputs → Where It Goes" — 5 color-coded rows, one per step
- Bottom strip: "Steps 1,2,3 run ONCE · Steps 4⇔5 loop up to 300 iterations"

**Color scheme:**
- Steps 1, 2 → Blue (#005B96), bg (#CCE5FF)
- Step 3 → Green (#007744), bg (#D4EDDA)
- Step 4 → Purple (#5500AA), bg (#F0E0FF)
- Step 5 → Orange (#E86A00), bg (#FFF5E6)
- Title/strips → Navy (#003366)

**All edits made during this session:**
1. Removed orange accent line at top that hid title
2. Increased figure height 14→15.5, moved title box to y=14.75 to separate from right-panel header
3. Raised right-panel header: `RPAN_TOP = step_y[0] + 1.12`
4. Pushed step rows down: `ry = RPAN_TOP - 0.90`, `ROW_H = (RPAN_TOP - RPAN_BOT - 1.0) / 5`
5. Colored output strip font: 10.5 → 12
6. Step equation font: 9.8 → 11.5
7. Step tag ("Step 1") font: 11 → 13
8. Arrow note + destination text font: 10 → 11.5
9. Removed red feedback loop arrow (LOOP ≤300 bracket on left of Steps 4 & 5)

---

## 2. Full TAGFC 5-Step Numerical Walkthrough (T=5, D=5)

### Setup
- T=5 time steps, D=5 sensor features
- K=2 classes: Good / Poor
- L=2 transformer layers
- Hyperparameters: λ₁=0.02, λ₂=0.01, λ₃=0.10, η=0.05, MAX_ITER=300, DELTA_BOUND=2.5, σ=1.5

### Query Matrix X (5×5)
```
        d=1   d=2   d=3   d=4   d=5
  t=1 [ 1.0   0.5   2.0   0.8   1.5 ]
  t=2 [ 2.0   1.0   1.8   0.9   1.2 ]
  t=3 [ 1.5   1.5   2.2   1.1   1.8 ]
  t=4 [ 3.0   2.0   3.5   1.8   2.5 ]
  t=5 [ 2.5   1.8   3.0   1.5   2.2 ]
```
Current: P(Good|X) = 0.78.  Goal: X* with P(Poor|X*) > 0.50

---

### STEP 1 — Attention Rollout

**Formula:** Â^l = 0.5·A^l + 0.5·I,  R = Â²·Â¹,  s_t = normalized column sums of R

**A¹ (Layer 1):**
```
      t1   t2   t3   t4   t5
t1 [0.40 0.20 0.15 0.15 0.10]
t2 [0.20 0.35 0.20 0.15 0.10]
t3 [0.15 0.20 0.35 0.20 0.10]
t4 [0.10 0.15 0.20 0.35 0.20]
t5 [0.10 0.10 0.15 0.20 0.45]
```

**A² (Layer 2):**
```
      t1   t2   t3   t4   t5
t1 [0.35 0.25 0.15 0.15 0.10]
t2 [0.20 0.30 0.25 0.15 0.10]
t3 [0.10 0.20 0.35 0.25 0.10]
t4 [0.10 0.10 0.20 0.40 0.20]
t5 [0.05 0.10 0.15 0.25 0.45]
```

**Rollout R = Â²·Â¹:**
```
     t1     t2     t3     t4     t5
t1 [0.497  0.168  0.125  0.123  0.088]
t2 [0.151  0.469  0.168  0.124  0.088]
t3 [0.104  0.152  0.486  0.168  0.090]
t4 [0.088  0.106  0.154  0.500  0.153]
t5 [0.071  0.089  0.124  0.170  0.546]
```

**s_t (column sums ÷ max=1.085):**
```
s_t = [0.840, 0.907, 0.974, 1.000, 0.890]
```
→ t=4 highest (1.000) — model relies most on t=4 to predict "Good"

**How X is used in Step 1:**
X is NOT directly in the rollout formula. It generates attention matrices via:
```
Q = X · W_Q,   K = X · W_K
A[i,j] = softmax( Q[i]·K[j]ᵀ / √d_k )
```
X at t=4 = [3.0, 2.0, 3.5, 1.8, 2.5] → large Q·Kᵀ dot products → A[4,4] large → s₄=1.000

---

### STEP 2 — Haar DWT

**Why frequency domain (key points):**
- Time domain: changing one value at t=4 → unrealistic spike
- Wavelet domain: changing cA2 shifts both t=3 AND t=4 smoothly → physically realistic
- Energy compaction: most signal energy in cA1, cA2 (slow trend coefficients)
- Joint time-frequency targeting: which time region + which frequency band

**Method:** 1-level Haar DWT, zero-padded to length 6
- cA_k = (x_{2k-1} + x_{2k}) / √2
- cD_k = (x_{2k-1} − x_{2k}) / √2
- cA1,cD1 → time pair (t=1,t=2)
- cA2,cD2 → time pair (t=3,t=4) ← HIGH ATTENTION REGION
- cA3,cD3 → t=5 alone

**C matrix (5 features × 6 coefficients):**
```
        cA1    cA2    cA3    cD1    cD2    cD3
  d=1 [ 2.121  3.182  1.768 -0.707 -1.061  1.768]
  d=2 [ 1.061  2.475  1.273 -0.354 -0.354  1.273]
  d=3 [ 2.687  4.030  2.121  0.141 -0.919  2.121]
  d=4 [ 1.202  2.051  1.061 -0.071 -0.495  1.061]
  d=5 [ 1.909  3.040  1.556  0.212 -0.495  1.556]
```
Initial δ = zeros(5×6),  X̃ = iDWT(C + 0) = X

---

### STEP 3 — Omega Weights (ω)

**Pipeline:**
```
|∂L/∂X| → Gaussian smooth (σ=1.5) → G_smooth → DWT → G_wav
s_t → project to wavelet positions → s_wav
M = s_wav × G_wav
ω = 1 / normalize(M)
```

**s_wav (s_t projected to 6 wavelet positions):**
```
s_wav = [0.874, 0.987, 0.890, 0.874, 0.987, 0.890]
         cA1    cA2    cA3    cD1    cD2    cD3
  where: avg(s1,s2)=0.874, avg(s3,s4)=0.987, s5=0.890
```

**ω matrix (5×6) — perturbation cost per coefficient:**
```
        cA1    cA2    cA3    cD1     cD2    cD3
  d=1 [ 1.75   1.00   2.39  16.30   6.99   2.39 ]
  d=2 [ 2.75   1.27   2.97  12.60   6.99   2.97 ]
  d=3 [ 2.11   1.12   2.49  16.30   6.99   2.49 ]
  d=4 [ 2.63   1.44   3.29  32.60   8.15   3.29 ]
  d=5 [ 2.87   1.65   3.65  32.60   9.52   3.65 ]
```
→ ω SMALLEST at (d=1, cA2) = 1.00 → cheapest to perturb
→ Detail coefficients (cD) ω = 6–32 → effectively blocked by soft-thresh

**Full causal chain: why t=4 changes most in X*:**
```
X[t=4] has high values
  ↓ Step 1
s₄ = 1.000 (highest saliency)
  ↓ Step 3 (s_t SKIPS Step 2, goes directly here)
s_wav[cA2] = 0.987 → M[cA2] large → ω[cA2] SMALL (=1.00)
  ↓ Step 4
L_sparse penalty small for cA2 → optimizer pushes δ[cA2] freely
  ↓ Step 5
threshold = 0.001 × ω[cA2] = 0.001 (tiny) → δ[cA2] survives, grows each iter
  ↓ iDWT
cA2 large → both t=3 and t=4 change most in X*
```

---

### STEP 4 — 4-Term Objective

**Formula:**
```
L = L_flip  +  λ₁·L_sparse(ω)  +  λ₂·L_cross  +  λ₃·L_manifold(C)
```
- L_flip = max(0, 0.5 − P(Poor|X̃))  → force flip to target class
- L_sparse = Σ ω·|δ|                  → sparse perturbation, weighted by importance
- L_cross                              → preserve cross-sensor relationships
- L_manifold                           → keep X* on real data manifold

At iteration 0: L_flip = max(0, 0.5 − 0.22) = 0.28

**Output:** gradient ∇L w.r.t. δ

**Where gradient values come from:**
- From PyTorch `loss.backward()` — automatic differentiation, NOT manually set
- Chain: δ → C+δ → iDWT → X̃ → Transformer → P(class) → L → .backward() → δ.grad
- ∂X̃/∂δ comes from iDWT (linear op), so gradient flows back as DWT(∂L/∂X̃)
- Values in toy examples were illustrative/assumed to show the pattern

**Assumed ∇L for illustration (5×6):**
```
        cA1    cA2    cA3    cD1    cD2    cD3
  d=1 [-0.12  -0.18  -0.08  -0.02  -0.04  -0.05]
  d=2 [-0.08  -0.12  -0.06  -0.01  -0.03  -0.04]
  d=3 [-0.10  -0.15  -0.07  -0.01  -0.03  -0.04]
  d=4 [-0.07  -0.10  -0.05  -0.01  -0.02  -0.03]
  d=5 [-0.09  -0.13  -0.06  -0.01  -0.03  -0.04]
```
Negative = increasing δ reduces L = moves prediction toward "Poor"

---

### STEP 5 — Proximal GD + Soft-Thresholding

**Three sub-operations per iteration:**
```
1. Gradient step:     δ ← δ − η·∇L          (η = 0.05)
2. Soft-threshold:    δ[d,l] ← sign(δ[d,l])·max(|δ[d,l]| − η·λ₁·ω[d,l],  0)
3. Bound clip:        δ[d,l] ← clip(δ[d,l], −2.5, +2.5)
```

threshold[d,l] = 0.05 × 0.02 × ω[d,l] = **0.001 × ω[d,l]**

**Iteration 1, d=1 (full detail):**
```
coeff   δ_raw    threshold    δ_after_thresh
cA1    0.0060    0.00175      0.00425
cA2    0.0090    0.00100      0.00800   ← survives (low ω → low threshold)
cA3    0.0040    0.00239      0.00161
cD1    0.0010    0.01630      0         ← ZEROED (ω=16.3 → high threshold)
cD2    0.0020    0.00699      0         ← ZEROED (ω=6.99)
cD3    0.0025    0.00239      0.00011
```

**Converged δ (after ~300 iterations):**
```
        cA1   cA2   cA3   cD1  cD2  cD3
  d=1 [ 0.52  0.85  0.31   0    0    0 ]
  d=2 [ 0.28  0.61  0.08   0    0    0 ]
  d=3 [ 0.44  0.72  0.21   0    0    0 ]
  d=4 [ 0.18  0.48  0.00   0    0    0 ]
  d=5 [ 0.31  0.59  0.00   0    0    0 ]
```
All detail coefficients = 0 throughout — zeroed immediately by soft-thresh

**Final X* = iDWT(C + δ):**
```
        d=1   d=2   d=3   d=4   d=5
  t=1 [ 1.37  0.70  2.26  0.95  1.71 ]
  t=2 [ 2.74  1.30  2.34  1.08  1.49 ]
  t=3 [ 2.10  1.71  2.81  1.46  2.25 ]
  t=4 [ 4.01  2.58  4.49  2.46  3.22 ]   ← LARGEST change (t=4 highest attention)
  t=5 [ 2.72  1.86  3.21  1.50  2.20 ]
```

---

## 3. δ (Delta) — What the Notation Means

δ = the **perturbation matrix** that TAGFC learns during Steps 4 & 5.

```
X*  =  iDWT( C  +  δ )
              ↑       ↑
         original   what
         wavelet    TAGFC
         coeffs     learns
```

- Shape: same as C → (D × L) = (5 × 6) in the example
- Starts at zeros, updated each iteration
- Non-zero entries = coefficients actually changed
- Zero entries = untouched (zeroed by soft-thresholding with ω)

**Three constraints enforced on δ:**

| Constraint | Enforced by |
|---|---|
| Small — don't change X too much | L_sparse in Step 4 |
| Sparse — most entries = 0 | Soft-thresholding in Step 5 with ω |
| Bounded — no extreme values | clip(δ, −2.5, +2.5) in Step 5 |

**One-line for professor:**
> "δ is the minimal sparse perturbation in wavelet space that, when inverted back to time domain, flips the model's prediction."

---

## 4. Why Frequency Domain — Key Points

1. **Realistic changes:** cA2 perturbation shifts t=3 and t=4 together smoothly, not a single spike
2. **Sparsity = interpretability:** one coefficient change = one structured pattern change
3. **Joint targeting:** "slow trend in t=3,t=4 window of PM2.5" (not just "t=4 changed")
4. **ω penalty meaningful:** high-freq detail gets high ω → model naturally ignores noise
5. **Analogy:** adjusting bass level in a music section (smooth) vs deleting one note (glitch)

---

## 5. The 8 Evaluation Metrics

### Quick Reference Table

| Metric | What it measures | Better value |
|---|---|---|
| **RCF** (Rate of Counterfactual Flip) | % of samples that successfully flipped | **Higher → 1.0** |
| **Coherence** | Cross-sensor physical consistency preserved | **Higher → 1.0** |
| **Sparsity** | Fraction of wavelet coefficients unchanged | **Higher → 1.0** |
| **Compactness** | Fraction of time steps unchanged | **Higher → 1.0** |
| **L1 Distance** | Average absolute change X→X* | **Lower → 0** |
| **L2 Distance** | Euclidean change, penalises spikes | **Lower → 0** |
| **DTW Distance** | Temporal shape preservation | **Lower → 0** |
| **Plausibility** | Distance of X* to real training samples | **Lower → 0** |

---

### RCF — Rate of Counterfactual Flip

```
RCF = (samples where f(X*) = target class) / total test samples
```

- RCF = 0.95 → 95 of 100 samples flipped ✓ (excellent)
- RCF = 0.60 → only 60 of 100 flipped ✗ (poor)
- **HIGHER IS BETTER** — always, no exception
- TAGFC wins because: L_flip = max(0, 0.5−P(target|X̃)) directly pushes P above 0.5
- Low RCF causes: optimiser not converged in 300 iters, DELTA_BOUND too tight, sample near decision boundary

---

### Coherence

Measures whether inter-feature correlations are preserved in X* vs X.

```
Corr_X  = D×D correlation matrix of X
Corr_X* = D×D correlation matrix of X*
Coherence = 1 − ||Corr_X − Corr_X*||_F / ||Corr_X||_F
```

- Bad: PM2.5 doubles at t=4 but NO2 drops → physically impossible → low coherence
- Good: all features shift in same time window proportionally → relationships preserved
- **HIGHER IS BETTER**
- TAGFC wins because: DWT shifts all features in same wavelet coefficient (cA2) → cross-sensor structure preserved
- L_cross term in Step 4 also directly optimises for this

---

### Other 6 Metrics (brief)

**Validity / Sparsity:**
```
Sparsity = zero entries in δ / total entries in δ
```
e.g., 20 zeros out of 30 entries → Sparsity = 0.667. Higher is better.

**Compactness:**
```
Compactness = unchanged time steps / T
```
Sparsity counts wavelet coeffs; Compactness counts time steps. Both should be high.

**L1 Distance:**
```
(1/T·D) · Σ Σ |X*[t,d] − X[t,d]|
```
Lower = smaller, more minimal change.

**L2 Distance:** Like L1 but squares differences — penalises large spikes more.

**DTW Distance:** Shape-level similarity between X and X*. TAGFC preserves shape because it only shifts coefficient magnitudes. Lower = better.

**Plausibility:**
```
min distance from X* to any training sample of the target class
```
Lower = X* looks like a real data point. TAGFC's L_manifold term optimises this directly.

---

## 6. Previous Session Context (from Tab 1)

- 29-slide PPTX: `Report/TAGFC_Presentation.pptx`
- Slide 11 = TAGFC flow diagram slide (inserted via `insert_flow_slide.py`)
- Original make_pptx.py generates 28 slides, insert_flow_slide.py adds slide 11 → 29 total
- Datasets: AQI India (T=14, D=12, K=3, N=60), NATOPS (T=51, D=24, K=6, N=30)
- TAGFC wins 7/8 metrics on AQI India
- Thesis trimmed from 93 → ~55 pages (see session_report_trimming_june2026.md in memory/)
- Earlier toy example uses T=4, D=2 (PM2.5, NO2): s_t=[0.21,0.24,0.26,0.29], ω=[1.00,2.04,6.25,2.44]
- c2 coefficient (index 2) zeroed in T=4 example because ω=6.25 → threshold > |δ|

---

## How to Use This File in a New Chat

Paste this at the start of your new chat:

> "Please read the file at:
> `c:\Users\abhia\Desktop\counterfactual_basis_kernel-main\Report\session_summary_tab2.md`
> as full context for our conversation about the TAGFC thesis project."
