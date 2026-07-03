/**
 * make_natops_summary.js
 * Generates TAGFC_NATOPS_Summary.docx — complete context for a new chat session.
 * Run: node make_natops_summary.js
 */
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageBreak,
} = require('docx');
const fs   = require('fs');
const path = require('path');

// ── Palette ──────────────────────────────────────────────────────────────────
const BLUE   = '1F3864'; const LBLUE  = '2E75B6'; const LLBLUE = 'D6E4F0';
const LGREEN = 'E1F5EE'; const LAMBER = 'FFF3CD'; const LRED   = 'FFE0E0';
const GREY   = 'F5F5F5'; const DGREY  = '444444'; const BLACK  = '000000';

// ── Border / cell helpers ────────────────────────────────────────────────────
const bd  = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
const bds = { top: bd, bottom: bd, left: bd, right: bd };
const mg  = (t=80,b=80,l=120,r=120) => ({ top:t, bottom:b, left:l, right:r });

// ── Typography helpers ────────────────────────────────────────────────────────
const h1 = t => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 120 },
  children: [new TextRun({ text: t, bold: true, size: 30, color: BLUE, font: 'Arial' })]
});
const h2 = t => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 80 },
  children: [new TextRun({ text: t, bold: true, size: 24, color: LBLUE, font: 'Arial' })]
});
const p = (t, extra={}) => new Paragraph({
  spacing: { before: 60, after: 60 },
  children: [new TextRun({ text: t, size: 20, font: 'Arial', color: BLACK, ...extra })]
});
const bullet = (t, bold=false) => new Paragraph({
  numbering: { reference: 'bullets', level: 0 }, spacing: { before: 40, after: 40 },
  children: [new TextRun({ text: t, size: 20, font: 'Arial', bold, color: BLACK })]
});
const sp = () => new Paragraph({ spacing:{before:60,after:60}, children:[new TextRun('')] });

// ── Colour box ────────────────────────────────────────────────────────────────
const box = (label, text, fill, tc=BLACK) => new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360],
  rows: [new TableRow({ children: [new TableCell({
    borders: bds, width: { size: 9360, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR }, margins: mg(100,100,150,150),
    children: [
      new Paragraph({ children: [new TextRun({ text: label, bold:true, size:18, color:tc, font:'Arial' })] }),
      new Paragraph({ spacing:{before:40}, children: [new TextRun({ text, size:20, color:BLACK, font:'Arial' })] }),
    ]
  })]})]
});

// ── 2-column table ────────────────────────────────────────────────────────────
const t2 = (rows, w1=2800, w2=6560) => new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: [w1, w2],
  rows: rows.map(([c1,c2,s1,s2]) => new TableRow({ children: [
    new TableCell({ borders:bds, width:{size:w1,type:WidthType.DXA},
      shading:{fill:s1||GREY,type:ShadingType.CLEAR}, margins:mg(),
      children:[new Paragraph({children:[new TextRun({text:c1,bold:true,size:18,font:'Arial',color:BLACK})]})] }),
    new TableCell({ borders:bds, width:{size:w2,type:WidthType.DXA},
      shading:{fill:s2||'FFFFFF',type:ShadingType.CLEAR}, margins:mg(),
      children:[new Paragraph({children:[new TextRun({text:c2,size:18,font:'Arial',color:BLACK})]})] }),
  ]}))
});

// ── 5-column comparison table ─────────────────────────────────────────────────
const cmpTable = (headers, rows, widths) => new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: widths,
  rows: [
    new TableRow({ children: headers.map((h,i) => new TableCell({
      borders:bds, width:{size:widths[i],type:WidthType.DXA},
      shading:{fill:BLUE,type:ShadingType.CLEAR}, margins:mg(70,70,90,90),
      children:[new Paragraph({children:[new TextRun({text:h,bold:true,size:16,color:'FFFFFF',font:'Arial'})]})]
    })) }),
    ...rows.map((row,ri) => new TableRow({ children: row.map((txt,ci) => new TableCell({
      borders:bds, width:{size:widths[ci],type:WidthType.DXA},
      shading:{fill: ci===headers.length-1 ? LGREEN : (ri%2===0?'FFFFFF':GREY), type:ShadingType.CLEAR},
      margins:mg(60,60,90,90),
      children:[new Paragraph({children:[new TextRun({text:txt,size:15,font:'Arial',bold:ci===headers.length-1,color:BLACK})]})]
    })) }))
  ]
});

// ═════════════════════════════════════════════════════════════════════════════
// Document
// ═════════════════════════════════════════════════════════════════════════════
const doc = new Document({
  numbering: { config: [{ reference:'bullets', levels:[{
    level:0, format:LevelFormat.BULLET, text:'•', alignment:AlignmentType.LEFT,
    style:{ paragraph:{ indent:{ left:720, hanging:360 } } }
  }] }] },
  styles: { default: { document: { run: { font:'Arial', size:20 } } } },
  sections: [{ properties: { page: {
    size: { width:12240, height:15840 },
    margin: { top:1080, right:1080, bottom:1080, left:1080 }
  } }, children: [

    // ── COVER ─────────────────────────────────────────────────────────────────
    new Paragraph({ alignment:AlignmentType.CENTER, spacing:{before:1200,after:200},
      children:[new TextRun({text:'TAGFC — NATOPS Research', bold:true, size:52, color:BLUE, font:'Arial'})] }),
    new Paragraph({ alignment:AlignmentType.CENTER, spacing:{before:80,after:80},
      children:[new TextRun({text:'Transformer Attention-Guided Frequency Counterfactual', size:30, color:LBLUE, font:'Arial', italics:true})] }),
    new Paragraph({ alignment:AlignmentType.CENTER, spacing:{before:80,after:80},
      children:[new TextRun({text:'Applied to NATOPS 6-class Gesture Recognition (UEA Archive)', size:22, color:DGREY, font:'Arial'})] }),
    new Paragraph({ alignment:AlignmentType.CENTER, spacing:{before:120,after:600},
      children:[new TextRun({text:'Complete Context for New Chat Session  |  PhD / MTP Thesis Work', size:20, color:DGREY, font:'Arial', italics:true})] }),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 1 — PROJECT OVERVIEW
    // ══════════════════════════════════════════════════════════════════════════
    h1('1. Project Overview'),
    box('One-Sentence Summary',
      'TAGFC generates minimal, sparse, physically-coherent counterfactual explanations for a Transformer-based multivariate time series classifier by combining (1) Attention Rollout saliency, (2) Haar wavelet-domain perturbation, (3) Mahalanobis cross-sensor coherence, and (4) Proximal Gradient Descent with soft-thresholding for exact L1 sparsity.',
      LLBLUE),
    sp(),
    h2('1.1 Research Goal'),
    p('Given a Transformer that classifies a 51-timestep, 24-channel gesture sequence into one of 6 NATOPS classes, produce a counterfactual X_cf — minimally different from the query X — such that the model predicts a different target class y*. The counterfactual should be (a) valid (actually flips prediction), (b) sparse (few coefficients changed), (c) physically plausible (respects inter-sensor correlations), and (d) interpretable in 3 dimensions: which sensor, which timestep, which frequency band.'),
    sp(),
    h2('1.2 Dataset: NATOPS'),
    t2([
      ['Full name',   'Naval Air Training and Operating Procedures Standardization'],
      ['Source',      'UEA Multivariate Time Series Archive'],
      ['Files',       'NATOPS_TRAIN.arff  |  NATOPS_TEST.arff'],
      ['Shape',       'T = 51 timesteps,  D = 24 sensor channels,  K = 6 gesture classes'],
      ['Sensors (24)','Right/Left: Hand (x,y,z), Elbow (x,y,z), Wrist (x,y,z), Thumb (x,y,z)'],
      ['Classes (6)', '0=I have command, 1=All clear, 2=Not clear, 3=Spread wings, 4=Fold wings, 5=Lock wings'],
      ['Preprocessing','Z-score normalise per channel (fit on TRAIN only). Labels re-indexed 0..5.'],
      ['Location',    'data/raw/NATOPS/NATOPS_TRAIN.arff  and  NATOPS_TEST.arff'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 2 — CODEBASE STRUCTURE
    // ══════════════════════════════════════════════════════════════════════════
    h1('2. Codebase Structure — 4 Folders'),
    t2([
      ['TAGFC_Natops/',   'MAIN FOLDER. Full TAGFC pipeline + all 3 baselines in one place. Has run successfully — outputs, figures, model, and pkl results all saved.', LLBLUE],
      ['Comte_Natops/',   'Standalone CoMTE-only pipeline. Separate Transformer trained. Own figures (F1-F11). Already run.', LGREEN],
      ['AB_CF_Natops/',   'Standalone AB-CF-only pipeline. Separate Transformer trained. Own figures (F1-F13). Already run.', LGREEN],
      ['NATOPS_XAI/',     'Earlier combined version (TAGFC + baselines). Also has results_natops.pkl and figures. Predecessor to TAGFC_Natops/.', GREY],
    ], 2200, 7160),
    sp(),
    h2('2.1 TAGFC_Natops/ — File Map'),
    t2([
      ['config.py',          'Single Config dataclass. All hyperparameters, paths, dataset constants. Singleton cfg = Config().'],
      ['data_pipeline.py',   'load_natops() — reads .arff, strips labels, z-scores. get_dataloaders() — PyTorch DataLoader wrappers.'],
      ['transformer_model.py','TransformerNATOPS class. Architecture + training utilities + attention_rollout() + input_gradient().'],
      ['haar_wavelet.py',    'HaarWT class. dwt2d(X:(T,D)) → (D,L). idwt2d(C:(D,L)) → (T,D). Pure NumPy, 3 levels, T=51 → L=64.'],
      ['tagfc_core.py',      'compute_omega() function + TAGFCOptimizer class. The full TAGFC algorithm.'],
      ['baselines.py',       'CoMTE class (greedy channel swap) + ABCF class (entropy segment swap). Both use NUN from training set.'],
      ['evaluation.py',      'compute_metrics() (8 metrics per pair) + aggregate() + run_wilcoxon() with Bonferroni correction.'],
      ['visualization.py',   '13 figure-generating functions (F1-F13). All save to outputs/figures/.'],
      ['main.py',            '7-step pipeline: data → train → explain (TAGFC+CoMTE+AB-CF) → metrics → Wilcoxon → figures.'],
      ['outputs/results_natops_tagfc.pkl', 'Saved raw results dict: keys tagfc, comte, abcf, tagfc_cfs.'],
      ['outputs/models/transformer_natops_tagfc.pt', 'Saved PyTorch state dict for TransformerNATOPS.'],
      ['outputs/figures/',   'F1-F13 PNG figures — training curves, confusion matrix, CF explanations, metrics, Wilcoxon scatter.'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 3 — TRANSFORMER ARCHITECTURE
    // ══════════════════════════════════════════════════════════════════════════
    h1('3. TransformerNATOPS Architecture'),
    t2([
      ['Input',        '(B, T=51, D=24) batch of gesture sequences'],
      ['Projection',   'Linear(24 → 64) + sinusoidal positional encoding (SinPE)'],
      ['Encoder',      '3 × TransformerEncoderLayer(d_model=64, nhead=4, d_ff=128, dropout=0.1, norm_first=True, batch_first=True)'],
      ['Pooling',      'LayerNorm → MeanPool over T dimension → Dropout(0.1)'],
      ['Output head',  'Linear(64 → 6) → logits (B, 6)'],
      ['Training',     'AdamW(lr=1e-3, wd=1e-4) + CosineAnnealingLR, 200 epochs, batch=32, CrossEntropyLoss'],
      ['Best model',   'Checkpoint saved when val_acc improves. Restored at end of training.'],
    ]),
    sp(),
    h2('3.1 attention_rollout(x_np) — Abnar & Zuidema (2020)'),
    t2([
      ['Input',   'x_np: (T=51, D=24) numpy array'],
      ['Step 1',  'Forward pass with capture_attn=True. Each layer manually runs self_attn(need_weights=True, average_attn_weights=False) to capture per-head (H, T, T) attention matrices.'],
      ['Step 2',  'Average over H heads per layer: A_mean = attn[0].mean(dim=0) → (T, T)'],
      ['Step 3',  'Residual correction: A_hat = 0.5 * A_mean + 0.5 * I_T'],
      ['Step 4',  'Row-normalise: A_hat = A_hat / rowsum(A_hat)'],
      ['Step 5',  'Rollout: R = I_T; for each layer: R = R @ A_hat. Result: (T, T)'],
      ['Output',  's = R.sum(dim=0) → (T,). Min-max normalise to [0,1]. High s_t = Transformer attended heavily to timestep t.'],
    ]),
    sp(),
    h2('3.2 input_gradient(x_np, target_class)'),
    t2([
      ['Purpose', 'Compute |∂logit[target_class] / ∂X| — how sensitive the model output is to each input feature.'],
      ['Method',  'Enable requires_grad on input tensor. Forward pass. Call .backward() on logit of target class. Return x.grad[0]: (T, D).'],
      ['Used by', 'compute_omega() in tagfc_core.py to build the wavelet-domain gradient importance map.'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 4 — HAAR WAVELET
    // ══════════════════════════════════════════════════════════════════════════
    h1('4. Haar Wavelet Transform (haar_wavelet.py)'),
    t2([
      ['Class',       'HaarWT(levels=3)'],
      ['dwt2d(X)',     'Input: (T=51, D=24). For each of D channels independently: apply 3-level Haar DWT. Pads to next power-of-2 if needed. Returns C: (D=24, L=64) — all coefficients concatenated.'],
      ['idwt2d(C)',    'Input: (D=24, L=64). Reconstructs (T, D) time series. Haar is orthogonal so iDWT is exact inverse.'],
      ['Coefficient layout', 'L=64 coefficients per sensor: [approx_L3 | detail_L3 | detail_L2 | detail_L1]. Approx = slow motion trend. Detail_L1 = fast joint transitions.'],
      ['Key property', 'Orthogonality: gradient w.r.t. coefficients = DWT(gradient w.r.t. X). Allows converting time-domain gradients to coefficient-domain gradients exactly.'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 5 — TAGFC ALGORITHM (FULL DETAIL)
    // ══════════════════════════════════════════════════════════════════════════
    h1('5. TAGFC Algorithm — Complete 5-Step Detail'),

    h2('Step 1: Attention Rollout → Temporal Saliency s ∈ [0,1]^T'),
    box('Formula',
      'A_hat^l = 0.5 * A^l + 0.5 * I_T   (per layer, averaged over heads)\nRollout = I_T @ A_hat^1 @ A_hat^2 @ ... @ A_hat^L\ns_t = column_sum(Rollout),  then min-max normalise to [0,1]\nHigh s_t = model attends to timestep t. Low s_t = model ignores timestep t.',
      LLBLUE),
    sp(),

    h2('Step 2: Haar DWT → Coefficient Space C ∈ R^{D×L}'),
    box('Formula',
      'C = HaarWT.dwt2d(X)   shape: (D=24, L=64)\nSeparates each sensor into 3 frequency scales:\n  Approx (L3): slow overall gesture trajectory\n  Detail L3: medium-speed joint transitions\n  Detail L1: fast tremor / noise\nPerturbation delta acts in this coefficient space.',
      LLBLUE),
    sp(),

    h2('Step 3: Omega Weights → Per-Coefficient Penalty ω ∈ [0,1]^{D×L}'),
    box('Formula',
      'G = |∂logit_{y_tgt}/∂X|                   (input gradient, shape T×D)\nG_smooth = GaussianFilter(G, sigma=1.5)      (smooth noise)\nG_norm = G_smooth / G_smooth.max()           (normalise to [0,1])\nG_wav = HaarWT.dwt2d(|G_norm|)               (wavelet-domain gradient, shape D×L)\nG_wav = G_wav / G_wav.max()                  (normalise)\ns_coeff[l] = s[round(l/L * T)]               (map temporal saliency to coeff positions)\nM = s_coeff[None,:] * G_wav + 1e-8           (joint attention × gradient importance)\nomega = 1/M,  then normalise to [0,1]\nLow omega = IMPORTANT coefficient (expensive to change).\nHigh omega = UNIMPORTANT coefficient (cheap to change freely).',
      LLBLUE),
    sp(),

    h2('Step 4: Four-Term Objective'),
    box('Formula',
      'delta* = argmin_delta  L_flip + lambda1*L_sparse + lambda2*L_cross + lambda3*L_manifold\n\nL_flip     = -log P(y_tgt | iDWT(C + delta))     [push model toward target class]\nL_sparse   = sum_{d,l} omega_{d,l} * |delta_{d,l}|  [omega-weighted L1 sparsity]\nL_cross    = diff^T * Sigma^{-1} * diff             [Mahalanobis cross-sensor coherence]\n             where diff = mean_T(X_cf - X)  shape (D,)\nL_manifold = ||X_cf - X_nn||^2_F / (T*D)           [stay near real training data]\n             where X_nn = nearest training sample of class y_tgt\n\nHyperparameters: lambda1=0.02, lambda2=0.01, lambda3=0.10\nSigma_inv computed from training data: cov(X_train.reshape(-1, D)) + 1e-4*I, then inverted.',
      LLBLUE),
    sp(),

    h2('Step 5: Proximal Gradient Descent with Soft-Thresholding'),
    box('Formula',
      'For k = 1 ... MAX_ITER=300:\n  X_cf = iDWT(C + delta)\n  grad_flip     = backward(-log P(y_tgt | X_cf)) w.r.t. X_cf   [shape: T×D]\n  grad_cross    = Sigma^{-1} @ mean_T(X_cf - X) / T            [shape: D, broadcast]\n  grad_manifold = 2*(X_cf - X_nn) / (T*D)                      [shape: T×D]\n  grad_X = grad_flip + lambda2*grad_cross + lambda3*grad_manifold\n  grad_delta = HaarWT.dwt2d(grad_X)             [convert to coeff domain]\n\n  Gradient step:  delta_new = delta - LR_OPT * grad_delta\n  Soft threshold: delta_prox = sign(delta_new) * max(|delta_new| - LR_OPT*lambda1*omega, 0)\n  Bound clip:     delta = clip(delta_prox, -2.5, +2.5)\n\n  If model predicts y_tgt: patience++. Stop when patience >= PATIENCE=25.\n\nOutput: X_cf_final = iDWT(C + best_delta)',
      LLBLUE),
    sp(),

    h2('5.1 3D Explanation Output'),
    t2([
      ['Sensor importance (D,)',   'sensor_imp[d] = sum_l |best_delta[d,l]|  — which of 24 sensors was changed most'],
      ['Temporal saliency (T,)',   'temporal_saliency = attention_rollout(X)  — which of 51 timesteps model attended to'],
      ['Frequency importance (L,)','freq_imp[l] = sum_d |best_delta[d,l]|  — which wavelet band (slow/medium/fast) was modified'],
      ['Raw delta (D×L)',         'best_delta: the full wavelet-coefficient perturbation matrix'],
      ['Omega (D×L)',              'omega: the penalty weight matrix (visualised as F9 heatmap)'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 6 — BASELINES
    // ══════════════════════════════════════════════════════════════════════════
    h1('6. Baselines (baselines.py)'),

    h2('6.1 CoMTE — Ates et al. (ICAPAI 2021)'),
    box('Citation', 'Ates, E., Aksar, B., Leung, V.J., Coskun, A.K. "Counterfactual Explanations for Multivariate Time Series." ICAPAI 2021. DOI: 10.1109/ICAPAI49758.2021.9462056', GREY),
    t2([
      ['Core idea',   'Find Nearest Unlike Neighbour (NUN) of target class. Greedily swap entire channels (all T timesteps) from NUN. At each step, pick the unswapped channel that maximises P(y_tgt | X_cf). Stop when prediction flips.'],
      ['Implementation', 'class CoMTE. explain(X, y_tgt) → (X_cf, info). NUN found by L2 distance in flattened space. Greedy loop over D channels. Returns channels_swapped count.'],
      ['Limitation',  'Swaps ENTIRE channels — no temporal or frequency granularity. Explanation = only "which sensor", not "when" or "which frequency".'],
      ['Key difference from TAGFC', 'CoMTE: binary channel-swap decision. TAGFC: continuous wavelet-coefficient perturbation guided by attention saliency.'],
    ]),
    sp(),

    h2('6.2 AB-CF — Li et al. (DaWaK 2023)'),
    box('Citation', 'Li, P., Bahri, O., Boubrahimi, S.F., Hamdi, S.M. "Attention-Based Counterfactual Explanation for Multivariate Time Series." DaWaK 2023. DOI: 10.1007/978-3-031-39831-5_26', GREY),
    t2([
      ['Core idea',   'Find NUN of target class. Slide window of length L=int(0.15*T)~7 over all D channels and time segments. Zero-pad each (channel, segment) probe and feed to model. Rank by Shannon entropy H = -sum(p_c * log2(p_c)). Lowest entropy = most discriminative. Swap from NUN in that order until flip.'],
      ['Implementation', 'class ABCF(window_frac=0.15). explain(X, y_tgt) → (X_cf, info). Generates D * ceil(T/L) candidates. Sorts ascending by entropy. Swaps in order.'],
      ['Limitation',  'Window-level granularity only. No frequency domain. No cross-sensor coherence. Single attention metric (entropy).'],
      ['Key difference from TAGFC', 'AB-CF uses model output entropy as proxy for importance. TAGFC uses Transformer attention rollout across ALL layers combined with input gradient in wavelet domain.'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 7 — EVALUATION
    // ══════════════════════════════════════════════════════════════════════════
    h1('7. Evaluation Framework (evaluation.py)'),
    h2('7.1 Eight Metrics'),
    t2([
      ['validity',       'Did CF flip prediction? (y_tgt == argmax P(y|X_cf)). Binary 0/1.   HIGHER IS BETTER.'],
      ['proximity_l1',   'L1 distance ||X_cf - X_orig||_1. LOWER IS BETTER.'],
      ['proximity_l2',   'L2 distance ||X_cf - X_orig||_2. LOWER IS BETTER.'],
      ['proximity_linf', 'L-infinity distance max|X_cf - X_orig|. LOWER IS BETTER.'],
      ['sparsity',       'Fraction of (T,D) features unchanged: mean(|X_cf - X_orig| < 1e-6). HIGHER IS BETTER.'],
      ['coherence',      'Mahalanobis distance: mean_T(X_cf-X)^T * Sigma_inv * mean_T(X_cf-X). LOWER IS BETTER.'],
      ['cf_confidence',  'Model confidence on target class P(y_tgt | X_cf). HIGHER IS BETTER.'],
      ['rcf',            'Relative CF distance: L2(X,X_cf) / L2(X,NUN). < 1 means CF closer than NUN. LOWER IS BETTER.'],
    ]),
    sp(),
    h2('7.2 Statistical Tests'),
    t2([
      ['Method',       'Paired Wilcoxon signed-rank test (non-parametric, does not assume normality).'],
      ['Correction',   'Bonferroni: alpha_corr = 0.05 / 6 metrics = 0.0083.'],
      ['Comparisons',  'TAGFC vs CoMTE  AND  TAGFC vs AB-CF. Both run in main.py step 6.'],
      ['Reporting',    '*** TAGFC better (p < alpha_corr AND TAGFC mean better). (!!) baseline better. Blank = not significant.'],
      ['Raw values',   'All per-sample metric values saved in results_natops_tagfc.pkl for offline analysis.'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 8 — WHAT HAS BEEN RUN
    // ══════════════════════════════════════════════════════════════════════════
    h1('8. Current Status — What Has Already Been Run'),
    box('All three pipelines have been executed',
      'TAGFC_Natops/main.py has completed. Model trained, counterfactuals generated for N_EXPLAIN=30 test samples (5 per class, balanced), all 3 methods compared, Wilcoxon tests run, all 13 figures saved.',
      LGREEN),
    sp(),
    t2([
      ['TAGFC_Natops/outputs/results_natops_tagfc.pkl', 'Raw results dict with keys: tagfc, comte, abcf, tagfc_cfs. Contains per-sample metric dicts.'],
      ['TAGFC_Natops/outputs/models/transformer_natops_tagfc.pt', 'Trained TransformerNATOPS state dict. Loads without retraining.'],
      ['TAGFC_Natops/outputs/figures/F1_training_curves.png', 'Train/val loss and accuracy over 200 epochs.'],
      ['F2_confusion_matrix.png', '6x6 confusion matrix on test set.'],
      ['F3_class_distribution.png', 'Class balance in train/test.'],
      ['F4_cf_timeseries_sample{0,1,2}.png', 'Original vs CF overlay for first 3 samples (24 sensors shown).'],
      ['F5_temporal_saliency_sample{0,1,2}.png', 'Attention rollout saliency bar chart across T=51 timesteps.'],
      ['F6_sensor_importance_sample{0,1,2}.png', 'Per-sensor importance V_d across 24 channels.'],
      ['F7_freq_importance_sample{0,1,2}.png', 'Per-frequency-band importance F_l across L=64 coefficients.'],
      ['F8_delta_heatmap_sample{0,1,2}.png', 'Heatmap of best_delta (D×L) — what was perturbed where in wavelet space.'],
      ['F9_omega_heatmap_sample{0,1,2}.png', 'Heatmap of omega (D×L) — the penalty weight / importance map.'],
      ['F10_metric_comparison.png', 'Bar chart: TAGFC vs CoMTE vs AB-CF on all 8 metrics (with error bars).'],
      ['F11_proximity_validity_scatter.png', 'Scatter: proximity_l1 vs validity rate for all 3 methods.'],
      ['F12_per_class_validity.png', 'Validity rate broken down by original gesture class (6 classes).'],
      ['F13_convergence_sample{0,1,2}.png', 'Optimisation history: L_flip, L_sparse, L_cross, L_manifold per iteration.'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 9 — WHY TAGFC IS BETTER
    // ══════════════════════════════════════════════════════════════════════════
    h1('9. Why TAGFC Is Better — 5 Arguments'),
    cmpTable(
      ['Property', 'CoMTE', 'AB-CF', 'TAGFC (yours)'],
      [
        ['Model access',           'Black box (NUN only)', 'Black box (entropy probe)', 'White box — Attention Rollout ALL L layers ALL H heads'],
        ['Granularity',            'Whole channel (T timesteps)', 'Window segment (~7 timesteps)', 'Wavelet coefficient (D×L=24×64 space)'],
        ['Frequency domain',       'No', 'No', 'YES — Haar DWT decomposes into slow/medium/fast bands'],
        ['Cross-sensor coherence', 'No', 'No', 'YES — Mahalanobis Sigma^{-1} penalises physically impossible changes'],
        ['Sparsity method',        'Binary channel swap', 'Greedy window swap', 'Exact L1 via Proximal GD soft-thresholding'],
        ['Explanation output',     '1D: sensor only', '2D: sensor + segment', '3D: sensor + timestep + frequency band'],
        ['Saliency metric',        'None', 'Shannon entropy (output only)', 'Attention rollout × input gradient in wavelet space'],
      ],
      [2800, 1800, 1800, 2960]
    ),
    sp(),
    box('Argument 1 — Model-Aware Saliency (All Layers)',
      'TAGFC uses Attention Rollout (Abnar & Zuidema ACL 2020) which multiplies residual-corrected attention matrices across ALL L=3 Transformer layers and averages over ALL H=4 heads. CoMTE and AB-CF treat the model as a black box. AB-CF uses model output entropy but only as a single-layer probe. TAGFC reads what the Transformer genuinely attended to across all its reasoning steps.',
      LGREEN),
    sp(),
    box('Argument 2 — Frequency Domain Perturbation',
      'No existing MTS counterfactual method operates in the frequency domain. TAGFC uses Haar DWT to decompose each of 24 gesture channels into approximation (slow overall motion) and detail coefficients (fast transitions). This allows TAGFC to explain WHICH FREQUENCY BAND of motion changed the prediction — e.g., "the slow trend in right-hand x changed, not the fast tremor". CoMTE and AB-CF can only say WHICH segment changed.',
      LGREEN),
    sp(),
    box('Argument 3 — Cross-Sensor Physical Coherence',
      'Arm joints are mechanically constrained: elbow and wrist positions are physically correlated. TAGFC enforces L_cross = diff^T * Sigma^{-1} * diff where Sigma is the inter-sensor covariance of real training data. This penalises counterfactuals that violate these constraints — e.g., right elbow moving without corresponding right wrist movement. Neither CoMTE nor AB-CF have any such constraint.',
      LGREEN),
    sp(),
    box('Argument 4 — Mathematically Exact L1 Sparsity',
      'TAGFC uses Proximal Gradient Descent. The soft-threshold operation delta_prox = sign(delta) * max(|delta| - eta*lambda1*omega, 0) sets coefficients EXACTLY to zero when they are below threshold — this is the exact mathematical solution to the L1 proximal operator. AB-CF achieves window-level sparsity (replaces whole segments). CoMTE achieves channel-level sparsity (replaces whole channels). TAGFC achieves coefficient-level sparsity guided by attention importance weights.',
      LGREEN),
    sp(),
    box('Argument 5 — 3-Dimensional Explanation',
      'TAGFC is the only method that simultaneously answers three questions: (1) WHICH SENSOR changed? (sensor_importance V_d, shape D=24) (2) WHEN did the model attend? (temporal_saliency s_t, shape T=51) (3) WHICH FREQUENCY BAND was modified? (freq_importance F_l, shape L=64). CoMTE answers only (1). AB-CF answers (1)+(2) partially. TAGFC answers all three with separate, interpretable outputs.',
      LGREEN),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 10 — HYPERPARAMETERS
    // ══════════════════════════════════════════════════════════════════════════
    h1('10. Key Hyperparameters (config.py)'),
    t2([
      ['T = 51',                   'Timesteps per gesture sequence'],
      ['D = 24',                   'Sensor channels'],
      ['K = 6',                    'Gesture classes'],
      ['WAV_LEVELS = 3',           'Haar DWT levels. T=51 → L=64 coefficients per sensor (padded to 2^6).'],
      ['D_MODEL = 64',             'Transformer embedding dimension'],
      ['N_HEADS = 4',              'Attention heads per layer'],
      ['N_LAYERS = 3',             'Transformer encoder layers'],
      ['D_FF = 128',               'Feed-forward hidden dimension'],
      ['TRAIN_LR = 1e-3',          'AdamW learning rate'],
      ['EPOCHS = 200',             'Training epochs'],
      ['LAMBDA_SPARSE = 0.02',     'lambda1: weight of omega-weighted L1 sparsity term'],
      ['LAMBDA_CROSS = 0.01',      'lambda2: weight of Mahalanobis cross-sensor coherence term'],
      ['LAMBDA_MANIFOLD = 0.10',   'lambda3: weight of nearest-neighbour manifold plausibility term'],
      ['LR_OPT = 0.05',            'Proximal GD step size (eta)'],
      ['MAX_ITER = 300',           'Maximum optimisation iterations per sample'],
      ['PATIENCE = 25',            'Stop after this many consecutive flipped iterations'],
      ['DELTA_BOUND = 2.5',        'Hard clip on wavelet coefficient perturbation magnitude'],
      ['GRAD_SMOOTH_SIGMA = 1.5',  'Gaussian smoothing sigma for input gradient'],
      ['N_EXPLAIN = 30',           'Number of test samples explained (5 per class, balanced)'],
      ['ABCF_WINDOW_FRAC = 0.15',  'AB-CF window length as fraction of T (0.15*51 ~ 7 timesteps)'],
    ]),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 11 — HOW TO RUN
    // ══════════════════════════════════════════════════════════════════════════
    h1('11. How to Run / Continue'),
    h2('11.1 Re-run Full Pipeline'),
    box('Commands',
      'cd counterfactual_basis_kernel-main/TAGFC_Natops\npython main.py\n\nModel will be loaded from outputs/models/ (already trained — no retraining).\nAll 3 methods will re-run on N_EXPLAIN=30 samples.\nResults saved to outputs/results_natops_tagfc.pkl.\nAll figures regenerated.',
      LLBLUE),
    sp(),
    h2('11.2 Increase Sample Size for Publication'),
    bullet('Change N_EXPLAIN = 30 → 200 in config.py for publishable Wilcoxon test power'),
    bullet('Wilcoxon needs n >= 20 per comparison. 200 samples gives robust statistics.'),
    bullet('Bonferroni alpha_corr = 0.05 / 6 = 0.0083. Report p-values with this threshold.'),
    sp(),
    h2('11.3 Load and Inspect Saved Results'),
    box('Python snippet',
      'import pickle\nwith open("TAGFC_Natops/outputs/results_natops_tagfc.pkl", "rb") as f:\n    data = pickle.load(f)\ntagfc_results = data["tagfc"]   # list of per-sample metric dicts\ncomte_results = data["comte"]\nabcf_results  = data["abcf"]\ntagfc_cfs     = data["tagfc_cfs"]  # list of (X_orig, X_cf, info) tuples\n\n# Print mean validity:\nimport numpy as np\nprint("TAGFC validity:", np.mean([r["validity"] for r in tagfc_results]))\nprint("CoMTE validity:", np.mean([r["validity"] for r in comte_results]))\nprint("AB-CF validity:", np.mean([r["validity"] for r in abcf_results]))',
      GREY),
    sp(),
    h2('11.4 Run Ablation Study'),
    bullet('TAGFC-full: current method (lambda1=0.02, lambda2=0.01, lambda3=0.10) — BASELINE'),
    bullet('TAGFC-noRollout: set omega = ones (uniform weights everywhere, no attention guidance)'),
    bullet('TAGFC-noWavelet: perturb raw time-domain X instead of wavelet coefficients'),
    bullet('TAGFC-noCross: set LAMBDA_CROSS = 0.0 (remove Mahalanobis coherence term)'),
    bullet('TAGFC-noManifold: set LAMBDA_MANIFOLD = 0.0 (remove nearest-neighbour term)'),
    bullet('Each ablation that performs worse PROVES that component contributes to the method'),
    sp(),
    new Paragraph({ children:[new PageBreak()] }),

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION 12 — WHAT TO TELL THE NEW ASSISTANT
    // ══════════════════════════════════════════════════════════════════════════
    h1('12. Context to Give New Chat Session'),
    box('Copy-paste this at the start of your new chat',
      'I am a PhD/MTP student. I have implemented TAGFC (Transformer Attention-Guided Frequency Counterfactual), a novel XAI method for multivariate time series. It is applied to the NATOPS 6-class gesture dataset (T=51, D=24 sensors, K=6 classes from the UEA archive).\n\nMy codebase is at: counterfactual_basis_kernel-main/TAGFC_Natops/\nKey files: tagfc_core.py (optimizer), transformer_model.py (attention rollout), haar_wavelet.py (DWT), baselines.py (CoMTE + AB-CF), evaluation.py (Wilcoxon), main.py (full pipeline).\n\nThe pipeline has ALREADY been run on N_EXPLAIN=30 samples. Results are saved in outputs/results_natops_tagfc.pkl. Figures F1-F13 are in outputs/figures/. Trained model is in outputs/models/transformer_natops_tagfc.pt.\n\nTAGFC has 5 steps: (1) Attention Rollout → saliency, (2) Haar DWT → coefficient space, (3) Omega weights (attention × gradient), (4) 4-term objective (flip + sparse + cross + manifold), (5) Proximal GD with soft-thresholding.\n\nHyperparams: lambda1=0.02, lambda2=0.01, lambda3=0.10, LR=0.05, MAX_ITER=300, PATIENCE=25, DELTA_BOUND=2.5, WAV_LEVELS=3 (T=51 → L=64 coefficients).',
      LAMBER, BLACK),
    sp(),
    bullet('Working directory: c:\\Users\\abhia\\Desktop\\counterfactual_basis_kernel-main', true),
    bullet('NATOPS data: data/raw/NATOPS/NATOPS_TRAIN.arff + NATOPS_TEST.arff', true),
    bullet('N_EXPLAIN=30 done. Need N_EXPLAIN=200 for publication-quality statistics.', true),
    bullet('Ablation study (5 variants) not yet run — needed to justify each TAGFC component.', true),
    bullet('Standalone folders also exist: Comte_Natops/, AB_CF_Natops/, NATOPS_XAI/', true),
    bullet('All three methods (TAGFC, CoMTE, AB-CF) run together in TAGFC_Natops/main.py', true),
    sp(),

    h2('12.1 Pending Work (Priority Order)'),
    t2([
      ['1. Scale to N=200',     'Change N_EXPLAIN=200 in config.py. Re-run main.py. Required for Wilcoxon statistical power.', LGREEN],
      ['2. Ablation study',     'Run 5 TAGFC variants (noRollout, noWavelet, noCross, noManifold, full). Proves each component is necessary.', LGREEN],
      ['3. MMD plausibility',   'Compute Maximum Mean Discrepancy between CF distribution and training distribution. Currently only Mahalanobis coherence is computed.', LAMBER],
      ['4. Paper writing',      'Chapter 4: TAGFC method. Chapter 5: Experiments. Chapter 3: Related work (Native Guide, CoMTE, AB-CF, SETS, Attention Rollout).', LAMBER],
      ['5. Human evaluation',   'User study: ask humans if CFs are "believable". Required before top-tier conference submission.', LRED],
    ], 2400, 6960),
    sp(),

    h2('12.2 Key Citations'),
    t2([
      ['Attention Rollout', 'Abnar, S., Zuidema, W. "Quantifying Attention Flow in Transformers." ACL 2020. arXiv:2005.00928'],
      ['CoMTE',             'Ates et al. "Counterfactual Explanations for Multivariate Time Series." ICAPAI 2021. DOI: 10.1109/ICAPAI49758.2021.9462056'],
      ['AB-CF',             'Li et al. "Attention-Based Counterfactual Explanation for Multivariate Time Series." DaWaK 2023. DOI: 10.1007/978-3-031-39831-5_26'],
      ['Native Guide',      'Delaney et al. "Instance-based Counterfactual Explanations for Time Series Classification." ICCBR 2021. arXiv:2009.13211'],
      ['SETS',              'Bahri et al. "Shapelet-Based Counterfactual Explanations for Multivariate Time Series." KDD MiLeTS 2022. arXiv:2208.10462'],
      ['Haar Wavelet',      'Mallat, S. "A Wavelet Tour of Signal Processing." 3rd ed. Academic Press, 2009.'],
      ['Proximal GD',       'Parikh, N., Boyd, S. "Proximal Algorithms." Foundations and Trends in Optimization. 2014.'],
      ['NATOPS dataset',    'UEA Multivariate Time Series Classification Archive. Bagnall et al. arXiv:1811.00075'],
    ]),

    sp(),
    new Paragraph({ alignment:AlignmentType.CENTER, spacing:{before:300},
      children:[new TextRun({text:'End of Summary — Ready for New Chat Session', size:18, color:DGREY, font:'Arial', italics:true})]
    }),

  ]}]
});

// ── Write file ────────────────────────────────────────────────────────────────
const outPath = path.join(__dirname, 'TAGFC_NATOPS_Summary.docx');
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outPath, buf);
  console.log('DONE:', outPath);
}).catch(err => {
  console.error('ERROR:', err.message);
  process.exit(1);
});
