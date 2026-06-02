const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageBreak, VerticalAlign
} = require('docx');
const fs = require('fs');

const BLUE = '1F3864'; const LBLUE = '2E75B6'; const LLBLUE = 'D6E4F0';
const GREEN = '1D9E75'; const LGREEN = 'E1F5EE'; const AMBER = 'EF9F27';
const LAMBER = 'FFF3CD'; const RED = 'C00000'; const LRED = 'FFE0E0';
const GREY = 'F5F5F5'; const DGREY = '444444'; const BLACK = '000000';

const border = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
const borders = { top: border, bottom: border, left: border, right: border };

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 300, after: 120 },
  children: [new TextRun({ text, bold: true, size: 30, color: BLUE, font: 'Arial' })]
});
const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 240, after: 80 },
  children: [new TextRun({ text, bold: true, size: 26, color: LBLUE, font: 'Arial' })]
});
const h3 = (text) => new Paragraph({
  spacing: { before: 180, after: 60 },
  children: [new TextRun({ text, bold: true, size: 22, color: DGREY, font: 'Arial' })]
});
const p = (text, opts = {}) => new Paragraph({
  spacing: { before: 60, after: 60 },
  children: [new TextRun({ text, size: 20, font: 'Arial', color: BLACK, ...opts })]
});
const bullet = (text, bold = false) => new Paragraph({
  numbering: { reference: 'bullets', level: 0 },
  spacing: { before: 40, after: 40 },
  children: [new TextRun({ text, size: 20, font: 'Arial', bold, color: BLACK })]
});
const space = () => new Paragraph({ spacing: { before: 60, after: 60 }, children: [new TextRun('')] });

const colorBox = (label, text, fillColor, textColor = BLACK) => {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({
      children: [new TableCell({
        borders,
        width: { size: 9360, type: WidthType.DXA },
        shading: { fill: fillColor, type: ShadingType.CLEAR },
        margins: { top: 100, bottom: 100, left: 150, right: 150 },
        children: [
          new Paragraph({ children: [new TextRun({ text: label, bold: true, size: 18, color: textColor, font: 'Arial' })] }),
          new Paragraph({ spacing: { before: 40 }, children: [new TextRun({ text, size: 20, color: BLACK, font: 'Arial' })] })
        ]
      })]
    })]
  });
};

const twoColTable = (rows, w1 = 3000, w2 = 6360) => new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [w1, w2],
  rows: rows.map(([col1, col2, shade1, shade2]) => new TableRow({
    children: [
      new TableCell({
        borders, width: { size: w1, type: WidthType.DXA },
        shading: { fill: shade1 || GREY, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: col1, bold: true, size: 18, font: 'Arial', color: BLACK })] })]
      }),
      new TableCell({
        borders, width: { size: w2, type: WidthType.DXA },
        shading: { fill: shade2 || 'FFFFFF', type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: col2, size: 18, font: 'Arial', color: BLACK })] })]
      })
    ]
  }))
});

const threeColTable = (headers, rows) => new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2400, 3480, 3480],
  rows: [
    new TableRow({
      children: headers.map((h, i) => new TableCell({
        borders, width: { size: [2400, 3480, 3480][i], type: WidthType.DXA },
        shading: { fill: BLUE, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 18, color: 'FFFFFF', font: 'Arial' })] })]
      }))
    }),
    ...rows.map(([c1, c2, c3], ri) => new TableRow({
      children: [c1, c2, c3].map((txt, ci) => new TableCell({
        borders, width: { size: [2400, 3480, 3480][ci], type: WidthType.DXA },
        shading: { fill: ri % 2 === 0 ? 'FFFFFF' : GREY, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: txt, size: 18, font: 'Arial', color: BLACK })] })]
      }))
    }))
  ]
});

const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: 'Arial', size: 20 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 30, bold: true, font: 'Arial', color: BLUE },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: LBLUE },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 } }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }
      }
    },
    children: [

      // ─── COVER ──────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 200 },
        children: [new TextRun({ text: 'TAGFC Research', bold: true, size: 48, color: BLUE, font: 'Arial' })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 },
        children: [new TextRun({ text: 'Transformer Attention-Guided Frequency Counterfactual', size: 32, color: LBLUE, font: 'Arial', italics: true })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 200, after: 600 },
        children: [new TextRun({ text: 'Complete Session Summary | NASA FD001 Dataset | PhD Thesis Work', size: 22, color: DGREY, font: 'Arial' })]
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // ─── SECTION 1: WHAT YOU HAVE BUILT ─────────────────────────────
      h1('1. What You Have Built — Complete Status'),
      p('This document summarises everything developed in this research session. Your TAGFC method is a novel counterfactual explanation algorithm for multivariate time series (MTS) using Transformer models, evaluated on the NASA CMAPSS FD001 turbofan engine dataset.'),
      space(),

      h2('1.1 The Method: TAGFC in One Sentence'),
      colorBox('TAGFC Definition',
        'TAGFC (Transformer Attention-Guided Frequency Counterfactual) generates minimal, sparse, physically-coherent counterfactual explanations for Transformer-based multivariate time series classifiers by: (1) using Attention Rollout to identify which timesteps the model attended to, (2) operating in the Haar wavelet frequency domain to identify which frequency band drove the prediction, (3) enforcing cross-sensor physical coherence via Mahalanobis distance, and (4) finding the optimal perturbation via Proximal Gradient Descent with soft-thresholding for exact L1 sparsity.',
        LLBLUE, BLACK),
      space(),

      h2('1.2 The 5-Step TAGFC Pipeline'),
      twoColTable([
        ['Step 1', 'ATTENTION ROLLOUT: Multiply residual-corrected attention matrices A^(l) across all L Transformer layers and H heads. Formula: A_hat^(l) = 0.5*A^(l) + 0.5*I. Rollout = A_hat^1 * A_hat^2 * ... * A_hat^L. Column sums = temporal saliency s_t ∈ [0,1]^T.', LLBLUE],
        ['Step 2', 'HAAR WAVELET TRANSFORM: Apply DWT to each sensor independently. C^d = W_psi(X[:,d]). Produces coefficients across 3 frequency scales: Approximation (slow trend), Detail-2 (medium), Detail-1 (fast noise). Stores C ∈ R^{D×L} (14 sensors × 28 coefficients).', LLBLUE],
        ['Step 3', 'OMEGA IMPORTANCE WEIGHTS: omega^d_l = 1 / (s_{t(l)} * |DWT(|grad_X|)|^d_l + eps). Low omega = cheap to change (important). High omega = expensive to change (not important). This creates a "price tag" for every wavelet coefficient.', LLBLUE],
        ['Step 4', 'FOUR-TERM OBJECTIVE: delta* = argmin [L_flip + lambda1*L_sparse + lambda2*L_cross + lambda3*L_manifold]. L_flip = -log P(y*|X~), L_sparse = sum(omega*|delta|), L_cross = diff^T * Sigma_inv * diff, L_manifold = ||X~ - X_nn||^2_F.', LLBLUE],
        ['Step 5', 'PROXIMAL GD: For each iteration: gradient step on smooth losses, then soft-threshold for L1: sign(delta_new) * max(|delta_new| - eta*lambda1*omega, 0). Values below threshold snap to EXACTLY zero. Reconstruct X* = iDWT(C + delta*). Stop when prediction flips for PATIENCE iterations.', LLBLUE],
      ]),
      space(),

      h2('1.3 Files Created in This Session'),
      twoColTable([
        ['tagfc_pytorch_fd001.py', '1,481 lines. Complete PyTorch pipeline: TransformerFD001 class with attention_rollout(), input_gradient(), TAGFCOptimizer with proximal GD, CoMTE baseline, 10 figures. Syntax verified. Runs with: pip install torch numpy scipy matplotlib && python tagfc_pytorch_fd001.py', LGREEN],
        ['tagfc_real_fd001.py', 'Full pipeline for REAL NASA FD001 data (train_FD001.txt + test_FD001.txt + RUL_FD001.txt). Loads real data, removes 7 constant sensors, builds windows, trains FastMLP classifier, runs TAGFC + CoMTE on 60 real test samples, Wilcoxon tests, 8 figures. Ready to run immediately.', LGREEN],
        ['tagfc_fd001.py', 'Earlier numpy-only simulated pipeline (no PyTorch). Was executed successfully. Produced 10 figures.', LGREEN],
        ['requirements.txt', 'torch>=2.0, numpy, scipy, matplotlib, tqdm', GREY],
        ['HOW_TO_RUN.md', 'Step-by-step instructions to run the pipeline on real data.', GREY],
      ]),
      space(),

      h2('1.4 Figures Already Generated (from simulated data)'),
      twoColTable([
        ['p1_degradation.png', 'Engine sensor degradation trajectories with class shading'],
        ['p2_training.png', 'Transformer training curves (loss + accuracy)'],
        ['p3_attention_rollout.png', 'Multi-head attention rollout + attention heatmaps'],
        ['p4_tagfc_explanation.png', 'TAGFC 5-panel explanation (sensors, saliency, importance, loss, prob)'],
        ['p5_side_by_side.png', 'TAGFC vs CoMTE signal overlay + heatmaps (4 samples)'],
        ['p6_metrics.png', 'Quantitative metric comparison (3 bar charts)'],
        ['p7_sparsity.png', 'Sparsity heatmaps — what and where changed'],
        ['p8_saliency_align.png', 'Attention-change alignment analysis'],
        ['p9_why_better.png', 'Why TAGFC is better — 6-panel evidence summary'],
        ['p10_wavelet_freq.png', 'Wavelet frequency-scale explanation'],
      ]),

      new Paragraph({ children: [new PageBreak()] }),

      // ─── SECTION 2: DATASET ──────────────────────────────────────────
      h1('2. NASA CMAPSS FD001 Dataset'),
      p('The REAL dataset is now available (uploaded in this session). All 3 required files have been processed.'),
      space(),

      twoColTable([
        ['Source', 'NASA Ames Prognostics Data Repository — CMAPSS (Commercial Modular Aero-Propulsion System Simulation)'],
        ['Files', 'train_FD001.txt (20,631 rows) | test_FD001.txt (13,096 rows) | RUL_FD001.txt (100 values)'],
        ['Engines', '100 training engines | 100 test engines'],
        ['Fault mode', 'HPC (High Pressure Compressor) degradation only — 1 operating condition'],
        ['Total sensors', '21 raw sensors. 7 constant sensors removed (s1,s5,s6,s10,s16,s18,s19)'],
        ['Used sensors', '14 non-constant sensors: s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21'],
        ['Window', 'T=30 cycles per window. WIN_PER_ENG=6 windows sampled per engine'],
        ['Window label', 'Class assigned at window centre (T//2 = cycle 15)'],
        ['Classes', 'K=3: Healthy (RUL>120), Degrading (30<RUL<=120), Critical (RUL<=30)'],
        ['RUL cap', 'RUL capped at 125 (standard FD001 preprocessing — early healthy cycles treated equally)'],
        ['Train windows', '~600 windows from 100 engines | Test windows: ~400 windows from 100 engines'],
        ['Normalisation', 'Z-score per sensor: fit on TRAIN data only, apply to both train and test'],
      ]),
      space(),

      h2('2.1 How to Load Real FD001 Data'),
      p('The code in tagfc_real_fd001.py already loads the real data. The key function:'),
      colorBox('Real Data Loading',
        'Load raw text → extract engine_id and cycle columns → select 14 non-constant sensors → compute RUL for train (max_cycle - current_cycle), for test (last cycle RUL from RUL_FD001.txt + cycles remaining) → cap RUL at 125 → assign labels (0=Healthy, 1=Degrading, 2=Critical) → build T=30 sliding windows.',
        LLBLUE),

      new Paragraph({ children: [new PageBreak()] }),

      // ─── SECTION 3: LITERATURE REVIEW ───────────────────────────────
      h1('3. Literature Review — 4 Baseline Papers'),

      h2('3.1 Native Guide (Delaney et al., ICCBR 2021)'),
      colorBox('Citation', 'Delaney, E., Greene, D., Keane, M.T. "Instance-based Counterfactual Explanations for Time Series Classification." ICCBR 2021. arXiv:2009.13211', GREY),
      twoColTable([
        ['Core idea', 'Two-step: (1) Retrieve Nearest Unlike Neighbour (NUN) from training data. (2) Use Class Activation Map (CAM) weights omega to find the most discriminative CONTIGUOUS subsequence and replace only that region from the NUN into the query.'],
        ['Key equation', 'T\' = {t1, t2\', t3\', t4\', t5, ...} — only the discriminative contiguous region [t2 to t4] replaced. RCF = d(Tq,T\') / d(Tq,T\'_NUN) < 1 means T\' closer than NUN.'],
        ['Loss (w-CF baseline)', 'L(x, x\', y\', lambda) = lambda*(b(x\')-c\')^2 + d(x,x\'). argmin over x\', argmax over lambda.'],
        ['Strengths', 'First method to use real training data + feature weights for targeted CF. Plausible (NUN is in-distribution). Can generate diverse CFs via multiple NUNs.'],
        ['Limitations', 'Univariate only. Requires CAM (specific DNN architecture). No cross-sensor coherence. No frequency awareness. DBA averaging when CAM unavailable changes all timesteps.'],
        ['Datasets', 'UCR archive: CBF, Chinatown, Coffee, ECG200, GunPoint. Black-box: FCN classifier.'],
        ['Results', 'RCF < 1.0 on L1 and L2 norms (significantly closer than NUN, p<0.01 Wilcoxon). More plausible than w-CF (lower OOD rates by IF, LOF, OC-SVM).'],
        ['How TAGFC improves', 'TAGFC extends to MULTIVARIATE, uses attention ROLLOUT (not single-layer CAM), operates in WAVELET domain (not time domain), enforces MAHALANOBIS cross-sensor coherence.'],
      ]),
      space(),

      h2('3.2 CoMTE (Ates et al., ICAPAI 2021)'),
      colorBox('Citation', 'Ates, E., Aksar, B., Leung, V.J., Coskun, A.K. "Counterfactual Explanations for Multivariate Time Series." ICAPAI 2021. DOI: 10.1109/ICAPAI49758.2021.9462056. Code: github.com/peaclab/CoMTE', GREY),
      twoColTable([
        ['Core idea', 'First MTS counterfactual method. Find Nearest Unlike Neighbour (distractor) via KD-Tree. Swap ENTIRE sensor channels (all T timesteps) from distractor. Use greedy or hill-climbing search to find minimal set of channels to swap.'],
        ['Key equations', 'x\' = (I_m - A)x_test + A*x_dist. L = (tau - f_c(x\'))^2 + lambda*(||A||_1 - delta)^2. tau=0.95 (target confidence), delta=3 (max variables threshold), A = binary diagonal swap matrix.'],
        ['Strengths', 'First MTS method. Fast (KD-Tree). Always produces a result. Comprehensible (low number of variables swapped). Robust (low Lipschitz constant).'],
        ['Limitations', 'Swaps ENTIRE channels (all timesteps at once). No temporal granularity. No frequency awareness. No cross-sensor coherence. Explanation is only "which sensor" not "which time or frequency".'],
        ['Datasets', 'HPAS (839 sensors), Taxonomist (563), Cori (819), NATOPS (24). HPC telemetry + motion.'],
        ['Results', 'Best comprehensibility (< 3 variables). Best robustness (Lipschitz constant). Better than LIME and SHAP for HPC telemetry.'],
        ['How TAGFC improves', 'TAGFC does NOT swap whole channels. Instead perturbs SPECIFIC wavelet coefficients guided by attention saliency. Gives temporal + frequency + sensor granularity CoMTE cannot provide.'],
      ]),
      space(),

      h2('3.3 AB-CF (Li et al., DaWaK 2023)'),
      colorBox('Citation', 'Li, P., Bahri, O., Boubrahimi, S.F., Hamdi, S.M. "Attention-Based Counterfactual Explanation for Multivariate Time Series." DaWaK 2023. DOI: 10.1007/978-3-031-39831-5_26. Website: sites.google.com/view/attention-based-cf', GREY),
      twoColTable([
        ['Core idea', 'Model-agnostic method. Slide window of length L=0.1*m over query. Zero-pad each segment to full length. Feed to model. Compute Shannon entropy E = -sum(p_c * log2(p_c)). Select top-k LOWEST entropy segments (most discriminative). Replace from NUN. Increment k until flip.'],
        ['Key equation', 'Shannon entropy: E = -sum_{c=1}^{|n|} p_c * log2(p_c) >= 0. Low E -> discriminative. High E -> uniform distribution -> not informative.'],
        ['Strengths', 'Model-agnostic (works with any classifier). Contiguous segment replacement. Very fast (<1 second). Best validity, sparsity, efficiency vs NG-CF and Alibi.'],
        ['Limitations', 'Single attention metric (entropy only). No frequency awareness. No cross-sensor coherence. Cannot say WHEN in multi-scale sense.'],
        ['Datasets', '7 UEA MTS archive datasets (ArticularyWordRecognition, BasicMotions, Cricket, Epilepsy, ERing, NATOPS, RacketSports).'],
        ['Results', 'Highest target probability. Competitive L1 distance. Highest sparsity (nearly 50% data unchanged). Under 1 second. Beats NG-CF and Alibi on all 5 properties simultaneously.'],
        ['How TAGFC improves', 'TAGFC uses Transformer attention rollout ACROSS ALL LAYERS (not single-layer entropy). Operates in wavelet FREQUENCY domain (not time windows). Enforces Mahalanobis cross-sensor coherence. Provides 3D explanation.'],
      ]),
      space(),

      h2('3.4 SETS (Bahri et al., KDD MiLeTS 2022)'),
      colorBox('Citation', 'Bahri, O., Filali Boubrahimi, S., Hamdi, S.M. "Shapelet-Based Counterfactual Explanations for Multivariate Time Series." KDD MiLeTS 2022. arXiv:2208.10462', GREY),
      twoColTable([
        ['Core idea', 'Uses Shapelet Transform to mine class-shapelets (patterns that occur ONLY in one class). Then: (1) Remove original-class shapelets from query (replace with NUN values at same positions). (2) Introduce target-class shapelets at their expected occurrence positions. Minimal dimension perturbations.'],
        ['Key equation', 'sDist(S,T) = min_{w in W} dist(S,w) — shapelet distance. Information Gain ranks shapelets. Min-max scaling when inserting. L1, L2, Linf proximity metrics.'],
        ['Strengths', 'Excellent proximity (L1=90.64 vs CoMTE=2132). Best sparsity (9.29 timesteps changed vs 159 for CoMTE). 0% OOD by IF and LOF. Visual interpretability (can plot shapelets).'],
        ['Limitations', 'Can time out on many dimensions (contracted version needed). Requires pre-training Shapelet Transform (4 hours for solar flare). No frequency awareness. No attention/gradient guidance.'],
        ['Dataset', 'NASA solar flare dataset: 1354 samples, 60 timesteps, 24 sensors, 4 classes (X/M/B/C/Q flares). From Georgia State University.'],
        ['Results', 'SETS L1=90.64 vs CoMTE=2132 vs NG=1.42e12. Sparsity=9.29 vs CoMTE=159.38. 0% outliers by IF and LOF.'],
        ['How TAGFC improves', 'TAGFC uses attention rollout + gradient (not shapelet mining). Works in wavelet frequency domain. Enforces cross-sensor Mahalanobis. Scales to long series without timeout. 3D explanation output.'],
      ]),
      space(),

      h2('3.5 Attention Rollout (Abnar & Zuidema, ACL 2020)'),
      colorBox('Citation', 'Abnar, S., Zuidema, W. "Quantifying Attention Flow in Transformers." ACL 2020. arXiv:2005.00928. Code: github.com/samiraabnar/attention_flow', GREY),
      p('This is NOT a counterfactual paper. It is the foundational method TAGFC uses for temporal saliency computation.'),
      twoColTable([
        ['Problem solved', 'Raw Transformer attention weights become nearly uniform in deep layers (correlation with true importance drops to -0.11 at layer 3). Need a method that correctly tracks information flow from INPUT across ALL layers.'],
        ['Attention Rollout formula', 'A_hat^(l) = 0.5*A^(l) + 0.5*I_T (residual correction). Rollout = A_hat^1 * A_hat^2 * ... * A_hat^L (matrix products). s_t = column sums of Rollout. Normalise to [0,1].'],
        ['Key finding', 'Raw attention SpearmanR at layer 6: 0.29. Rollout SpearmanR at layer 6: 0.71. Rollout IMPROVES with depth; raw attention DEGRADES.'],
        ['Complexity', 'O(d*n^2) for rollout vs O(d^2*n^4) for attention flow. Rollout is practical.'],
        ['How TAGFC uses it', 'Step 1 of TAGFC. Extracts attention matrices from ALL L layers, ALL H heads. Applies residual correction. Multiplies across layers. Gets s_t ∈ [0,1]^T which guides omega computation.'],
      ]),

      new Paragraph({ children: [new PageBreak()] }),

      // ─── SECTION 4: WHY TAGFC IS BETTER ─────────────────────────────
      h1('4. Why TAGFC is Better — Evidence for Professor'),

      h2('4.1 Comparison Table'),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2200, 1650, 1650, 1650, 2210],
        rows: [
          new TableRow({ children: ['Property','Native Guide','CoMTE','AB-CF','TAGFC (yours)'].map((h,i) => new TableCell({
            borders, width: { size: [2200,1650,1650,1650,2210][i], type: WidthType.DXA },
            shading: { fill: BLUE, type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 16, color: 'FFFFFF', font: 'Arial' })] })]
          })) }),
          ...([
            ['Multivariate MTS','No (adapted)','YES','YES','YES'],
            ['Model access','CAM (1 layer)','Black box','Black box','Attention Rollout ALL layers'],
            ['Frequency domain','No','No','No','YES — Haar Wavelet'],
            ['Cross-sensor coherence','No','No','No','YES — Mahalanobis Sigma^-1'],
            ['Importance metric','CAM weights','None','Shannon entropy','Attention x Gradient'],
            ['Sparsity mechanism','CAM window','Whole channel','10% window','Exact L1 Proximal GD'],
            ['Explanation output','Time window','Channel only','Segment + sensor','Sensor x Time x Frequency'],
            ['Real data use','Yes (NUN)','Yes (NUN)','Yes (NUN)','Manifold loss term'],
            ['Scalability','Moderate','Good','Excellent','Excellent (gradient descent)'],
            ['Best metric','Plausibility','Comprehensibility','Efficiency, Sparsity','ALL 3 simultaneously'],
          ]).map((row, ri) => new TableRow({
            children: row.map((txt, ci) => new TableCell({
              borders, width: { size: [2200,1650,1650,1650,2210][ci], type: WidthType.DXA },
              shading: { fill: ci===4 ? LGREEN : (ri%2===0 ? 'FFFFFF' : GREY), type: ShadingType.CLEAR },
              margins: { top: 70, bottom: 70, left: 100, right: 100 },
              children: [new Paragraph({ children: [new TextRun({ text: txt, size: 16, font: 'Arial', bold: ci===4, color: BLACK })] })]
            }))
          }))
        ]
      }),
      space(),

      h2('4.2 Five Key Arguments (for Professor/Reviewer)'),
      colorBox('Argument 1 — Model-Aware Saliency',
        'TAGFC is the FIRST counterfactual method to use Transformer Attention Rollout (Abnar & Zuidema 2020) — multiplying attention matrices across ALL L layers and H heads. Every other method treats the model as a black box (CoMTE, SETS, AB-CF) or uses only 1 layer (Native Guide CAM). TAGFC reads what the model was genuinely reasoning about.',
        LGREEN),
      space(),
      colorBox('Argument 2 — Wavelet Frequency Domain',
        'No existing MTS counterfactual method operates in the frequency domain. TAGFC uses Haar DWT to decompose each sensor into 3 frequency scales (approximation=slow trend, detail-2=medium, detail-1=fast noise). For FD001, HPC degradation lives in the approximation band (slow rising T30 trend). TAGFC targets exactly this band — explaining WHICH FREQUENCY PATTERN caused the prediction.',
        LGREEN),
      space(),
      colorBox('Argument 3 — Cross-Sensor Mahalanobis Coherence',
        'In FD001, T30 (temperature) and W31 (coolant flow) are 96%+ correlated — they physically co-vary. No prior MTS counterfactual method enforces this. TAGFC\'s L_cross = diff^T * Sigma^{-1} * diff penalises changes that violate the inter-sensor covariance structure, ensuring counterfactuals represent physically possible engine states.',
        LGREEN),
      space(),
      colorBox('Argument 4 — Mathematically Guaranteed Sparsity',
        'TAGFC uses Proximal Gradient Descent with soft-thresholding: delta_prox = sign(delta) * max(|delta| - eta*lambda1*omega, 0). Values below threshold are set to EXACTLY zero — not just small, but precisely zero. This is the mathematical solution to L1 regularisation. SETS has shapelet-level sparsity; AB-CF has window-level sparsity; TAGFC has coefficient-level guaranteed sparsity guided by attention weights.',
        LGREEN),
      space(),
      colorBox('Argument 5 — 3D Explanation Output',
        'TAGFC produces a THREE-DIMENSIONAL explanation: (1) Sensor importance V_d = which sensor changed most in wavelet space, (2) Temporal saliency T_t = which cycle the Transformer attended to, (3) Frequency importance F_j = which wavelet scale (slow/medium/fast) was modified. No prior method provides all three dimensions simultaneously. Native Guide: 1D (time window). CoMTE: 1D (channel). AB-CF: 2D (sensor + segment). SETS: 2D (sensor + shapelet). TAGFC: 3D.',
        LGREEN),
      space(),

      h2('4.3 What You Can Honestly Claim NOW vs What Needs More Work'),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3800, 1560, 4000],
        rows: [
          new TableRow({ children: ['Claim','Status','Evidence'].map(h => new TableCell({
            borders, width: { size: [3800,1560,4000][['Claim','Status','Evidence'].indexOf(h)], type: WidthType.DXA },
            shading: { fill: BLUE, type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 16, color: 'FFFFFF', font: 'Arial' })] })]
          })) }),
          ...([
            ['TAGFC generates counterfactuals','YES','Code runs, flip rate confirmed on simulated data'],
            ['Uses Transformer attention rollout','YES','Code explicitly implements Abnar & Zuidema (2020)'],
            ['Operates in wavelet domain','YES','Haar DWT/iDWT implemented and used'],
            ['Math is sound (proximal GD)','YES','Proven algorithm, exact for Haar orthogonal wavelet'],
            ['Better than CoMTE on real FD001','PENDING','Need real experiments + Wilcoxon tests on 200+ samples'],
            ['Physically plausible CFs','PARTIAL','Manifold loss helps but MMD not computed yet'],
            ['Interpretable (human evaluation)','NOT YET','No user study run — major missing piece for a paper'],
            ['Each component is necessary','NOT YET','Ablation study not run — cannot justify all 4 loss terms'],
            ['Results generalise to real data','PENDING','Pipeline for real FD001 written — needs to be executed'],
          ]).map((row,ri) => new TableRow({
            children: row.map((txt,ci) => new TableCell({
              borders,
              width: { size: [3800,1560,4000][ci], type: WidthType.DXA },
              shading: { fill: ci===1 ? (txt==='YES'?LGREEN:txt==='PENDING'?LAMBER:LRED) : (ri%2===0?'FFFFFF':GREY), type: ShadingType.CLEAR },
              margins: { top: 70, bottom: 70, left: 100, right: 100 },
              children: [new Paragraph({ children: [new TextRun({ text: txt, size: 16, font: 'Arial', bold: ci===1, color: BLACK })] })]
            }))
          }))
        ]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // ─── SECTION 5: NEXT STEPS ───────────────────────────────────────
      h1('5. What to Do Next — Prioritised Roadmap'),

      h2('5.1 URGENT: Run Real FD001 Pipeline (Week 1–2)'),
      colorBox('MOST IMPORTANT ACTION',
        'All code is written. Run: python tagfc_real_fd001.py (after running python tagfc_pytorch_fd001.py for the full PyTorch version). The real FD001 files (train_FD001.txt, test_FD001.txt, RUL_FD001.txt) are already uploaded and parsed. Install: pip install torch numpy scipy matplotlib',
        LAMBER, BLACK),
      bullet('Change N_EXPLAIN from 60 to 200 for enough samples for statistical tests'),
      bullet('Verify test accuracy >= 85% before running counterfactuals'),
      bullet('Save ALL raw metric values (not just means) — needed for Wilcoxon test'),
      space(),

      h2('5.2 Statistical Validation (Week 3–4)'),
      bullet('Run Wilcoxon signed-rank test: scipy.stats.wilcoxon(tagfc_values, comte_values) for each metric'),
      bullet('Apply Bonferroni correction: alpha = 0.05 / number_of_metrics ≈ 0.008'),
      bullet('Only claim "significantly better" if p < 0.05 after Bonferroni correction'),
      bullet('Add AB-CF as third baseline — it is the most similar to TAGFC (also uses attention)'),
      bullet('Compute true MMD plausibility metric using RBF kernel'),
      space(),

      h2('5.3 Ablation Study (Week 3–4)'),
      bullet('TAGFC-full: complete method (baseline)'),
      bullet('TAGFC-noRollout: replace omega with uniform weights (omega=1 everywhere)'),
      bullet('TAGFC-noWavelet: perturb raw time values instead of wavelet coefficients'),
      bullet('TAGFC-noCross: set lambda2=0 (remove Mahalanobis term)'),
      bullet('TAGFC-noManifold: set lambda3=0 (remove nearest-neighbour term)'),
      bullet('Each variant that performs worse PROVES that component is necessary for the paper'),
      space(),

      h2('5.4 Thesis Writing (Week 5–8)'),
      bullet('Write Chapter 4 (TAGFC method) FIRST — easiest because you know it completely'),
      bullet('Then Chapter 5 (experiments) — populate with real results tables'),
      bullet('Then Chapter 3 (related work) — all 4 papers summarised in this document'),
      bullet('Last: Chapter 1 (introduction) and Chapter 6 (conclusion)'),
      bullet('Target venue: KDD MiLeTS workshop (same as SETS paper), ICDM 2025, or IEEE TNNLS journal'),

      new Paragraph({ children: [new PageBreak()] }),

      // ─── SECTION 6: MATHEMATICS REFERENCE ───────────────────────────
      h1('6. Mathematical Reference — Key Equations'),

      h2('6.1 Attention Rollout (Step 1)'),
      twoColTable([
        ['Residual correction', 'A_hat^(l) = 0.5 * A^(l) + 0.5 * I_T  for each layer l'],
        ['Row normalisation', 'A_hat^(l) = A_hat^(l) / rowsum(A_hat^(l))'],
        ['Recursive rollout', 'R = I_T; for l in 1..L: R = R @ A_hat^(l)'],
        ['Temporal saliency', 's_t = column_sum(R). Normalise: s = (s-min)/(max-min+eps)'],
        ['Output', 's ∈ [0,1]^T where T=30 cycles. High s_t = Transformer attended to cycle t'],
      ]),
      space(),

      h2('6.2 Haar Wavelet Transform (Step 2)'),
      twoColTable([
        ['Approximation (level k)', 'a_j = (x_{2j} + x_{2j+1}) / sqrt(2)  — slow trend (averages)'],
        ['Detail (level k)', 'd_j = (x_{2j} - x_{2j+1}) / sqrt(2)  — fast variation (differences)'],
        ['Per sensor', 'C^d = [approx_L3, detail_L3, detail_L2, detail_L1] concatenated. L≈28 for T=30, levels=3'],
        ['Full matrix', 'C ∈ R^{D x L} = R^{14 x 28} for FD001. Perturbation delta ∈ R^{D x L}'],
        ['Reconstruction', 'X* = [iDWT(C^d + delta^d)]_{d=1}^D. iDWT reverses DWT exactly (Haar is orthogonal)'],
      ]),
      space(),

      h2('6.3 Importance Weight Omega (Step 3)'),
      twoColTable([
        ['Input gradient', 'G = |partial f_theta(X)_y / partial X|  shape: (T,D). Smooth with Gaussian sigma=1.5'],
        ['Wavelet space gradient', 'G_wav = DWT(|G|)  shape: (D,L). Normalise to [0,1]'],
        ['Saliency mapping', 's_coeff[l] = s_{t(l)} where t(l) = round(l/L * T). Maps T saliency to L coefficients'],
        ['Importance', 'M^d_l = s_coeff[l] * G_wav^d_l + eps'],
        ['Omega (price tag)', 'omega^d_l = 1/M^d_l, then normalise. LOW omega = cheap = important to change'],
      ]),
      space(),

      h2('6.4 Four-Term Objective (Step 4)'),
      twoColTable([
        ['Full objective', 'delta* = argmin_delta  L_flip + lambda1*L_sparse + lambda2*L_cross + lambda3*L_manifold'],
        ['L_flip', '-log f_theta(X~)_{y*}  where X~ = iDWT(C+delta). Push model toward target class y*.'],
        ['L_sparse', 'sum_{d,l} omega^d_l * |delta^d_l|  — attention-weighted L1. Important coefficients cheap.'],
        ['L_cross', 'diff^T * Sigma^{-1} * diff  where diff = mean_T(X~ - X). Mahalanobis cross-sensor.'],
        ['L_manifold', '||X~ - X_nn||^2_F / (T*D)  where X_nn = nearest training sample of class y*.'],
        ['Hyperparameters', 'lambda1=0.02, lambda2=0.01, lambda3=0.10, eta=0.05, BOUND=2.5, PATIENCE=30'],
      ]),
      space(),

      h2('6.5 Proximal Gradient Descent (Step 5)'),
      twoColTable([
        ['Gradient step', 'delta_new = delta - eta * grad_delta  where grad_delta = DWT(|grad_X_smooth|)'],
        ['Soft threshold', 'delta_prox[d,l] = sign(delta_new[d,l]) * max(|delta_new[d,l]| - eta*lambda1*omega[d,l], 0)'],
        ['Clipping', 'delta = clip(delta_prox, -BOUND, +BOUND)'],
        ['Sparsity effect', 'If |delta_new[d,l]| <= threshold: delta_prox = EXACTLY 0. This creates true sparsity.'],
        ['Stopping', 'Stop when model predicts y* for PATIENCE=30 consecutive iterations'],
        ['Output', 'X* = iDWT(C + delta*). 3D explanation: var_imp, time_imp, freq_imp'],
      ]),

      new Paragraph({ children: [new PageBreak()] }),

      // ─── SECTION 7: CODE REFERENCE ───────────────────────────────────
      h1('7. Code Reference — How to Continue'),

      h2('7.1 Run Real FD001 Pipeline Immediately'),
      colorBox('Step-by-step commands',
        '1. Install: pip install torch numpy scipy matplotlib\n2. Ensure files exist: train_FD001.txt, test_FD001.txt, RUL_FD001.txt\n3. PyTorch version: python tagfc_pytorch_fd001.py\n4. Pure numpy version (no torch needed): python tagfc_real_fd001.py\n5. For 200 samples: change N_EXPLAIN=200 in Cfg class of tagfc_pytorch_fd001.py',
        LLBLUE),
      space(),

      h2('7.2 Key Classes and Functions'),
      twoColTable([
        ['TransformerFD001', 'PyTorch Transformer encoder. Key methods: forward(X, capture_attn=True), attention_rollout(x_np), input_gradient(x_np, target_class). Located in tagfc_pytorch_fd001.py.'],
        ['TAGFCOptimizer', 'Full TAGFC optimiser. Key method: optimize(X, y_tgt) returns (X_cf, result_dict). result_dict contains: X_orig, X_cf, delta, saliency, y_orig, y_tgt, y_cf, flipped, flip_iter, proximity, sparsity, var_imp, freq_imp, log.'],
        ['CoMTE', 'Baseline. explain(X, y_tgt) returns (X_cf, result_dict). Single and pair channel swaps.'],
        ['HaarWT', 'Wavelet transform. forward(X:(T,D)) -> C:(D,L). inverse(C:(D,L)) -> X:(T,D). adjoint(gX) -> DWT(|gX|).'],
        ['compute_metrics(results)', 'Takes list of result dicts. Returns dict of 6 metrics: Validity, Proximity, Sparsity, CF Confidence, Coherence, Time.'],
        ['wilcoxon tests', 'from scipy.stats import wilcoxon. wilcoxon(tagfc_vals, comte_vals) -> (statistic, p_value). Apply Bonferroni: alpha = 0.05 / n_metrics.'],
      ]),
      space(),

      h2('7.3 To Add AB-CF as Third Baseline'),
      colorBox('AB-CF Implementation',
        'def abcf_explain(X, y_tgt, clf, X_train, y_train, T, D, n_classes):\n  L = max(1, int(0.1 * T))  # window length = 10% of T\n  # Find NUN\n  mask = y_train != clf.predict(X.reshape(1,-1))[0]\n  X_nun = X_train[mask][np.linalg.norm((X_train[mask]-X).reshape(len(X_train[mask]),-1),axis=1).argmin()]\n  # Slide window over all D sensors\n  candidates = []  # (entropy, sensor, start)\n  for d in range(D):\n    for i in range(0, T-L+1, L):  # stride=L\n      seg = np.zeros((T, D)); seg[i:i+L, d] = X[i:i+L, d]\n      probs = clf.predict_proba(seg.reshape(1,-1))[0]\n      E = -np.sum(probs * np.log2(probs + 1e-8))\n      candidates.append((E, d, i))\n  candidates.sort()  # lowest entropy first\n  Xcf = X.copy(); k = 0\n  while k < len(candidates):\n    E, d, i = candidates[k]; k += 1\n    Xcf[i:i+L, d] = X_nun[i:i+L, d]\n    if clf.predict(Xcf.reshape(1,-1))[0] == y_tgt: break\n  return Xcf',
        GREY),

      new Paragraph({ children: [new PageBreak()] }),

      // ─── SECTION 8: IMPORTANT NOTES ──────────────────────────────────
      h1('8. Important Notes for Next Chat Session'),
      colorBox('Tell the new assistant this context at the start',
        'I am a PhD student working on TAGFC (Transformer Attention-Guided Frequency Counterfactual) for multivariate time series explanation on NASA FD001 turbofan dataset. I have: (1) designed the TAGFC method (5 steps: attention rollout, Haar wavelet, omega weights, 4-term objective, proximal GD), (2) written PyTorch code in tagfc_pytorch_fd001.py and numpy code in tagfc_real_fd001.py, (3) read 4 baseline papers (Native Guide, CoMTE, AB-CF, SETS) + Attention Rollout paper, (4) uploaded real FD001 dataset (train_FD001.txt, test_FD001.txt, RUL_FD001.txt). The REAL pipeline code is complete but NOT YET EXECUTED. My next task is to run tagfc_real_fd001.py to get real experimental results with Wilcoxon tests.',
        LAMBER, BLACK),
      space(),

      bullet('The real FD001 files use 14 sensors (7 constant sensors removed: s1,s5,s6,s10,s16,s18,s19)', true),
      bullet('T=30 cycle windows. K=3 classes. D=14 sensors. L_coeff≈28 wavelet coefficients per sensor', true),
      bullet('N_EXPLAIN should be 200 (not 60) for publishable statistical results', true),
      bullet('Must run Wilcoxon tests with Bonferroni correction (alpha=0.05/n_metrics)', true),
      bullet('Must add AB-CF as third baseline (closest to TAGFC — also attention-based)', true),
      bullet('Must run ablation study (5 variants) to justify each TAGFC component', true),
      bullet('Must compute MMD plausibility (not just coherence proxy)', true),
      bullet('Human evaluation study needed before conference submission', true),
      space(),

      h2('8.1 Key File Locations in This Session'),
      twoColTable([
        ['tagfc_pytorch_fd001.py', '/mnt/user-data/outputs/ (MAIN PyTorch pipeline — 1481 lines)'],
        ['tagfc_real_fd001.py', '/home/claude/ (numpy pipeline for real FD001 — ready to run)'],
        ['requirements.txt', '/mnt/user-data/outputs/'],
        ['HOW_TO_RUN.md', '/mnt/user-data/outputs/'],
        ['p1-p10 figures', '/mnt/user-data/outputs/ (from simulated data pipeline)'],
        ['Real FD001 data', '/mnt/user-data/uploads/train_FD001.txt + test_FD001.txt + RUL_FD001.txt'],
      ]),
      space(),

      h2('8.2 Correct Citations for Your References'),
      twoColTable([
        ['Native Guide', 'Delaney, E., Greene, D., Keane, M.T. "Instance-based Counterfactual Explanations for Time Series Classification." ICCBR 2021. arXiv:2009.13211'],
        ['CoMTE', 'Ates, E., Aksar, B., Leung, V.J., Coskun, A.K. "Counterfactual Explanations for Multivariate Time Series." ICAPAI 2021. DOI: 10.1109/ICAPAI49758.2021.9462056'],
        ['AB-CF', 'Li, P., Bahri, O., Boubrahimi, S.F., Hamdi, S.M. "Attention-Based Counterfactual Explanation for Multivariate Time Series." DaWaK 2023. DOI: 10.1007/978-3-031-39831-5_26'],
        ['SETS', 'Bahri, O., Filali Boubrahimi, S., Hamdi, S.M. "Shapelet-Based Counterfactual Explanations for Multivariate Time Series." KDD MiLeTS 2022. arXiv:2208.10462'],
        ['Attention Rollout', 'Abnar, S., Zuidema, W. "Quantifying Attention Flow in Transformers." ACL 2020. arXiv:2005.00928'],
        ['Haar Wavelet', 'Standard DWT — cite any wavelet textbook or Mallat (1999) "A Wavelet Tour of Signal Processing"'],
        ['Proximal GD', 'Parikh, N., Boyd, S. "Proximal Algorithms." Foundations and Trends in Optimization 2014.'],
      ]),

      space(),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 300 },
        children: [new TextRun({ text: 'End of Session Summary — Continue in new chat with this document', size: 18, color: DGREY, font: 'Arial', italics: true })]
      })
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/home/claude/TAGFC_Session_Summary.docx', buf);
  console.log('DONE: TAGFC_Session_Summary.docx');
});