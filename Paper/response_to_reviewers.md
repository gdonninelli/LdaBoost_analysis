# Response to Committee Comments

This document maps each committee comment to concrete manuscript revisions. It is written as a paper-facing summary and avoids implementation-level references.

## Major Concerns

### 1) Validity of recalculating LDA with residual-based updates
**Comment summary:** clarify why Fisher optimization remains valid when Step 1 is updated at each boosting round.

**Implemented changes:**
- Added explicit stagewise formulation:
  - gradients: $g^{(m)} = Y - P^{(m-1)}$
  - induced labels: $z_i^{(m)} = \arg\max_c g_{i,c}^{(m)}$
  - Fisher objective at stage $m$: maximize $|G^\top S_B(z^{(m)})G| / |G^\top S_W(z^{(m)})G|$
- Clarified that each stage solves a standard LDA problem over a well-defined categorical partition.

**Location in manuscript:**
- Methods, integrated LdaBoost subsection.

---

### 2) Scalability and computational burden of per-round LDA
**Comment summary:** discuss cost of iterative LDA versus one-shot LDA preprocessing.

**Implemented changes:**
- Added a dedicated "Computational scalability" subsection with symbolic cost comparison:
  - one-shot LDA+GBM: $\mathcal{C}_{\mathrm{LDA}} + M\mathcal{C}_{\mathrm{tree}}$
  - LdaBoost: $M\mathcal{C}_{\mathrm{LDA}} + M\mathcal{C}_{\mathrm{tree}}$
- Added runtime-ablation evidence in Results with fixed-seed summaries (mean and SD).
- Added a direct runtime comparison in the HAR discussion and in Table 1 (test-validation time block):
  - HAR: GBM $675.07$s, PCA+GBM $652.62$s, LDA+GBM $10.25$s, LdaBoost $286.22$s.
  - This corresponds to about $65.9\times$ speedup of LDA+GBM vs GBM and $63.7\times$ vs PCA+GBM.

**Location in manuscript:**
- Computational scalability subsection.
- Results (HAR paragraph and Table 1).

---

### 3) Step 1 uncertainty (residuals are not labels)
**Comment summary:** Step 1 needed stronger formal clarification.

**Implemented changes:**
- Reframed Step 1 using induced labels from gradients instead of raw residuals as labels.
- Updated explanatory bullets and pseudocode to match this formal definition.

**Location in manuscript:**
- Methods (integrated LdaBoost explanation and algorithm box).

---

### 4) Step 4 dependence on Step 1 reliability
**Comment summary:** potential instability in Step 1 may propagate to Step 4.

**Implemented changes:**
- Added deterministic tie handling and fallback description in text (reuse previous valid projection when induced labels collapse).
- Aligned the pseudocode description with this robustness rule.

**Location in manuscript:**
- Methods (integrated LdaBoost explanation and algorithm box).

## Minor Concerns

### A) PCA notation should be independent of labels
**Implemented:** revised Methods opening to separate unsupervised PCA notation from supervised label notation.

### B) Oversimplified LDA weight interpretation
**Implemented:** replaced with Fisher-consistent interpretation involving between-class separation and within-class variability/covariance.

### C) Typographical and wording issues
**Implemented:** corrected typographical inaccuracies and improved local wording clarity.

### D) Simulation phrasing for $N=1{,}000$
**Implemented:** revised language to neutral finite-sample wording.

## Additional Committee Points (Current Status)

### 1) Asymmetry in tuning strategy across methods
**Status:** addressed.

**Implemented:** clarified that fixed-parameter comparisons are transformation-focused contrasts and not fully retuned head-to-head rankings.

---

### 2) PCA components tuned while LDA fixed at $C-1$
**Status:** addressed.

**Implemented:** explained this as a Fisher-LDA design choice and acknowledged it as an asymmetry, with reduced-LDA-dimensionality analysis identified as future work.

---

### 3) Missing formal statistical significance tests
**Status:** partially addressed.

**Implemented:**
- Added nonparametric significance diagnostics for simulation comparisons (Friedman + pairwise Wilcoxon with Holm correction).
- Reported that no pairwise contrast remained significant after Holm correction at $\alpha=0.05$.
- Added clear real-data fold/sample context and explicitly stated that full paired inferential testing on all real datasets remains future work.

---

### 4) Computational overhead and scalability
**Status:** addressed.

**Implemented:** symbolic complexity section plus empirical runtime evidence.

---

### 5) Binary case ($C=2$) and one-component LDA constraint
**Status:** addressed.

**Implemented:** clarified that binary LDA yields one component per stage and that competitiveness depends on iterative direction updates across rounds.

---

### 6) Positioning versus boosting-variant literature
**Status:** addressed.

**Implemented:** added positioning paragraph contrasting LdaBoost with margin/filter/projection families and emphasizing stagewise supervised projection updates.

---

### 7) Gaussian simulation design may favor LDA
**Status:** addressed.

**Implemented:** added explicit limitation note and identified non-Gaussian extension as future work.

---

### 8) HAR case where baseline GBM is strongest in CV
**Status:** addressed.

**Implemented:** added focused interpretation that reducing 561 predictors to $C-1=5$ discriminant components may discard nonlinear interaction information used by trees in the original space.

## Latest Results Update Included in This Revision

The manuscript now reflects the newest empirical tables and comments:
- Table 1 updated with revised cross-validated and validation accuracies.
- Table 1 expanded with a "Test time (s)" block (HAR and RAINFALL).
- HAR results paragraph updated with explicit large runtime gains for one-shot LDA+GBM versus GBM and PCA+GBM, while noting the small CV accuracy trade-off.
- A new figure has been inserted immediately below Table 1 to visually summarize that LDA-based methods are frequently among the strongest models across datasets.

## Validation Performed

- The manuscript compiles successfully to PDF after these revisions.
- The updated table, timing block, and figure placement have been verified in the compiled document.
