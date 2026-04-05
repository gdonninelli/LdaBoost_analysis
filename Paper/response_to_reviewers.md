# Response to Committee Comments

This document maps each comment in ReviewComments.md to concrete revisions made in the manuscript and codebase.

## Major Concerns

### 1) Validity of recalculating LDA with residual-based updates
**Comment summary:** clarify why Fisher optimization is still valid when Step 1 is updated each boosting round.

**Implemented changes:**
- Added explicit stagewise construction:
  - gradients: $g^{(m)} = Y - P^{(m-1)}$
  - induced labels: $z_i^{(m)} = \arg\max_c g_{i,c}^{(m)}$
  - Fisher objective over $z^{(m)}$: maximize $|G^\top S_B(z^{(m)})G| / |G^\top S_W(z^{(m)})G|$
- Clarified that each round solves a standard LDA objective on a well-defined categorical partition.

**Locations:**
- Paper/main.tex (Methods, integrated LdaBoost subsection)

---

### 2) Scalability and computational burden of per-round LDA
**Comment summary:** discuss cost of iterative LDA relative to one-shot LDA preprocessing.

**Implemented changes:**
- Added a dedicated "Computational scalability" subsection with symbolic cost comparison:
  - one-shot LDA+GBM: $\mathcal{C}_{\mathrm{LDA}} + M\mathcal{C}_{\mathrm{tree}}$
  - LdaBoost: $M\mathcal{C}_{\mathrm{LDA}} + M\mathcal{C}_{\mathrm{tree}}$
- Added empirical runtime-ablation results section and table (5 fixed seeds).
- Added reproducible benchmark script and output artifacts.

**Ablation protocol (why, how, data):**
- Why this ablation: to directly address the committee concern on computational burden by pairing the symbolic complexity argument with measured wall-clock evidence.
- How it was conducted: two synthetic scenarios were generated and, for each scenario, the same train/test protocol (stratified split, test size 0.25) was repeated over fixed seeds (11, 29, 47, 71, 89).
- Models compared in each run: GBM, PCA+GBM, LDA+GBM (one-shot transform), and LdaBoost (dynamic per-round LDA).
- Metrics recorded: fit time, predict time, total time, and test accuracy; results were aggregated as mean and standard deviation.
- Data used: balanced correlated Gaussian synthetic datasets with $\rho=0.5$ in two settings:
  - binary high-dimensional: $N=1200$, $p=300$, $C=2$
  - multiclass moderate-dimensional: $N=1200$, $p=100$, $C=5$
- Scope note: this ablation is a runtime-context experiment with fixed hyperparameters; accuracy values are reported as comparative context, not as tuned performance ceilings.

**Locations:**
- Paper/main.tex (Computational scalability subsection)
- Paper/main.tex (Runtime-ablation evidence subsection and Table runtime_ablation)
- simulations/runtime_ablation.py
- simulations/output_pipeline_confront/runtime_ablation_raw.csv
- simulations/output_pipeline_confront/runtime_ablation_summary.csv
- simulations/output_pipeline_confront/runtime_ablation_summary.json

---

### 3) Step 1 uncertainty (residuals are not labels)
**Comment summary:** Step 1 needs stronger explanation and formal validity.

**Implemented changes:**
- Reframed Step 1 to use induced labels from gradients rather than raw residuals as labels.
- Added explicit text that Step 1 is an LDA fit on induced class assignments at each stage.
- Updated algorithmic bullets and pseudocode to match the formal definition.

**Locations:**
- Paper/main.tex (Step 1 bullet and pseudocode)

---

### 4) Step 4 depends on Step 1 reliability
**Comment summary:** if Step 1 is unstable, Step 4 becomes unreliable.

**Implemented changes:**
- Added tie-handling and fallback policy in manuscript (reuse previous projection if induced labels collapse).
- Updated implementation in both core and simulation algorithms to enforce this behavior.

**Locations:**
- Paper/main.tex (Step 1 explanation and pseudocode)
- LdaBoost/algorithm.py (_fit_lda_with_fallback and fit updates)
- simulations/LdaBoosting/algorithm.py (_fit_lda_with_fallback and fit updates)

## Minor Concerns

### A) PCA notation should not rely on labels
**Implemented:** replaced opening Methods sentence with a PCA-unsupervised notation clarification and separate supervised label notation.

**Location:**
- Paper/main.tex (Methods opening paragraph)

---

### B) Oversimplified LDA weight explanation
**Implemented:** replaced the sentence with a Fisher-consistent interpretation including both between-class separation and within-class variability/covariance.

**Location:**
- Paper/main.tex (LDA paragraph in Methods)

---

### C) Typographical issues (e.g., "repored")
**Implemented:** corrected typo and related language fixes.

**Locations:**
- Paper/main.tex (HAR dataset paragraph and nearby tuning text)

---

### D) Simulation sentence and wording for N=1,000
**Implemented:** revised opening simulation sentence and replaced small-sample wording with neutral finite-sample / N=1,000 wording.

**Locations:**
- Paper/main.tex (Comparative performance subsection)

## Additional Committee Points (Current Status)

### 1) Asymmetry in tuning strategy across methods
**Status:** addressed in manuscript discussion.

**Implemented:** added explicit clarification that fixed-parameter comparisons were designed as transformation-focused contrasts, not fully optimized head-to-head rankings, and that this introduces asymmetry because GBM-tuned parameters are reused in transformed spaces.

**Locations:**
- Paper/main.tex (Tuning strategy subsection)

---

### 2) PCA components tuned while LDA fixed at $C-1$
**Status:** addressed as explicit methodological choice plus limitation.

**Implemented:** clarified that retaining $C-1$ follows Fisher-LDA construction, but acknowledged this as a comparison asymmetry and identified reduced-LDA-component ablation as relevant future work.

**Locations:**
- Paper/main.tex (Tuning strategy subsection)

---

### 3) Missing formal statistical significance tests
**Status:** addressed as explicit limitation.

**Implemented:** added statement that reported cross-validated means/SD are descriptive and that fold-wise paired significance tests were not conducted, so inferential claims should be interpreted cautiously.

**Locations:**
- Paper/main.tex (Tuning strategy subsection)

---

### 4) Computational overhead and scalability
**Status:** already addressed.

**Implemented:** symbolic complexity comparison + runtime-ablation table and reproducible outputs.

**Locations:**
- Paper/main.tex (Computational scalability and runtime-ablation evidence subsections)
- simulations/runtime_ablation.py
- simulations/output_pipeline_confront/runtime_ablation_summary.csv

---

### 5) Binary case ($C=2$) and one-component LDA constraint
**Status:** addressed with dedicated interpretation.

**Implemented:** added discussion that binary LDA yields one component per round and clarified that LdaBoost competitiveness comes from iterative direction updates across rounds; also noted stronger conceptual motivation in multiclass settings.

**Locations:**
- Paper/main.tex (Comparative performance of different pipelines: simulations subsection)

---

### 6) Positioning versus boosting-variant literature
**Status:** addressed with positioning paragraph.

**Implemented:** added a conceptual comparison to boosting variants that modify margins/filtering/projection geometry (including LPBoost and FilterBoost style families), highlighting that LdaBoost differs by recomputing supervised projections inside each boosting stage.

**Locations:**
- Paper/main.tex (Introduction)

---

### 7) Gaussian simulation design favors LDA
**Status:** addressed as explicit limitation.

**Implemented:** added a limitation statement that Gaussian class-mean simulations may favor LDA and may not fully transfer to non-Gaussian/mixed/nonlinear regimes; non-Gaussian simulation extension identified as future work.

**Locations:**
- Paper/main.tex (LdaBoost Simulations subsection)

---

### 8) HAR case where baseline GBM is stronger in CV
**Status:** addressed with focused failure-mode interpretation.

**Implemented:** expanded HAR discussion to explain why reducing 561 signals to $C-1=5$ discriminant components can lose nonlinear interaction information that trees exploit in the original feature space.

**Locations:**
- Paper/main.tex (HAR subsection in Results)

## Validation performed
- LaTeX build successful after revisions:
  - Paper/main.tex compiled to Paper/main.pdf
- Python syntax checks passed:
  - LdaBoost/algorithm.py
  - simulations/LdaBoosting/algorithm.py
  - simulations/runtime_ablation.py
- Runtime-ablation artifacts generated successfully under simulations/output_pipeline_confront.

## Progress status
- Current plan position: Phase 5, Step 21 (final consistency check) completed.
- Operational decision resolved: keep the new figure set and accept the current figure-file deltas for final packaging.
