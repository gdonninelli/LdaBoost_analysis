# Main.tex Action Plan From Photo Notes (Timestamp Order)

This checklist follows the screenshots in chronological order and maps each requested change to concrete locations in [Paper/main.tex](Paper/main.tex). For yellow highlights, ready-to-paste text is proposed below.

## Image 1 (Reviewer 1, points 1-3)

### What to do in main.tex
1. Align the tuning narrative with the actual experiment status in [Paper/main.tex](Paper/main.tex#L322).
2. Add (or explicitly postpone) the LDA-components ablation requested for HAR and YEAST in [Paper/main.tex](Paper/main.tex#L332).
3. Keep statistical-language claims conservative where tests are not significant, especially in [Paper/main.tex](Paper/main.tex#L83) and [Paper/main.tex](Paper/main.tex#L474).

### Yellow-highlight text proposal (Point 2, LDA component ablation)
Use this if you want to state it as an additional planned/ongoing analysis:

As a sensitivity analysis, we evaluate how LDA+GBM changes when the number of retained discriminant components is set below the theoretical maximum C-1. Specifically, for HAR (C=6) we consider r in {1, 3, 5}, and for YEAST (C=10) we consider r in {1, 5, 9}. This ablation isolates how much of the gain is driven by supervised dimensionality itself versus the specific choice r=C-1.

If you already run it, append one sentence with the observed pattern and best r per dataset.

### Yellow-highlight text proposal (Point 3, significance wording)
Use this if you want to match the reviewer-safe wording:

Based on the available test-set comparisons, we do not observe statistically significant differences among methods at alpha=0.05; therefore, results are interpreted as descriptive performance patterns rather than inferential superiority claims.

## Image 2-3 (Reviewer 1, point 4: computational burden + Table A1)

### What to do in main.tex
1. Insert a short bridge paragraph immediately before [Paper/main.tex](Paper/main.tex#L450) to connect real-data runtime evidence to the simulation section.
2. Add one Appendix table named Table A1 (single runtime table, not two separate runtime tables), placed before [Paper/main.tex](Paper/main.tex#L590).
3. Keep runtime interpretation concise in Results and move full runtime detail to Appendix.

Runtime-ablation source files already available for this step:
- [simulations/runtime_ablation.py](simulations/runtime_ablation.py)
- [simulations/output_pipeline_confront/runtime_ablation_summary.csv](simulations/output_pipeline_confront/runtime_ablation_summary.csv)

### Yellow-highlight text proposal (before the simulations paragraph)
Before moving to simulation results, we report the computational profile on real datasets in Appendix Table A1 (mean runtime by dataset and method). The table shows that one-shot LDA+GBM is generally the fastest configuration, while LdaBoost offers a favorable accuracy-runtime compromise despite iterative projection updates.

### Yellow-highlight text proposal (Appendix intro for Table A1)
Table A1 reports mean runtime by dataset and method (seconds), aggregated across validation folds/runs. This consolidated table is provided to summarize computational cost in one place.

### Images/tables to include or keep
1. Keep the real-dataset summary figure [Paper/figures/accuracy_comparison.pdf](Paper/figures/accuracy_comparison.pdf), currently included at [Paper/main.tex](Paper/main.tex#L432).
2. Keep the simulation figures already linked in [Paper/main.tex](Paper/main.tex#L458).
3. Add one consolidated Appendix runtime table (Table A1).

## Image 4-5 (Reviewer 1, point 6: positioning vs LPBoost/FilterBoost/kernel projection)

### What to do in main.tex
1. Expand positioning text just before the sentence at [Paper/main.tex](Paper/main.tex#L77).
2. Keep the distinction clear: LdaBoost updates supervised projections inside boosting rounds, not as one-shot preprocessing.

### Yellow-highlight text proposal (to place before line 77)
Related boosting variants have modified learning dynamics through margin optimization (for example LPBoost), filtering/sampling strategies (for example FilterBoost), or projection-based transformations in kernel spaces. LdaBoost is different in that it updates a supervised LDA projection at each boosting stage using the current gradient signal, so feature projection and ensemble refinement are coupled inside the same stagewise loop rather than separated into preprocessing and prediction phases.

## Image 6 (Reviewer 1, points 7-8 + minor point on significant/significantly)

### What to do in main.tex
1. Add explicit limitations text in a Discussion section between [Paper/main.tex](Paper/main.tex#L374) and [Paper/main.tex](Paper/main.tex#L578).
2. Add a focused HAR “failure mode” interpretation in that same Discussion section (you already have a good version in Results at [Paper/main.tex](Paper/main.tex#L386), but reviewer intent is a Discussion placement).
3. Replace over-strong significance wording in [Paper/main.tex](Paper/main.tex#L83) and [Paper/main.tex](Paper/main.tex#L474).

### Yellow-highlight text proposal (Point 7, simulation limitation)
The simulation design is based on multivariate normal generators with class-specific means, which is a favorable setting for discriminant methods. For this reason, these results should be interpreted as controlled-condition evidence rather than direct guarantees for non-Gaussian, heavy-tailed, mixed-type, or strongly nonlinear real-data scenarios. Extending the simulation campaign to those settings is a priority for future work.

### Yellow-highlight text proposal (Point 8, HAR failure mode)
The HAR results highlight a relevant boundary case: compressing 561 original predictors to C-1=5 discriminant components can remove nonlinear interaction structure that tree ensembles can exploit in the full space. This explains why baseline GBM can remain strongest in cross-validation on HAR, even when LDA-based pipelines remain highly competitive on holdout performance.

### Yellow-highlight text proposal (minor wording cleanup)
Suggested replacement at [Paper/main.tex](Paper/main.tex#L83):

The results show that incorporating LDA-based features can improve GBM accuracy, with LdaBoost often providing the most consistent improvements over the baseline.

Suggested replacement at [Paper/main.tex](Paper/main.tex#L474):

Notably, in both binary and multiclass settings, LDA+GBM closely mirrored the performance trajectory of LdaBoost. In the binary case, LdaBoost often outperformed LDA+GBM at moderate feature dimensionality, suggesting that LDA-based extraction can improve predictive performance in more complex feature spaces.

## Image 7 (Reviewer 2 major concerns on Step 1 validity)

### What to do in main.tex
1. Add a dedicated Discussion opener that clarifies pseudo-labels versus residuals and why Fisher optimization remains valid each round.
2. Place it as the first paragraph of a new Discussion section (between Results and Conclusion).

### Yellow-highlight text proposal (Discussion opening paragraph)
In LdaBoost, LDA is not fitted directly on continuous residuals. At each iteration, gradient information is converted into pseudo-labels via the largest positive class-wise component, yielding a categorical partition on which Fisher optimization is well-defined. The stagewise projection therefore tracks the directions where the current model makes the largest class errors, while preserving the supervised categorical structure required by LDA.

## Image 8 (Reviewer 2 minor concerns + “complete Discussion section” note)

### What to do in main.tex
1. Add an explicit Discussion section between [Paper/main.tex](Paper/main.tex#L374) and [Paper/main.tex](Paper/main.tex#L578).
2. In that section, integrate four short blocks:
   - method rationale (pseudo-label mechanism),
   - binary-case interpretation,
   - simulation-limitations statement,
   - HAR failure-mode interpretation.
3. Keep Results for empirical reporting; move interpretation-heavy text into Discussion.

## Discrepancies vs RESULTS.md (highlighted in red)

- <span style="color:red"><strong>DISCREPANCY 1:</strong> The photo-response claim says all methods were fully and independently retuned, but [RESULTS.md](RESULTS.md#L4) states scope = learning-rate-only, with a grid only for learning_rate in [RESULTS.md](RESULTS.md#L5).</span>

- <span style="color:red"><strong>DISCREPANCY 2:</strong> The tuning narrative in [Paper/main.tex](Paper/main.tex#L324) and [Paper/main.tex](Paper/main.tex#L328) describes broader tuning and an additional LdaBoost_tuned step, while [RESULTS.md](RESULTS.md#L1) to [RESULTS.md](RESULTS.md#L5) report a 3x3 outer/inner setup with learning-rate-only tuning for the reported tables.</span>

- <span style="color:red"><strong>DISCREPANCY 3:</strong> Rainfall holdout sample sizes differ: [Paper/main.tex](Paper/main.tex#L335) reports 2,190 train + 730 test context, while [RESULTS.md](RESULTS.md#L33) reports fallback holdout 1,752 train + 438 test.</span>

- <span style="color:red"><strong>DISCREPANCY 4:</strong> The yellow photo note requests an ablation on the number of retained LDA discriminant components (HAR and YEAST), but [RESULTS.md](RESULTS.md#L9) to [RESULTS.md](RESULTS.md#L47) contain only the four-method comparison and no LDA-component ablation output.</span> -> run the lda_components_ablation.py on the LDA+GBM.

- <span style="color:red"><strong>DISCREPANCY 5:</strong> The photo notes request a single Appendix runtime Table A1; currently runtime is integrated into Table 1 in [Paper/main.tex](Paper/main.tex#L400) and [Paper/main.tex](Paper/main.tex#L418), and a separate runtime-ablation block exists in Results, so Appendix Table A1 is still not explicitly present.</span>

Runtime-ablation confirmation:
- The runtime ablation has been computed and is reproducible from [simulations/runtime_ablation.py](simulations/runtime_ablation.py), with summary values in [simulations/output_pipeline_confront/runtime_ablation_summary.csv](simulations/output_pipeline_confront/runtime_ablation_summary.csv).
- These values are consistent with the runtime-ablation table currently reported in [Paper/main.tex](Paper/main.tex#L494).

Numeric consistency check: the CV, test-accuracy, and timing values in [RESULTS.md](RESULTS.md#L9) to [RESULTS.md](RESULTS.md#L47) are aligned with the current values in [Paper/main.tex](Paper/main.tex#L408) to [Paper/main.tex](Paper/main.tex#L420).

## Practical next edit order in main.tex
1. Fix tuning narrative and significance wording in [Paper/main.tex](Paper/main.tex#L322).
2. Insert the bridge text before [Paper/main.tex](Paper/main.tex#L450).
3. Add the new Discussion section between [Paper/main.tex](Paper/main.tex#L374) and [Paper/main.tex](Paper/main.tex#L578), using the yellow-text proposals above.
4. Add consolidated Table A1 in Appendix near [Paper/main.tex](Paper/main.tex#L590).
5. Re-check all numeric/sample-size statements against [RESULTS.md](RESULTS.md).