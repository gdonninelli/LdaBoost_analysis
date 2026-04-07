# Comparison: Implemented Changes vs Proposed Changes in New.md

## Scope
This note compares:
1. What is already implemented in the repository and manuscript.
2. What your colleague proposes in [New.md](New.md).
3. What is still missing or implemented differently.

Reference implementation files:
- [Paper/main.tex](Paper/main.tex)
- [Paper/response_to_reviewers.md](Paper/response_to_reviewers.md)
- [LdaBoost/algorithm.py](LdaBoost/algorithm.py)
- [simulations/LdaBoosting/algorithm.py](simulations/LdaBoosting/algorithm.py)
- [simulations/runtime_ablation.py](simulations/runtime_ablation.py)

Latest related commits already in branch:
- `b1b66a2` (further fix)
- `28eeb68` (adding significance)

## Executive Status Matrix
| Point in New.md | Current status | What was done | Difference vs proposal |
|---|---|---|---|
| R1-1 Tuning asymmetry | Partially addressed | Explicit limitation text added in [Paper/main.tex](Paper/main.tex) (Tuning strategy) | Full independent retuning of all methods was **not** implemented |
| R1-2 LDA components ablation | Partially addressed | Limitation acknowledged in [Paper/main.tex](Paper/main.tex) | New LDA-component ablation for HAR/YEAST (c=1,3,5 / c=1,5,9) was **not** run |
| R1-3 Statistical significance | Implemented differently | Nonparametric tests added (simulation; plus real-dataset artifacts present) and discussed in [Paper/main.tex](Paper/main.tex) | Instead of only discussing limitation, actual tests were run; however real-dataset test artifacts are currently untracked |
| R1-4 Computational overhead | Implemented | Complexity subsection + runtime ablation table and scripts in [Paper/main.tex](Paper/main.tex), [simulations/runtime_ablation.py](simulations/runtime_ablation.py) | Aligned with proposal |
| R1-5 Binary C=2 case | Implemented | Dedicated binary one-component interpretation added in [Paper/main.tex](Paper/main.tex) | Aligned with proposal |
| R1-6 Boosting literature positioning | Implemented | LPBoost/FilterBoost/projection-family positioning paragraph added in [Paper/main.tex](Paper/main.tex) | Aligned with proposal |
| R1-7 Gaussian simulation limitation | Implemented | Limitation explicitly stated in [Paper/main.tex](Paper/main.tex) | Non-Gaussian simulations still pending |
| R1-8 HAR failure mode | Implemented | Failure-mode analysis paragraph added in [Paper/main.tex](Paper/main.tex) | Aligned with proposal; re-tuning-based recheck still pending |
| Minor: remove “significantly” in abstract | Implemented (abstract) | Abstract wording softened in [Paper/main.tex](Paper/main.tex) | Word “significant” still appears in Introduction narrative; not in abstract claim |
| Minor: figure caption consistency | Not fully addressed | Some figure assets renamed/reorganized | A systematic caption harmonization pass is still advisable |
| Minor: limitations section | Partially addressed | Limitations added in relevant subsections and conclusion text | No standalone “Limitations” section header yet |
| R2 major 1/3/4 Step-1 validity | Implemented | Stagewise induced-label derivation + fallback/tie policy in manuscript and code: [Paper/main.tex](Paper/main.tex), [LdaBoost/algorithm.py](LdaBoost/algorithm.py), [simulations/LdaBoosting/algorithm.py](simulations/LdaBoosting/algorithm.py) | Aligned with proposal |
| R2 major 2 scalability of iterative LDA | Implemented | Complexity + runtime evidence added in [Paper/main.tex](Paper/main.tex), [simulations/runtime_ablation.py](simulations/runtime_ablation.py) | Aligned with proposal |
| R2 minor concerns (PCA notation, LDA wording, typos, N=1000 wording) | Implemented | All addressed and mapped in [Paper/response_to_reviewers.md](Paper/response_to_reviewers.md) | Aligned with proposal |

## Why your colleague and my implementation differ

### A. Different priority: fast reviewer compliance vs full experimental redesign
Your colleague's proposal is research-expansive (retune all models, add new LDA-component ablations, possibly add more simulations). My implementation strategy was initially reviewer-compliance focused: remove methodological ambiguity, add scalability evidence, and avoid introducing long-running experimental branches unless explicitly requested.

Practical consequence:
1. The manuscript became much stronger on theory/clarity/scalability quickly.
2. But it is not yet a fully symmetric benchmark study in the sense your colleague now requests.

### B. Point 1 (full retuning) is a scope change, not a text-only fix
Your colleague asks for independent retuning for all pipelines (at least add learning_rate search, keep subsample at 0.6). That is a valid methodological strengthening, but it is a new experiment campaign.

What I did instead:
1. I made the asymmetry explicit and transparent in [Paper/main.tex](Paper/main.tex).
2. I framed fixed-parameter comparisons as controlled transformation contrasts, not final leaderboard claims.

Why this matters:
1. This avoids overclaiming.
2. It does not resolve the asymmetry empirically.

### C. Point 2 (reduced LDA components) is also an experiment, not a wording fix
Your colleague's suggestion (HAR: c=1,3,5; YEAST: c=1,5,9) is concrete and useful.

What I did instead:
1. I acknowledged this asymmetry and marked reduced-LDA-component ablation as future work.

Why this matters:
1. Reviewers see intellectual honesty.
2. They may still ask for the actual table before acceptance.

### D. Point 3 (statistical tests) moved forward more than requested
Here I actually went beyond the earlier limitation-only approach:
1. Simulation significance tests were computed and integrated.
2. Real-dataset significance artifacts were also generated.

Remaining gap:
1. Manuscript/rebuttal synchronization still needs a final consistency pass so text and all artifacts tell the same story.

### E. Points 4/5/6/7/8: mostly aligned with your colleague
For these points, the manuscript now largely follows your colleague's direction:
1. Runtime/scalability: added theory + runtime ablation.
2. Binary C=2 case: now discussed explicitly.
3. Related boosting variants: positioning paragraph added.
4. Gaussian simulation limitation: explicitly acknowledged.
5. HAR failure mode: discussed with a concrete explanatory hypothesis.

## Point-by-point argumentation you can use in discussion

### 1) Tuning asymmetry
Colleague view: asymmetry may bias method ranking.

Current implementation view: acknowledged and bounded claim scope, but did not yet run full retuning.

Decision you need:
1. If goal is "major revision acceptance", full retuning is the safer path.
2. If goal is "defensible interim revision", current text is acceptable but weaker.

### 2) LDA components ablation
Colleague view: without c-variation, PCA vs LDA comparison remains asymmetric.

Current implementation view: limitation disclosed; no new table yet.

Decision you need:
1. Add a compact sensitivity table now (recommended).
2. Or keep as future work (risk: reviewer insists).

### 3) Statistical significance
Colleague view: mean±SD is not enough.

Current implementation view: tests were added, including real-dataset artifacts.

Decision you need:
1. Synchronize manuscript text with real-dataset significance artifacts.
2. Commit/signpost those files in supplementary material.

### 4) Computational overhead
Colleague view: missing practical runtime cost discussion.

Current implementation view: solved with complexity section + runtime ablation.

Decision you need:
1. Keep as is; this point is now defensible.

### 5) Binary C=2 interpretation
Colleague view: one LDA component per iteration is restrictive and needs explanation.

Current implementation view: addressed explicitly (iterative direction updates explain competitiveness).

Decision you need:
1. Keep as is; optional to add one extra sentence in conclusion.

### 6) Boosting literature positioning
Colleague view: needs relation to LPBoost/FilterBoost-like families.

Current implementation view: added conceptual relationship and distinction.

Decision you need:
1. If desired, add 1-2 citations for those families to further strengthen.

### 7) Gaussian simulation limitation
Colleague view: LDA-favorable DGP may limit generalization.

Current implementation view: limitation now explicit.

Decision you need:
1. If time permits, add one non-Gaussian scenario; otherwise limitation statement may suffice.

### 8) HAR failure mode
Colleague view: deserves deeper treatment.

Current implementation view: now discussed as supervised compression vs nonlinear interaction loss.

Decision you need:
1. Re-check after full retuning (Point 1), because ranking may change.

## Negotiation-ready summary (short)
If you need a direct message to your colleague, use this:

1. "Your methodological concerns are mostly integrated in text and interpretation (points 4-8 fully, points 1-2 as explicit limitations)."
2. "I also added formal significance tests (simulation + real-dataset artifacts), which goes beyond the previous mean±SD-only setup."
3. "The two remaining high-impact experimental tasks are: (i) full independent retuning across methods, and (ii) reduced-LDA-component ablation for HAR/YEAST."
4. "If we run these two, the revision becomes methodologically much harder to challenge."

## Statistical Significance: What exists now

### 1. Simulation-based significance (already integrated)
Implemented and cited in manuscript:
- [simulations/output_pipeline_confront/pipeline_confront_significance_tests.csv](simulations/output_pipeline_confront/pipeline_confront_significance_tests.csv)
- [simulations/output_pipeline_confront/pipeline_confront_significance_tests.json](simulations/output_pipeline_confront/pipeline_confront_significance_tests.json)

Main reported results (Friedman on matched blocks, n=25 each):
- Binary 1k: p=0.302
- Ternary 1k: p=0.073
- Quinary 1k: p=0.338

Interpretation currently used: no global difference at alpha=0.05 in these simulation blocks.

### 2. Real-dataset significance (artifacts exist, not yet integrated)
Untracked artifacts currently present:
- [real_datasets/real_dataset_fold_accuracies.json](real_datasets/real_dataset_fold_accuracies.json)
- [real_datasets/real_dataset_fold_accuracy_summary.csv](real_datasets/real_dataset_fold_accuracy_summary.csv)
- [real_datasets/real_dataset_significance_tests.csv](real_datasets/real_dataset_significance_tests.csv)
- [real_datasets/real_dataset_significance_tests.json](real_datasets/real_dataset_significance_tests.json)

Key p-values from [real_datasets/real_dataset_significance_tests.csv](real_datasets/real_dataset_significance_tests.csv):
- HAR Friedman p=0.000112 (global difference)
- SONAR Friedman p=0.00219 (global difference)
- Rainfall Friedman p=0.0527 (borderline)
- IRIS Friedman p=0.223 (ns)
- YEAST Friedman p=0.819 (ns)

Pairwise (Holm-corrected) shows significant gaps in HAR and SONAR, limited signal in Rainfall, none in IRIS/YEAST.

Important inconsistency to fix before submission:
- [Paper/main.tex](Paper/main.tex) still states real-data paired inferential testing is future work.
- But the above real-dataset significance files indicate those tests have now been run.

## Main Differences vs Colleague Proposal
1. The biggest difference is Point 1 (full retuning across methods): currently this is discussed as a limitation, not executed.
2. Point 2 (reduced LDA components ablation for HAR/YEAST) is also still not executed as a new analysis table.
3. Point 3 (significance testing) is now ahead of the original proposal in practice (simulation + real-dataset test artifacts exist), but manuscript alignment is incomplete for the real-dataset part.

## Recommended reply strategy to colleague
1. Confirm completed items: R1-4/5/6/7/8 and R2 major/minor methodological clarifications are implemented.
2. Confirm partial items: R1-1 and R1-2 are discussed but not experimentally completed.
3. Highlight statistical update: tests are available (including real datasets), but manuscript text needs a synchronization pass.
4. Decide whether to run full retuning and LDA-component ablation now; if yes, these are the two remaining high-impact experimental tasks.
