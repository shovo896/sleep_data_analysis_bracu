# Response to reviewers

**Manuscript:** NEURI-D-26-00299  
**Revised title:** *An Explainable Stacking Framework for Multiclass Recorded Sleep-Status Classification*

We thank the editor and reviewers. The revision substantially changes both the analysis and the paper. We verified that `None` is a literal source-file string rather than a missing value, discovered extensive duplicate predictor profiles after identifier removal, replaced the single hold-out result with repeated duplicate-aware validation, released executable code and all predictions/splits, removed unvalidated explainability analyses, and narrowed the manuscript to a reproducibility-focused benchmark audit. We do not retain the earlier 97.33% single-split result as a primary finding.

## Reviewer 1

1. **Outcome coding.** We now read the CSV with missing-value conversion disabled and stop if an empty or undocumented outcome occurs. The revised audit establishes 219 literal `None`, 77 `Insomnia`, and 78 `Sleep Apnea` values, with zero blanks/missing values. We do not interpret `None` as a clinically verified absence of disorder (Methods, “Data audit and target definition”; Table 1).
2. **Primary configuration.** All primary analyses now use only the specified RF+XGBoost stack with a logistic meta-learner. SVM has been removed from the manuscript and primary code. Soft voting uses the same two base learners.
3. **Reproducibility.** `revised_analysis.py` is a single executable analysis. `revision_results/metadata.json` records Python, pandas, NumPy, scikit-learn, and XGBoost versions; `split_assignments.csv` has every fold/group assignment; and the code refits preprocessing inside every inner base-model fold. `requirements.txt` is updated.
4. **Validation and matched comparisons.** We replaced the single split with two shuffled repetitions of five-fold StratifiedGroupKFold. Exact retained-predictor duplicates never cross a fold. Comparators are stacking, soft voting, Random Forest, XGBoost, logistic regression, shallow decision tree, and histogram gradient boosting, all on identical outer folds (Table 2).
5. **Statistics.** `predictions.csv` provides reference labels, predictions, and class probabilities for every model/fold/repetition. `paired_comparison.json` provides the matched accuracy bootstrap and exact McNemar calculation. Stacking is not claimed to be superior: its accuracy difference from soft voting is -0.0031 (95% bootstrap interval -0.0162 to 0.0070), and the first-repeat McNemar result is p=0.6875.
6. **Explanations.** We removed SHAP, LIME, PCA, and ALE results rather than retaining unstable explanatory figures. The revised paper explains why their interpretation is not supported by this duplicate-rich, conflicting-label dataset.

## Reviewer 2

1. **Novelty.** We no longer present stacking itself as novel. The stated contribution is the auditable duplicate-aware evaluation protocol and the resulting correction to the earlier optimistic estimate.
2. **Clinical interpretation.** “Risk stratification,” “clinical decision support,” and diagnostic claims were removed. The title, abstract, discussion, conclusion, and limitations identify the task as prediction of recorded source-file labels only.
3. **Rigorous validation.** The single 75-row hold-out result is no longer the primary result. Repeated grouped outer validation, performance variability, probability scores, calibration artefacts, and a paired comparison are reported.
4. **Explanation methods.** The post-hoc explanation claims were removed, not treated as biological or clinical mechanisms.
5. **Competitive tabular comparator.** Histogram gradient boosting was added alongside the conventional baseline, tree, and ensemble comparators. It is evaluated on the same grouped folds.
6. **Presentation and references.** The manuscript was rewritten and condensed, with newly generated tables rather than decorative figures. References were reduced to sources directly relevant to the methods and dataset. The suggested traffic forecasting, image segmentation, and remote-sensing object-detection papers are outside the topic and are therefore not cited.

## Reviewer 3

1. The Discussion now explicitly separates internal benchmark discrimination from the requirements for real clinical screening.
2. The Results now explains the small number of discordant predictions, the confidence interval crossing zero, and the limited power of the paired comparison.
3. Rather than speculate about collinearity-driven SHAP/LIME rankings, we removed those rankings and state the reason.
4. The inconsistent RF+XGBoost+SVM robustness analysis is removed completely.
5. The Limitations now notes possible classification bias and unknown clinical meaning of the literal `None` label.
6. Local explanation figures are removed; the released prediction file records all information needed to audit each model prediction.
7. The Limitations recommends prospective cohorts, development-only tuning, calibration assessment, and external validation.
8. Suggested citations were assessed for topical relevance; none are about sleep-health tabular modelling or the reported methodology, so they were not added.

## Reviewer 4

1. We verified the source labels and report the result explicitly. We also found 265 duplicate retained-predictor rows and 22 groups with conflicting recorded outcomes; group-based validation is now mandatory.
2. The revised code generates every numerical result and exports versions, groups, splits, probabilities, and comparisons. The wording “supplied” and unverified-result language has been removed.
3. No test-set hyperparameter tuning occurs. Model specifications are fixed and stated. Person ID is removed before group construction.
4. Repeated grouped validation replaces the single random split. Logistic regression, a shallow tree, and histogram gradient boosting are added. Multiclass Brier scores and first-repetition reliability curves are released.
5. SHAP/LIME/ALE material has been removed from the manuscript and title.

## Reviewer 5

1. The raw import, literal-label check, cleaning steps, and outcome counts are implemented in `load_data()` and reported in Table 1.
2. The 75-observation 97.33% estimate is no longer used as evidence. The revised results report ten duplicate-aware outer folds and their variability.
3. Complete executable code, package versions, seeds, group assignments, fold metrics, and observation-level predictions are included in the repository.
4. The two-learner RF+XGBoost configuration is used consistently throughout. SVM has been removed.

## Files supplied for revision

- `sleep/JESTCH_Templates/NEURI_revision_targeted.tex` and
  `sleep/JESTCH_Templates/NEURI_revision_body.tex`: revised manuscript source
  in the original journal template.
- `response_to_reviewers.md`: this point-by-point response.
- `revised_analysis.py`: executable analysis.
- `revision_results/`: regenerated metadata, split IDs, metrics, predictions/probabilities, paired statistics, and calibration files.
