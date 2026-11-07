# Final targeted revision -- NEURI-D-26-00299

Use the files below for the revised submission. They replace the earlier
exploratory revision drafts at the repository root.

## Manuscript

- `sleep/JESTCH_Templates/NEURI_revision_targeted.tex`: primary manuscript in
  the original journal template.
- `sleep/JESTCH_Templates/NEURI_revision_body.tex`: reviewer-aligned Methods,
  Results, Discussion, Limitations, and Conclusion included by the primary
  source.
- `sleep/JESTCH_Templates/NEURI_revision_targeted.pdf`: compiled final
  manuscript (five pages).
- `sleep/JESTCH_Templates/calibration_first_repeat.png`: generated internal
  calibration figure used by the manuscript.
- `sleep/JESTCH_Templates/architecture_duplicate_aware.png`: generated
  full-width workflow figure matching the executable grouped-validation code.
- `sleep/JESTCH_Templates/xai_feature_stability.png` and
  `sleep/JESTCH_Templates/xai_local_audit.png`: generated full-width,
  scope-limited XGBoost base-learner XAI figures.

The original submitted source,
`sleep/JESTCH_Templates/JESTECH_sleep_stacking_prism_clean.tex`, is unchanged
for provenance.

## Response letter

- `response_to_reviewers.pdf`: compiled point-by-point response.
- `response_to_reviewers.tex` and `response_to_reviewers.md`: editable source
  versions.

## Reproducibility package

- `revised_analysis.py`: executable duplicate-aware analysis.
- `requirements.txt`: required Python packages.
- `revision_results/`: generated metadata, grouped split assignments,
  fold-level metrics, observation-level predictions/probabilities, paired
  statistics, calibration data/figure, XAI feature-stability tables, and local
  XAI audit files.

Run `python3 revised_analysis.py` from the repository root to regenerate
`revision_results/`. The old `stacking_cls.ipynb` is retained only as source
provenance and must not be used for manuscript results.
