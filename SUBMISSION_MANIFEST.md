# Revision artefacts

`revised_analysis.py` is the single executable analysis for the revision. The
companion `revised_stacking_cls.ipynb` runs that exact analysis from Jupyter. It
reads the raw CSV without outcome imputation, verifies the literal `None`
label, and uses grouped nested validation so exact predictor duplicates cannot
cross a development/evaluation boundary.

Run:

```bash
python revised_analysis.py
```

It recreates `revision_results/metadata.json`, split assignments, per-fold
metrics, observation-level probabilities, and the paired comparison.  These
ge[text](../Desktop/sleep)nerated files are the source of every numerical claim in the revised paper.

Submission files are `revised_submission.pdf` with its `.tex` and `.bib`
sources, plus `response_to_reviewers.pdf` with editable `.tex` and `.md`
versions. `revised_submission_package.zip` collects these documents, the code,
notebook, requirements, and all generated reproducibility artefacts.

The older exploratory notebooks, including `stacking_cls.ipynb`, are retained
for provenance only and must not be used to generate manuscript results.
