"""Reproducible, duplicate-aware evaluation for the sleep-status benchmark.

Run with: python revised_analysis.py

The script deliberately does not impute or relabel the outcome.  It verifies
that ``None`` is a literal source label, prevents exact predictor duplicates
from crossing a validation boundary, and writes every split and prediction
needed to reproduce the reported comparisons.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


DATA_PATH = Path("Sleep_health_and_lifestyle_dataset (1).csv")
OUTPUT_DIR = Path("revision_results")
LABELS = np.array(["Insomnia", "None", "Sleep Apnea"])
# Two independently shuffled grouped repetitions keep the run practical on a
# laptop while yielding ten outer-fold estimates.  Add seeds only in a planned
# sensitivity analysis, not after inspecting results.
SEEDS = (17, 29)
OUTER_FOLDS = 5
INNER_FOLDS = 4


def load_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series]:
    """Read and validate source labels; create duplicate-protection groups."""
    raw = pd.read_csv(path, keep_default_na=False)
    required = {"Person ID", "Sleep Disorder", "Blood Pressure", "BMI Category"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Required source columns are missing: {sorted(missing)}")

    outcome = raw["Sleep Disorder"].astype(str).str.strip()
    if outcome.eq("").any() or not set(outcome.unique()).issubset(set(LABELS)):
        raise ValueError(
            "The outcome must contain only literal labels Insomnia, None, and Sleep Apnea; "
            "blank or undocumented outcomes must be resolved before modelling."
        )
    if not (outcome == "None").any():
        raise ValueError("The literal None outcome category was not found.")

    data = raw.copy()
    data["BMI Category"] = data["BMI Category"].replace({"Normal Weight": "Normal"})
    bp = data["Blood Pressure"].astype(str).str.extract(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
    if bp.isna().any().any():
        raise ValueError("Blood Pressure contains values that cannot be parsed as systolic/diastolic.")
    data[["Systolic_BP", "Diastolic_BP"]] = bp.astype(int)
    X = data.drop(columns=["Person ID", "Sleep Disorder", "Blood Pressure"])

    # Identical predictors must never appear in both development and evaluation data.
    # Factorisation of the full predictor tuple avoids relying on hash equality.
    groups = pd.Series(pd.factorize(pd.MultiIndex.from_frame(X), sort=True)[0], index=X.index).astype(str)
    y = pd.Categorical(outcome, categories=LABELS, ordered=True).codes
    return X, y, groups, outcome


def make_preprocessor(X: pd.DataFrame, sparse_threshold: float = 0.3) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer(
        [("numeric", StandardScaler(), numeric),
         ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        sparse_threshold=sparse_threshold,
    )


def classifier_factory(name: str, seed: int):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=1
        )
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            random_state=seed, n_jobs=1,
        )
    raise KeyError(name)


def make_base_pipeline(name: str, seed: int, X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor(X)),
        ("classifier", classifier_factory(name, seed)),
    ])


def aligned_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Return columns in LABELS/coded-class order, even for a rare missing class."""
    probabilities = model.predict_proba(X)
    result = np.zeros((len(X), len(LABELS)))
    result[:, model.classes_.astype(int)] = probabilities
    # Avoid platform-specific floating-point round-off warnings in log-loss.
    return result / result.sum(axis=1, keepdims=True)


def fit_stack(
    X_train: pd.DataFrame, y_train: np.ndarray, groups_train: pd.Series,
    X_eval: pd.DataFrame, seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit manual OOF stacking with preprocessing re-fit in every inner fold."""
    inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    base_names = ("Random Forest", "XGBoost")
    oof = np.zeros((len(X_train), len(base_names) * len(LABELS)))
    final_probabilities: dict[str, np.ndarray] = {}

    for model_offset, name in enumerate(base_names):
        for fold, (inner_train, inner_valid) in enumerate(inner.split(X_train, y_train, groups_train)):
            fitted = make_base_pipeline(name, seed + fold, X_train)
            fitted.fit(X_train.iloc[inner_train], y_train[inner_train])
            oof[inner_valid, model_offset * len(LABELS):(model_offset + 1) * len(LABELS)] = aligned_probabilities(
                fitted, X_train.iloc[inner_valid]
            )
        final = make_base_pipeline(name, seed, X_train)
        final.fit(X_train, y_train)
        final_probabilities[name] = aligned_probabilities(final, X_eval)

    meta = LogisticRegression(max_iter=4000, C=1.0, random_state=seed)
    meta.fit(oof, y_train)
    stacked_features = np.hstack([final_probabilities[name] for name in base_names])
    stack_probabilities = np.zeros((len(X_eval), len(LABELS)))
    stack_probabilities[:, meta.classes_.astype(int)] = meta.predict_proba(stacked_features)
    stack_probabilities /= stack_probabilities.sum(axis=1, keepdims=True)
    return stack_probabilities, final_probabilities


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(len(LABELS))[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, probabilities, labels=np.arange(len(LABELS))),
        "multiclass_brier": multiclass_brier(y_true, probabilities),
    }


def add_record(rows, predictions, *, repeat, fold, model, indices, groups, y_true, probabilities):
    for local, source_index in enumerate(indices):
        predictions.append({
            "repeat": repeat, "fold": fold, "model": model,
            "source_row": int(source_index), "duplicate_group": str(groups.iloc[local]),
            "true_label": LABELS[y_true[local]], "predicted_label": LABELS[probabilities[local].argmax()],
            **{f"p_{label.replace(' ', '_')}": float(probabilities[local, j]) for j, label in enumerate(LABELS)},
        })
    rows.append({"repeat": repeat, "fold": fold, "model": model, **metrics(y_true, probabilities)})


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, y, groups, raw_outcome = load_data(DATA_PATH)
    metadata = {
        "n_rows": len(X), "n_predictor_duplicate_groups": int(groups.nunique()),
        "outcome_counts": raw_outcome.value_counts().reindex(LABELS).to_dict(),
        "outcome_is_literal": True, "exact_duplicate_predictor_rows": int(X.duplicated().sum()),
        "predictor_groups_with_conflicting_outcomes": int(
            pd.DataFrame({"group": groups, "outcome": raw_outcome}).groupby("group")["outcome"].nunique().gt(1).sum()
        ),
        "seeds": list(SEEDS), "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "python": sys.version, "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "model_selection": "Fixed specifications; no test-set tuning was performed.",
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    metric_rows, prediction_rows, split_rows = [], [], []
    for repeat, seed in enumerate(SEEDS, start=1):
        outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(outer.split(X, y, groups), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_groups, test_groups = groups.iloc[train_index], groups.iloc[test_index]
            if set(train_groups).intersection(set(test_groups)):
                raise RuntimeError("Duplicate-protection group leaked across a validation fold.")
            for idx in test_index:
                split_rows.append({"repeat": repeat, "fold": fold, "source_row": int(idx), "duplicate_group": str(groups.iloc[idx])})

            stack_probability, base_probability = fit_stack(X_train, y_train, train_groups, X_test, seed + fold)
            soft_probability = (base_probability["Random Forest"] + base_probability["XGBoost"]) / 2
            model_probabilities = {
                "Stacking (RF + XGBoost)": stack_probability,
                "Soft voting (RF + XGBoost)": soft_probability,
                "Random Forest": base_probability["Random Forest"],
                "XGBoost": base_probability["XGBoost"],
            }
            for name, estimator in {
                "Logistic regression": LogisticRegression(max_iter=4000, random_state=seed + fold),
                "Shallow decision tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=seed + fold),
            }.items():
                pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("classifier", estimator)])
                pipeline.fit(X_train, y_train)
                model_probabilities[name] = aligned_probabilities(pipeline, X_test)
            hist_gradient = Pipeline([
                ("preprocessor", make_preprocessor(X_train, sparse_threshold=0.0)),
                ("classifier", HistGradientBoostingClassifier(
                    max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed + fold
                )),
            ])
            hist_gradient.fit(X_train, y_train)
            model_probabilities["Histogram gradient boosting"] = aligned_probabilities(hist_gradient, X_test)
            for name, probability in model_probabilities.items():
                add_record(metric_rows, prediction_rows, repeat=repeat, fold=fold, model=name,
                           indices=test_index, groups=test_groups.reset_index(drop=True), y_true=y_test, probabilities=probability)

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    splits = pd.DataFrame(split_rows)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    splits.to_csv(OUTPUT_DIR / "split_assignments.csv", index=False)

    summary = fold_metrics.groupby("model")[["accuracy", "macro_f1", "log_loss", "multiclass_brier"]].agg(["mean", "std", "min", "max"])
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv")

    paired = fold_metrics.pivot(index=["repeat", "fold"], columns="model", values="accuracy")
    difference = paired["Stacking (RF + XGBoost)"] - paired["Soft voting (RF + XGBoost)"]
    rng = np.random.default_rng(20261106)
    bootstrap = np.array([rng.choice(difference.to_numpy(), size=len(difference), replace=True).mean() for _ in range(10000)])
    first_repeat = predictions[predictions["repeat"] == 1]
    stack = first_repeat[first_repeat["model"] == "Stacking (RF + XGBoost)"].sort_values("source_row")
    soft = first_repeat[first_repeat["model"] == "Soft voting (RF + XGBoost)"].sort_values("source_row")
    stack_correct = stack["true_label"].to_numpy() == stack["predicted_label"].to_numpy()
    soft_correct = soft["true_label"].to_numpy() == soft["predicted_label"].to_numpy()
    b, c = int((stack_correct & ~soft_correct).sum()), int((~stack_correct & soft_correct).sum())
    mcnemar_p = float(binomtest(min(b, c), n=b + c, p=0.5).pvalue) if b + c else 1.0
    paired_report = {
        "paired_accuracy_difference_stack_minus_soft": {
            "mean": float(difference.mean()), "sd": float(difference.std(ddof=1)),
            "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "n_repeat_fold_pairs": int(len(difference)),
        },
        "mcnemar_first_repeat_oof": {"stack_only_correct": b, "soft_only_correct": c, "exact_p": mcnemar_p},
        "caution": "Repeated folds are correlated; these summaries quantify internal-validation variability and are not external-validation evidence.",
    }
    (OUTPUT_DIR / "paired_comparison.json").write_text(json.dumps(paired_report, indent=2) + "\n")

    # Calibration is assessed on the first repetition's out-of-fold predictions;
    # it is descriptive, because it is still an internal validation exercise.
    calibration_rows = []
    fig, axes = plt.subplots(1, len(LABELS), figsize=(14, 4.2), sharey=True)
    for class_index, (class_name, axis) in enumerate(zip(LABELS, axes)):
        for model, colour in (("Stacking (RF + XGBoost)", "#1f77b4"),
                              ("Soft voting (RF + XGBoost)", "#ff7f0e")):
            frame = first_repeat[first_repeat["model"] == model]
            observed, predicted = calibration_curve(
                (frame["true_label"] == class_name).astype(int),
                frame[f"p_{class_name.replace(' ', '_')}"] , n_bins=5, strategy="quantile"
            )
            axis.plot(predicted, observed, "o-", label=model, color=colour)
            calibration_rows.extend({
                "model": model, "class": class_name, "mean_predicted_probability": float(x),
                "observed_frequency": float(y)
            } for x, y in zip(predicted, observed))
        axis.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1)
        axis.set(title=class_name, xlabel="Mean predicted probability", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Internal out-of-fold calibration, first grouped repetition", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_first_repeat.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(calibration_rows).to_csv(OUTPUT_DIR / "calibration_first_repeat.csv", index=False)
    print(json.dumps(metadata, indent=2))
    print("\nRepeated group-aware nested-validation summary:")
    print(summary.round(4).to_string())
    print("\nPaired comparison:")
    print(json.dumps(paired_report, indent=2))


if __name__ == "__main__":
    main()
