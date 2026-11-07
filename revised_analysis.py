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
import shap
import xgboost
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
XAI_REPEAT = 1
XAI_STABILITY_FOLDS = 5


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


def original_feature_name(transformed_name: str, X: pd.DataFrame) -> str:
    """Map a one-hot/preprocessed feature name back to its source column."""
    name = transformed_name.split("__", 1)[-1]
    if name in X.columns:
        return name
    for column in X.select_dtypes(include=["object"]).columns:
        if name.startswith(f"{column}_"):
            return column
    return name


def normalise_shap_values(values) -> np.ndarray:
    """Handle SHAP's list and ndarray multiclass return conventions."""
    if hasattr(values, "values"):
        values = values.values
    values = np.asarray(values)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    if values.ndim != 3:
        raise ValueError(f"Expected three-dimensional multiclass SHAP values, got {values.shape}.")
    return values


def write_architecture_figure() -> None:
    """Draw the actual duplicate-aware analysis workflow used by this script."""
    fig, axis = plt.subplots(figsize=(18, 8.5))
    axis.set_axis_off()
    boxes = [
        (0.03, 0.66, 0.26, 0.23, "1  Data audit", "374 rows\nLiteral outcomes only\nNo target imputation", "#e8f1fb"),
        (0.37, 0.66, 0.26, 0.23, "2  Deterministic preparation", "Remove Person ID; harmonise BMI\nParse blood pressure\n109 predictor-profile groups", "#e8f1fb"),
        (0.71, 0.66, 0.26, 0.23, "3  Outer evaluation", "2 shuffled repetitions × 5 folds\nStratifiedGroupKFold\nDuplicate groups never cross folds", "#e8f1fb"),
        (0.06, 0.17, 0.26, 0.25, "4  Inner stack construction", "Within outer training only:\n4-fold grouped OOF probabilities\nPreprocessing refit in every fold", "#edf7ef"),
        (0.38, 0.17, 0.26, 0.25, "5  Models", "Random Forest + XGBoost →\nlogistic meta-learner\nNo encoded-feature pass-through", "#edf7ef"),
        (0.70, 0.17, 0.26, 0.25, "6  Outer-test artefacts", "Predictions + probabilities\nMetrics, calibration, paired test\nXAI stability and local-audit files", "#edf7ef"),
    ]
    for x, y, width, height, heading, body, colour in boxes:
        axis.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=colour, edgecolor="#2a5c86", linewidth=1.6, transform=axis.transAxes,
        ))
        axis.text(x + 0.02, y + height - 0.06, heading, transform=axis.transAxes,
                  fontsize=13, weight="bold", color="#163d5c", va="top")
        axis.text(x + width / 2, y + height / 2 - 0.025, body, transform=axis.transAxes,
                  fontsize=11.2, ha="center", va="center", linespacing=1.45)
    arrows = [((0.29, 0.775), (0.37, 0.775)), ((0.63, 0.775), (0.71, 0.775)),
              ((0.84, 0.66), (0.84, 0.46)), ((0.70, 0.295), (0.64, 0.295)),
              ((0.38, 0.295), (0.32, 0.295)), ((0.18, 0.42), (0.18, 0.62))]
    for start, end in arrows:
        axis.add_patch(FancyArrowPatch(start, end, transform=axis.transAxes,
                                       arrowstyle="-|>", mutation_scale=18,
                                       linewidth=1.8, color="#2a5c86"))
    axis.text(0.5, 0.04, "Every learned operation is fitted within the relevant training partition; outer-test groups are never used for fitting or tuning.",
              transform=axis.transAxes, ha="center", fontsize=11, color="#424242")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUTPUT_DIR / "architecture_duplicate_aware.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_xai_artifacts(X: pd.DataFrame, y: np.ndarray, groups: pd.Series) -> None:
    """Create auditable SHAP diagnostics for the XGBoost base learner.

    This is deliberately labelled as a base-learner diagnostic, not an
    explanation of the full stack.  It uses five outer-test folds from the
    first grouped repetition and writes both fold-to-fold stability and one
    local prediction audit with an additivity/fidelity residual.
    """
    seed = SEEDS[XAI_REPEAT - 1]
    outer = StratifiedGroupKFold(n_splits=XAI_STABILITY_FOLDS, shuffle=True, random_state=seed)
    feature_rows, local_rows = [], []
    local_metadata = None

    for fold, (train_index, test_index) in enumerate(outer.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        fitted = make_base_pipeline("XGBoost", seed + fold, X_train)
        fitted.fit(X_train, y[train_index])
        probabilities = aligned_probabilities(fitted, X_test)
        preprocessor = fitted.named_steps["preprocessor"]
        classifier = fitted.named_steps["classifier"]
        transformed_test = preprocessor.transform(X_test)
        explainer = shap.TreeExplainer(classifier)
        shap_values = normalise_shap_values(explainer.shap_values(transformed_test))
        feature_names = list(preprocessor.get_feature_names_out())
        all_class_importance = np.abs(shap_values).mean(axis=(0, 2))
        aggregated: dict[str, float] = {}
        for feature_name, importance in zip(feature_names, all_class_importance):
            source = original_feature_name(feature_name, X)
            aggregated[source] = aggregated.get(source, 0.0) + float(importance)
        ranking = pd.Series(aggregated).sort_values(ascending=False)
        for rank, (feature, importance) in enumerate(ranking.items(), start=1):
            feature_rows.append({"repeat": XAI_REPEAT, "fold": fold, "feature": feature,
                                 "mean_abs_shap": importance, "rank": rank,
                                 "explained_model": "XGBoost base learner"})

        if fold == 1:
            local_position = int(np.argmax(probabilities.max(axis=1)))
            predicted_class = int(probabilities[local_position].argmax())
            raw_margin = np.asarray(classifier.predict(transformed_test, output_margin=True))
            expected = np.asarray(explainer.expected_value)
            reconstructed_margin = float(expected[predicted_class] + shap_values[local_position, :, predicted_class].sum())
            fidelity_residual = abs(reconstructed_margin - float(raw_margin[local_position, predicted_class]))
            local_metadata = {
                "repeat": XAI_REPEAT, "fold": fold, "source_row": int(test_index[local_position]),
                "duplicate_group": str(groups.iloc[test_index[local_position]]),
                "true_label": str(LABELS[y[test_index[local_position]]]),
                "predicted_label": str(LABELS[predicted_class]),
                "predicted_probability": float(probabilities[local_position, predicted_class]),
                "explained_model": "XGBoost base learner (not the stack)",
                "fidelity_measure": "absolute TreeSHAP additivity residual on raw-margin scale",
                "fidelity_abs_residual": fidelity_residual,
            }
            for class_index, label in enumerate(LABELS):
                local_metadata[f"p_{label.replace(' ', '_')}"] = float(probabilities[local_position, class_index])
            for feature_name, value in zip(feature_names, shap_values[local_position, :, predicted_class]):
                local_rows.append({**local_metadata, "transformed_feature": feature_name,
                                   "source_feature": original_feature_name(feature_name, X),
                                   "shap_value_raw_margin": float(value)})

    features = pd.DataFrame(feature_rows)
    features.to_csv(OUTPUT_DIR / "xai_feature_stability_by_fold.csv", index=False)
    stability = features.groupby("feature")["mean_abs_shap"].agg(
        mean="mean", sd="std", min="min", max="max",
        p10=lambda values: values.quantile(0.10), p90=lambda values: values.quantile(0.90),
    )
    rank_summary = features.groupby("feature")["rank"].agg(median_rank="median", min_rank="min", max_rank="max")
    stability = stability.join(rank_summary).sort_values("mean", ascending=False).reset_index()
    stability["stability_band"] = "10th--90th empirical percentile across 5 grouped outer folds"
    stability["explained_model"] = "XGBoost base learner; not the stacked classifier"
    stability.to_csv(OUTPUT_DIR / "xai_feature_stability_summary.csv", index=False)

    display = stability.head(12).iloc[::-1]
    fig, axis = plt.subplots(figsize=(13.5, 8.2))
    lower = (display["mean"] - display["p10"]).clip(lower=0)
    upper = (display["p90"] - display["mean"]).clip(lower=0)
    axis.barh(display["feature"], display["mean"], color="#2f6f9f", alpha=0.9)
    axis.errorbar(display["mean"], display["feature"], xerr=np.vstack([lower, upper]),
                  fmt="none", color="#173f5f", capsize=4, linewidth=1.5,
                  label="10th--90th fold-to-fold band")
    axis.set(xlabel="Mean absolute TreeSHAP value (raw-margin scale)",
             title="Feature-attribution stability: XGBoost base learner\n(first repeat; five grouped outer-test folds)")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "xai_feature_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    local_frame = pd.DataFrame(local_rows)
    local_frame.to_csv(OUTPUT_DIR / "xai_local_audit.csv", index=False)
    (OUTPUT_DIR / "xai_local_audit.json").write_text(json.dumps(local_metadata, indent=2) + "\n")
    plot_rows = local_frame.reindex(local_frame["shap_value_raw_margin"].abs().sort_values(ascending=False).head(12).index).iloc[::-1]
    fig, axis = plt.subplots(figsize=(13.5, 8.2))
    colours = np.where(plot_rows["shap_value_raw_margin"] >= 0, "#b64b4b", "#3d7ea6")
    axis.barh(plot_rows["transformed_feature"], plot_rows["shap_value_raw_margin"], color=colours)
    axis.axvline(0, color="0.25", linewidth=0.9)
    axis.set(xlabel="TreeSHAP contribution to predicted-class raw margin",
             title=(f"Local XGBoost explanation: CSV row {local_metadata['source_row']} (0-based), "
                    f"true={local_metadata['true_label']}, predicted={local_metadata['predicted_label']}, "
                    f"P={local_metadata['predicted_probability']:.3f}"))
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "xai_local_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


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
        "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__,
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
    write_architecture_figure()
    write_xai_artifacts(X, y, groups)
    print(json.dumps(metadata, indent=2))
    print("\nRepeated group-aware nested-validation summary:")
    print(summary.round(4).to_string())
    print("\nPaired comparison:")
    print(json.dumps(paired_report, indent=2))


if __name__ == "__main__":
    main()
