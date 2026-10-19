"""Train, compare, evaluate, and log end-to-end classifiers."""

import argparse
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

from heart_ml.data import FEATURES, TARGET, load_dataset
from heart_ml.eda import create_eda_figures
from heart_ml.pipeline import make_model

SEARCH_GRIDS: dict[str, dict[str, list[object]]] = {
    "logistic_regression": {"classifier__C": [0.1, 1.0, 10.0]},
    "random_forest": {
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 3],
    },
}
EXPERIMENT_NAME = "heart-disease-classification"


def train_and_log(
    data_path: str | Path,
    output_dir: str | Path = "models",
    tracking_uri: str | None = None,
) -> dict[str, object]:
    """Compare two models using train-only CV, evaluate holdout, and log artifacts."""
    dataset = load_dataset(data_path)
    if dataset[TARGET].nunique() != 2 or dataset[TARGET].value_counts().min() < 2:
        raise ValueError("Training needs at least two examples from each binary target class.")

    test_size = 0.2
    random_state = 42
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[FEATURES],
        dataset[TARGET],
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[TARGET],
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reports_path = Path("reports")
    reports_path.mkdir(parents=True, exist_ok=True)
    cleaned_path = Path("data/processed/heart_cleaned.csv")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(cleaned_path, index=False)
    eda_path = reports_path / "figures"
    create_eda_figures(dataset, eda_path)
    training_metadata = {
        "dataset_rows": len(dataset),
        "features": FEATURES,
        "target": TARGET,
        "positive_class_definition": "original target num > 0",
        "test_size": test_size,
        "random_state": random_state,
        "cross_validation": {
            "type": "StratifiedKFold",
            "n_splits": cv.get_n_splits(),
            "shuffle": True,
        },
        "selection_metric": "roc_auc",
        "search_grids": SEARCH_GRIDS,
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "mlflow": importlib.metadata.version("mlflow"),
            "joblib": importlib.metadata.version("joblib"),
        },
    }
    candidate_artifacts: dict[str, dict[str, Path]] = {}

    local_tracking_uri = f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri or local_tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_location = (Path.cwd() / "mlruns" / EXPERIMENT_NAME).resolve().as_uri()
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
    mlflow.set_experiment(EXPERIMENT_NAME)
    results: list[dict[str, object]] = []
    best_candidate: tuple[float, str, GridSearchCV] | None = None

    for model_name, grid in SEARCH_GRIDS.items():
        search = GridSearchCV(
            estimator=make_model(model_name),
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        tuning_path = reports_path / f"{model_name}_tuning.csv"
        pd.DataFrame(search.cv_results_).loc[
            :, ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].to_csv(tuning_path, index=False)
        probabilities = search.best_estimator_.predict_proba(x_test)[:, 1]
        predictions = search.best_estimator_.predict(x_test)
        metrics = {
            "cv_roc_auc_mean": float(search.best_score_),
            "cv_roc_auc_std": float(
                cross_val_score(
                    search.best_estimator_, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
                ).std()
            ),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_params(search.best_params_)
            mlflow.log_param("cv_folds", cv.get_n_splits())
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("scoring", "roc_auc")
            mlflow.log_param("positive_class_definition", "original target num > 0")
            mlflow.log_text(
                json.dumps(grid, indent=2),
                artifact_file="metadata/search_grid.json",
            )
            mlflow.log_dict(training_metadata, artifact_file="metadata/training_metadata.json")
            mlflow.log_artifact("requirements.txt", artifact_path="environment")
            mlflow.log_metrics(metrics)
            mlflow.log_metric("training_rows", len(x_train))
            mlflow.log_metric("holdout_rows", len(x_test))
            mlflow.log_artifact(str(tuning_path), artifact_path="tuning")
            model_metrics_path = reports_path / f"{model_name}_metrics.json"
            model_metrics_path.write_text(
                json.dumps(
                    {"model": model_name, "best_params": search.best_params_, **metrics},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            mlflow.log_artifact(str(model_metrics_path), artifact_path="metrics")
            mlflow.log_artifact(str(cleaned_path), artifact_path="data")
            mlflow.log_artifacts(str(eda_path), artifact_path="eda")

            figure, axis = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test, predictions, display_labels=["no disease", "disease"], ax=axis
            )
            axis.set_title(f"{model_name}: holdout confusion matrix")
            confusion_path = reports_path / f"{model_name}_confusion_matrix.png"
            figure.tight_layout()
            figure.savefig(confusion_path, dpi=140)
            plt.close(figure)
            mlflow.log_artifact(str(confusion_path), artifact_path="plots")

            figure, axis = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, probabilities, ax=axis)
            axis.set_title(f"{model_name}: holdout ROC curve")
            roc_path = reports_path / f"{model_name}_roc_curve.png"
            figure.tight_layout()
            figure.savefig(roc_path, dpi=140)
            plt.close(figure)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            candidate_path = output_path / f"{model_name}_pipeline.joblib"
            joblib.dump(search.best_estimator_, candidate_path)
            mlflow.log_artifact(str(candidate_path), artifact_path="pipeline")
            candidate_artifacts[model_name] = {
                "confusion_matrix": confusion_path,
                "roc_curve": roc_path,
                "tuning": tuning_path,
                "pipeline": candidate_path,
                "metrics": model_metrics_path,
            }
            results.append({"model": model_name, "best_params": search.best_params_, **metrics})

        candidate = (search.best_score_, model_name, search)
        if best_candidate is None or candidate[0] > best_candidate[0]:
            best_candidate = candidate

    assert best_candidate is not None
    best_score, best_name, best_search = best_candidate
    selected_metrics = next(result for result in results if result["model"] == best_name)
    best_path = output_path / "heart_model.joblib"
    joblib.dump(best_search.best_estimator_, best_path)
    pickle_path = output_path / "heart_model.pkl"
    with pickle_path.open("wb") as model_file:
        pickle.dump(best_search.best_estimator_, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "selected_model": best_name,
        "selection_metric": "mean 5-fold training ROC-AUC",
        "selected_cv_roc_auc_mean": float(best_score),
        "rows": int(len(dataset)),
        "features": FEATURES,
        "target_positive_definition": "original target num > 0",
        "models": results,
    }
    metrics_path = reports_path / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with mlflow.start_run(run_name="selected_model"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("selection_metric", summary["selection_metric"])
        mlflow.log_params(best_search.best_params_)
        mlflow.log_metrics(
            {
                f"selected_{key}": float(value)
                for key, value in selected_metrics.items()
                if key not in {"model", "best_params"}
            }
        )
        mlflow.log_metric("selected_cv_roc_auc_mean", float(best_score))
        mlflow.log_dict(summary, artifact_file="metrics/metrics.json")
        mlflow.log_dict(training_metadata, artifact_file="metadata/training_metadata.json")
        mlflow.log_artifact("requirements.txt", artifact_path="environment")
        mlflow.log_artifact(str(best_path), artifact_path="model")
        mlflow.log_artifact(str(pickle_path), artifact_path="model")
        mlflow.log_artifact(str(cleaned_path), artifact_path="data")
        mlflow.log_artifacts(str(eda_path), artifact_path="eda")
        mlflow.log_artifact(
            str(candidate_artifacts[best_name]["confusion_matrix"]),
            artifact_path="plots",
        )
        mlflow.log_artifact(str(candidate_artifacts[best_name]["roc_curve"]), artifact_path="plots")
        mlflow.log_artifact(str(candidate_artifacts[best_name]["tuning"]), artifact_path="tuning")
        mlflow.log_artifact(str(candidate_artifacts[best_name]["metrics"]), artifact_path="metrics")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/heart.csv", help="Path to the UCI CSV")
    parser.add_argument("--output", default="models", help="Directory for the serialized pipeline")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI")
    args = parser.parse_args()
    train_and_log(args.data, args.output, args.tracking_uri)


if __name__ == "__main__":
    main()
