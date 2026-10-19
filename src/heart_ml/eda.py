"""Generate exploratory plots for the cleaned heart disease dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from heart_ml.data import FEATURES, TARGET
from heart_ml.pipeline import RAW_NUMERIC_FEATURES


def create_eda_figures(data, output_dir: str | Path) -> list[Path]:
    """Create class balance, feature distribution, and correlation figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x=TARGET, ax=axis)
    axis.set(title="Heart disease target class balance", xlabel="Target (0=no disease, 1=disease)")
    figure.tight_layout()
    class_balance_path = output_path / "class_balance.png"
    figure.savefig(class_balance_path, dpi=140)
    plt.close(figure)

    data[FEATURES].hist(figsize=(14, 11), bins=16, edgecolor="white")
    plt.suptitle("UCI Cleveland feature distributions")
    plt.tight_layout()
    histograms_path = output_path / "feature_histograms.png"
    plt.savefig(histograms_path, dpi=140)
    plt.close("all")

    figure, axis = plt.subplots(figsize=(12, 9))
    correlation_features = [*RAW_NUMERIC_FEATURES, TARGET]
    sns.heatmap(
        data[correlation_features].corr(),
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        ax=axis,
    )
    axis.set_title("Numeric feature and target correlation (exploratory, not causal)")
    figure.tight_layout()
    correlation_path = output_path / "correlation_heatmap.png"
    figure.savefig(correlation_path, dpi=140)
    plt.close(figure)

    return [class_balance_path, histograms_path, correlation_path]
