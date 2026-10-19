import pandas as pd

from heart_ml.data import FEATURES, TARGET
from heart_ml.eda import create_eda_figures


def test_create_eda_figures_writes_expected_plots(tmp_path):
    data = pd.DataFrame(
        {
            **{feature: [1, 2, 3, 4] for feature in FEATURES},
            TARGET: [0, 1, 0, 1],
        }
    )

    figures = create_eda_figures(data, tmp_path)

    assert {figure.name for figure in figures} == {
        "class_balance.png",
        "feature_histograms.png",
        "correlation_heatmap.png",
    }
    assert all(figure.is_file() and figure.stat().st_size > 0 for figure in figures)
