import pandas as pd
import pytest

from heart_ml.data import COLUMNS, FEATURES, TARGET, load_dataset, save_cleaned_dataset


def test_load_headerless_uci_data_maps_target_and_missing_values(tmp_path):
    row = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, "?", 3, 2]
    path = tmp_path / "heart.csv"
    pd.DataFrame([row], columns=COLUMNS).to_csv(path, header=False, index=False)

    data = load_dataset(path)

    assert list(data.columns) == COLUMNS
    assert data.loc[0, TARGET] == 1
    assert pd.isna(data.loc[0, "ca"])


def test_load_named_data_binarizes_target(tmp_path):
    row = dict.fromkeys(FEATURES, 1)
    frame = pd.DataFrame([{**row, "target": 0}, {**row, "target": 4}])
    path = tmp_path / "named.csv"
    frame.to_csv(path, index=False)

    assert load_dataset(path)[TARGET].tolist() == [0, 1]


def test_invalid_named_data_reports_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"age": [50], "num": [0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_save_cleaned_dataset_maps_target_and_keeps_missing_values(tmp_path):
    source = tmp_path / "uci.csv"
    row = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, "?", 3, 2]
    pd.DataFrame([row]).to_csv(source, header=False, index=False)

    destination = save_cleaned_dataset(source, tmp_path / "processed" / "clean.csv")
    cleaned = pd.read_csv(destination)

    assert list(cleaned.columns) == COLUMNS
    assert cleaned.loc[0, TARGET] == 1
    assert pd.isna(cleaned.loc[0, "ca"])
