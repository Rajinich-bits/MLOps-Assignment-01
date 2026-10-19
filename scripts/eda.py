"""Create exploratory plots and print basic dataset summaries."""

from heart_ml.data import TARGET, load_dataset
from heart_ml.eda import create_eda_figures


def main() -> None:
    data = load_dataset("data/raw/heart.csv")
    output = "reports/figures"
    print(f"Rows: {len(data)}")
    print(f"Class counts:\n{data[TARGET].value_counts().sort_index()}")
    print(f"Missing values:\n{data.isna().sum()}")
    create_eda_figures(data, output)
    print(f"Saved EDA figures under {output}/")


if __name__ == "__main__":
    main()
