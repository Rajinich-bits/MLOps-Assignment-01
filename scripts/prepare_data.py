"""Write a normalized, binary-target CSV while retaining missing feature values."""

import argparse

from heart_ml.data import save_cleaned_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/heart.csv", help="Path to raw UCI CSV")
    parser.add_argument(
        "--output",
        default="data/processed/heart_cleaned.csv",
        help="Path for cleaned CSV",
    )
    args = parser.parse_args()
    destination = save_cleaned_dataset(args.input, args.output)
    print(f"Saved cleaned dataset to {destination}")


if __name__ == "__main__":
    main()
