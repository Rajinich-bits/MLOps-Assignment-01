"""Download the processed Cleveland data file from the official UCI archive."""

import argparse
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def download(destination: str | Path, overwrite: bool = False) -> Path:
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = destination.parent / "heart-disease-uci.zip"
    try:
        request = urllib.request.Request(
            UCI_ARCHIVE_URL,
            headers={"User-Agent": "heart-disease-mlops-course-project/1.0"},
        )
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, context=context, timeout=30) as response:
            with temporary_zip.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file)
        with zipfile.ZipFile(temporary_zip) as archive:
            matches = [
                name for name in archive.namelist() if name.endswith("processed.cleveland.data")
            ]
            if not matches:
                raise FileNotFoundError(
                    "processed.cleveland.data was not present in the UCI archive."
                )
            with archive.open(matches[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    finally:
        temporary_zip.unlink(missing_ok=True)
    print(f"Downloaded UCI Cleveland data to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/heart.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download(args.output, args.overwrite)


if __name__ == "__main__":
    main()
