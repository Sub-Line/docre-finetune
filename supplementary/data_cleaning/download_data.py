"""
Download and setup RE-DocRED dataset.
"""
import os
import json
import requests
from pathlib import Path
from typing import Dict, Any


def download_file(url: str, filepath: Path) -> bool:
    """Download file from URL to filepath."""
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def check_and_download_data(data_dir: str) -> bool:
    """Check if RE-DocRED data exists, download if not."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # RE-DocRED data URLs (correct filenames from the repository)
    files_to_download = {
        "train_annotated.json": "https://github.com/tonytan48/Re-DocRED/raw/main/data/train_revised.json",
        "dev.json": "https://github.com/tonytan48/Re-DocRED/raw/main/data/dev_revised.json",
        "test.json": "https://github.com/tonytan48/Re-DocRED/raw/main/data/test_revised.json"
    }

    missing_files = []
    for filename in files_to_download.keys():
        filepath = data_path / filename
        if not filepath.exists():
            missing_files.append(filename)

    if not missing_files:
        print("All RE-DocRED data files found.")
        return True

    print(f"Missing data files: {missing_files}")
    download_choice = input("Download RE-DocRED data? (y/n): ").strip().lower()

    if download_choice != 'y':
        print("Skipping data download. Please ensure data files are in the data directory.")
        return False

    # Download missing files
    success = True
    for filename in missing_files:
        url = files_to_download[filename]
        filepath = data_path / filename
        if not download_file(url, filepath):
            success = False

    return success


def load_relation_info(data_dir: str) -> Dict[str, Any]:
    """Load relation information from rel_info.json."""
    rel_info_path = Path(data_dir) / "rel_info.json"

    if not rel_info_path.exists():
        print(f"Warning: {rel_info_path} not found. Using default relation mapping.")
        return {}

    with open(rel_info_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test the download functionality
    test_data_dir = "./test_data"
    check_and_download_data(test_data_dir)