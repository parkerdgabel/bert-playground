#!/usr/bin/env python3
"""
Download Spaceship Titanic dataset from Kaggle.

Make sure you have:
1. Kaggle API credentials in ~/.kaggle/kaggle.json
2. Accepted the competition rules at https://www.kaggle.com/c/spaceship-titanic/rules
"""

import os
import sys
from pathlib import Path
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_spaceship_titanic():
    """Download the Spaceship Titanic dataset."""
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create data directory
    data_dir = Path("data/spaceship-titanic")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download competition files
        print("Downloading Spaceship Titanic dataset...")
        api.competition_download_files(
            'spaceship-titanic',
            path=str(data_dir)
        )
        
        print("Download complete!")
        
        # Extract zip files
        import zipfile
        for zip_file in data_dir.glob("*.zip"):
            print(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_file.unlink()  # Remove zip after extraction
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in sorted(data_dir.glob("*.csv")):
            print(f"  - {file.name}")
            
        return True
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        if "403" in str(e):
            print("\nPlease make sure you have:")
            print("1. Valid Kaggle API credentials in ~/.kaggle/kaggle.json")
            print("2. Accepted the competition rules at:")
            print("   https://www.kaggle.com/c/spaceship-titanic/rules")
        return False


if __name__ == "__main__":
    success = download_spaceship_titanic()
    sys.exit(0 if success else 1)