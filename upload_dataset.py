"""
Upload the balanced dataset to Google Cloud Storage.

@author: Abhinav Raghavendra
@year: 2025
"""

import subprocess
import os

def upload_dataset():
    # Define source and destination paths
    source_dir = "balanced_dataset"
    bucket_name = "fetus-ultrasound-balanced-with-metadata"
    destination = f"gs://{bucket_name}/balanced_dataset"
    
    # Upload the dataset
    upload_cmd = f"gsutil -m cp -r {source_dir}/* {destination}/"
    print(f"Uploading balanced dataset to {destination}")
    subprocess.run(upload_cmd, shell=True, check=True)
    
    print("\nDataset upload complete!")
    print(f"Dataset available at: {destination}")

if __name__ == "__main__":
    upload_dataset() 