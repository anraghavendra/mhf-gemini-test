"""
Upload the balanced JSONL files to Google Cloud Storage.

@author: Abhinav Raghavendra
@year: 2025
"""

import subprocess
import os

def upload_jsonl():
    # Define source and destination paths
    source_dir = "jsonl"
    bucket_name = "fetus-ultrasound-balanced-with-metadata"
    destination = f"gs://{bucket_name}/jsonl"
    
    # Upload the JSONL files
    upload_cmd = f"gsutil -m cp {source_dir}/balanced_*.jsonl {destination}/"
    print(f"Uploading balanced JSONL files to {destination}")
    subprocess.run(upload_cmd, shell=True, check=True)
    
    print("\nJSONL files upload complete!")
    print(f"Files available at: {destination}")

if __name__ == "__main__":
    upload_jsonl() 