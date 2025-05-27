"""
Upload JSONL files to Google Cloud Storage.

@author: Abhinav Raghavendra
@year: 2025
"""

import subprocess
import os

# Define the JSONL files and the GCS bucket
jsonl_files = [
    'jsonl/train_dataset.jsonl',
    'jsonl/val_dataset.jsonl',
    'jsonl/test_dataset.jsonl'
]
bucket_path = 'gs://fetus-ultrasound-with-metadata'

# Upload each JSONL file
for jsonl_file in jsonl_files:
    if not os.path.exists(jsonl_file):
        print(f"Warning: {jsonl_file} not found")
        continue
        
    # Construct the gsutil command to upload the JSONL file
    command = ['gsutil', 'cp', jsonl_file, f'{bucket_path}/']
    
    try:
        subprocess.run(command, check=True)
        print(f"Uploaded {jsonl_file} to {bucket_path}/ successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading {jsonl_file} to GCS: {e}") 