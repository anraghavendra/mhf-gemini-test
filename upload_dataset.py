"""
Upload the partitioned dataset to Google Cloud Storage.

@author: Abhinav Raghavendra
@year: 2025
"""

import subprocess
import os
from pathlib import Path

def upload_to_gcs(local_dir, bucket_name):
    """
    Upload a local directory to Google Cloud Storage bucket.
    
    Args:
        local_dir: Path to the local directory to upload
        bucket_name: Name of the GCS bucket
    """
    # Construct the gsutil command for recursive upload
    command = ['gsutil', '-m', 'cp', '-r', local_dir, f'gs://{bucket_name}/']
    
    try:
        print(f"Starting upload of {local_dir} to gs://{bucket_name}/...")
        subprocess.run(command, check=True)
        print(f"Upload completed successfully!")
        
        # List the contents of the bucket to verify
        print("\nVerifying upload by listing bucket contents:")
        list_command = ['gsutil', 'ls', '-r', f'gs://{bucket_name}/']
        subprocess.run(list_command, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error during upload: {e}")

def main():
    # Define the bucket name and local directory
    bucket_name = 'fetus-ultrasound-with-metadata'
    local_dir = 'partitioned_dataset'
    
    # Check if the local directory exists
    if not os.path.exists(local_dir):
        print(f"Error: Local directory '{local_dir}' not found!")
        return
    
    # Upload the dataset
    upload_to_gcs(local_dir, bucket_name)

if __name__ == "__main__":
    main() 