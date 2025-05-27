"""
Create a Google Cloud Storage bucket for the balanced dataset with metadata.

@author: Abhinav Raghavendra
@year: 2025
"""

import subprocess
import os

def create_bucket():
    # Define bucket name
    bucket_name = "fetus-ultrasound-balanced-with-metadata"
    
    # Create the bucket
    create_cmd = f"gsutil mb -l us-central1 gs://{bucket_name}"
    print(f"Creating bucket: {bucket_name}")
    subprocess.run(create_cmd, shell=True, check=True)
    
    # Set bucket permissions to public
    make_public_cmd = f"gsutil iam ch allUsers:objectViewer gs://{bucket_name}"
    print("Setting bucket permissions to public")
    subprocess.run(make_public_cmd, shell=True, check=True)
    
    print(f"\nBucket {bucket_name} created successfully!")
    print(f"Bucket URL: https://storage.googleapis.com/{bucket_name}")

if __name__ == "__main__":
    create_bucket() 