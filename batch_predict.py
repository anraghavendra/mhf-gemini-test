"""
Run batch prediction on the fine-tuned model.

@author: Abhinav Raghavendra
@year: 2025
"""

import subprocess
from google.cloud import aiplatform

def upload_to_gcs(local_file, gcs_path):
    """Upload a file to Google Cloud Storage."""
    command = ['gsutil', 'cp', local_file, gcs_path]
    try:
        subprocess.run(command, check=True)
        print(f"Uploaded {local_file} to {gcs_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading {local_file}: {e}")
        raise

def run_batch_prediction(project, location, model_id, gcs_source, gcs_destination):
    """Runs a batch prediction job on Vertex AI."""
    print(f"Using model: {model_id}")
    aiplatform.init(project=project, location=location)
    model = aiplatform.Model(model_id)
    print("Starting batch prediction job...")
    batch_prediction_job = model.batch_predict(
        job_display_name="ultrasound-batch-eval",
        gcs_source=[gcs_source],
        gcs_destination_prefix=gcs_destination,
        instances_format="jsonl",
        predictions_format="jsonl",
        sync=True
    )
    print("Batch prediction job complete. Output at:", batch_prediction_job.output_info.gcs_output_directory)
    return batch_prediction_job.output_info.gcs_output_directory

def main():
    # First, convert the test dataset to batch prediction format
    print("Converting test dataset to batch prediction format...")
    subprocess.run(['python', 'convert_batch_format.py'], check=True)
    
    # Upload the converted files to GCS
    bucket = "gs://fetus-ultrasound-with-metadata"
    upload_to_gcs("batch_prediction_input.jsonl", f"{bucket}/batch_prediction_input.jsonl")
    upload_to_gcs("ground_truth.jsonl", f"{bucket}/ground_truth.jsonl")
    
    # Run batch prediction
    project = "mhf-test"
    location = "us-central1"
    model_id = "projects/263165751323/locations/us-central1/models/7487026572805799936@1"
    gcs_source = f"{bucket}/batch_prediction_input.jsonl"
    gcs_output_prefix = f"{bucket}/batch_predictions/"

    run_batch_prediction(project, location, model_id, gcs_source, gcs_output_prefix)

if __name__ == "__main__":
    main() 