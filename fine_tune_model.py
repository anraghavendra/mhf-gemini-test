"""
Fine-tune the Gemini model on the balanced fetal ultrasound dataset.

@author: Abhinav Raghavendra
@year: 2025
"""

import time
import vertexai
from vertexai.tuning import sft

# Initialize Vertex AI with your project
PROJECT_ID = "mhf-test"
vertexai.init(project=PROJECT_ID, location="us-central1")

# Define dataset paths for balanced dataset
TRAIN_DATASET = "gs://fetus-ultrasound-balanced-with-metadata/jsonl/balanced_train_dataset.jsonl"
VAL_DATASET = "gs://fetus-ultrasound-balanced-with-metadata/jsonl/balanced_val_dataset.jsonl"

print("Starting fine-tuning job with balanced dataset...")
print(f"Training dataset: {TRAIN_DATASET}")
print(f"Validation dataset: {VAL_DATASET}")

# Create fine-tuning job
sft_tuning_job = sft.train(
    source_model="gemini-2.0-flash-001",
    train_dataset=TRAIN_DATASET,
    validation_dataset=VAL_DATASET,
)

# Polling for job completion
print("\nFine-tuning job started successfully!")
print(f"Job name: {sft_tuning_job.name}")

while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()
    print("Job still running...")

print("\nFine-tuning completed successfully!")
print(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
print(f"Tuned model endpoint: {sft_tuning_job.tuned_model_endpoint_name}")
print(f"Experiment: {sft_tuning_job.experiment}") 