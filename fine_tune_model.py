"""
Fine-tune the Gemini model on the fetal ultrasound dataset.

@author: Abhinav Raghavendra
@year: 2025
"""

import time
import vertexai
from vertexai.tuning import sft

# Initialize Vertex AI with your project
PROJECT_ID = "mhf-test"
vertexai.init(project=PROJECT_ID, location="us-central1")

# Create fine-tuning job
sft_tuning_job = sft.train(
    source_model="gemini-2.0-flash-001",
    train_dataset="gs://fetus-ultrasound-with-metadata/train_dataset.jsonl",
    validation_dataset="gs://fetus-ultrasound-with-metadata/val_dataset.jsonl",
)

# Polling for job completion
print("Starting fine-tuning job...")
print(f"Job name: {sft_tuning_job.name}")  # Print the job name
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()
    print("Job still running...")

print("\nFine-tuning completed!")
print(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
print(f"Tuned model endpoint: {sft_tuning_job.tuned_model_endpoint_name}")
print(f"Experiment: {sft_tuning_job.experiment}") 