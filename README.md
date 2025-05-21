# Fetus Ultrasound Classification with Gemini

This repository contains code for processing fetal ultrasound images and fine-tuning a Gemini model for classification.

## Project Structure

### Data Processing
The `data_processing/` directory contains scripts for dataset preparation:
- `create_gcs_bucket.py` - Creates and configures the Google Cloud Storage bucket
- `create_medical_jsonl.py` - Creates JSONL files for model training
- `delete_old_folders.py` - Utility for cleaning up old data directories
- `download_dataset.py` - Downloads the dataset from Kaggle
- `extract_folders.py` - Extracts and organizes dataset folders
- `remove_annotated_images.py` - Removes annotated images from the dataset
- `upload_jsonl_to_gcs.py` - Uploads JSONL files to Google Cloud Storage
- `upload_to_gcs.py` - Uploads images to Google Cloud Storage

### Model Training
The `model_training/` directory contains scripts for model fine-tuning:
- `fine_tune_model.py` - Fine-tunes the Gemini model using Vertex AI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud credentials:
```bash
gcloud auth login
gcloud auth application-default login
```

3. Configure your Google Cloud project:
```bash
gcloud config set project mhf-test
```

## Usage

1. Process the dataset:
```bash
cd data_processing
python create_gcs_bucket.py
python download_dataset.py
python extract_folders.py
python remove_annotated_images.py
python create_medical_jsonl.py
python upload_jsonl_to_gcs.py
```

2. Fine-tune the model:
```bash
cd model_training
python fine_tune_model.py
```

## Data

The processed dataset is stored in Google Cloud Storage:
- Training data: `gs://fetus-ultrasound-data/train-clean.jsonl`
- Validation data: `gs://fetus-ultrasound-data/validation-clean.jsonl` 