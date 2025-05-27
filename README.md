# Fetus Ultrasound Classification with Gemini

This repository contains code for processing fetal ultrasound images and fine-tuning a Gemini model for classification.

## Project Structure

### Data Processing
- `download_dataset.py` - Downloads the dataset from Kaggle
- `match_metadata.py` - Matches ultrasound images with their metadata
- `generate_overlays.py` - Generates overlays for the ultrasound images
- `partition_dataset.py` - Splits the dataset into train/val/test sets
- `generate_jsonl.py` - Creates JSONL files for model training
- `upload_jsonl.py` - Uploads JSONL files to Google Cloud Storage

### Model Training
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

4. Set up Kaggle credentials:
- Download your `kaggle.json` file from Kaggle
- Place it in the project root directory

## Usage

Follow these steps in order:

1. Download the dataset:
```bash
python download_dataset.py
```

2. Match metadata with images:
```bash
python match_metadata.py
```

3. Generate overlays:
```bash
python generate_overlays.py
```

4. Partition the dataset:
```bash
python partition_dataset.py
```

5. Generate JSONL files:
```bash
python generate_jsonl.py
```

6. Upload JSONL files to GCS:
```bash
python upload_jsonl.py
```

7. Fine-tune the model:
```bash
python fine_tune_model.py
```

## Data

The processed dataset is stored in Google Cloud Storage:
- Training data: `gs://fetus-ultrasound-with-metadata/train_dataset.jsonl`
- Validation data: `gs://fetus-ultrasound-with-metadata/val_dataset.jsonl`
- Test data: `gs://fetus-ultrasound-with-metadata/test_dataset.jsonl`

## Model

The project uses Gemini 2.0 Flash as the base model for fine-tuning. The fine-tuned model will be available through Vertex AI after training completes.

## Requirements

- Python 3.8+
- Google Cloud SDK
- Kaggle API
- Required Python packages (see requirements.txt) 