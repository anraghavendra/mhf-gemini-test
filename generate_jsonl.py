import json
import os
import pandas as pd
from pathlib import Path
import re

def natural_sort_key(s):
    # Extract numbers from the filename for sorting
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def create_jsonl_example(image_path, label, bucket_path, ellipse_params=None, metadata=None):
    # Base prompt
    base_prompt = """You are a diagnostic medical AI trained in fetal neuroimaging.

Analyze the provided fetal brain ultrasound image and classify it as one of the following:
– normal
– benign
– malignant

The brain region is outlined by a green ellipse overlay. The shape of this green ellipse is crucial as it reflects the overall shape and development of the brain. Focus your analysis strictly within this green elliptical boundary, examining both the brain structures and any potential abnormalities.

Base your classification on the following medically relevant criteria:

    Brain Region Analysis
    – Examine the entire brain region within the green ellipse for any abnormalities
    – Look for tumors, masses, or unusual growths
    – Check for any abnormal tissue patterns or densities
    – Assess the overall brain development and structure
    – Evaluate if the brain shape is appropriate for gestational age

    Symmetry and Morphology
    – Are the brain hemispheres symmetric?
    – Is the midline intact or shifted?
    – Are sulci, gyri, and ventricles normal in size and shape for gestational age?
    – Is the overall brain shape (as indicated by the green ellipse) normal and proportional?

    Lesions or Masses
    – Are there any focal lesions or space-occupying masses?
    – Is there evidence of calcification, cystic regions, or hemorrhage?
    – Look for any abnormal growths or tumors within the brain region

    Ventricular System
    – Is there ventriculomegaly, hydrocephalus, or other abnormal dilation?
    – Are choroid plexuses normal and symmetric?

    Tissue Integrity
    – Are there areas of hyperechogenicity or hypoechogenicity suggesting necrosis or inflammation?
    – Check for any abnormal tissue patterns or densities

    Mass Effect or Deformation
    – Is there compression or displacement of normal brain structures?
    – Are adjacent tissues affected by any lesion?
    – Has the brain shape been altered by any mass or growth?

    Gestational Appropriateness
    – Do all visible structures appear appropriate for the estimated gestational age?
    – Is the brain shape and size appropriate for the gestational age?"""

    # Add ellipse parameters to the prompt if available
    if ellipse_params is not None:
        ellipse_info = f"""
Brain Region Measurements:
The green ellipse surrounding the brain region has the following characteristics:
- Center: ({ellipse_params['ellipse_center_x']}, {ellipse_params['ellipse_center_y']}) pixels
- Major and Minor Axes: ({ellipse_params['ellipse_axis_x']}, {ellipse_params['ellipse_axis_y']}) pixels
- Rotation Angle: {ellipse_params['ellipse_angle']} degrees

Use these measurements to assess:
1. Brain size and shape relative to gestational age
2. Symmetry of brain structures
3. Proportional relationships between brain regions
4. Overall brain morphology and development
5. Whether the brain shape is normal and appropriate for development

Note: While the position and orientation of the ellipse are not relevant, the shape and measurements are crucial indicators of brain development. Focus on:
1. The shape and proportions of the brain region (as indicated by the green ellipse measurements)
2. Any abnormalities, tumors, or unusual growths within the brain region
3. Whether the brain shape and size are appropriate for normal development"""
        base_prompt += ellipse_info

    # Add metadata to the prompt if available
    if metadata is not None:
        metadata_info = f"""
Additional Metadata:
- Baseline Value: {metadata['baseline_value']} bpm
- Accelerations: {metadata['accelerations']} bpm
- Fetal Movement: {metadata['fetal_movement']} bpm
- Uterine Contractions: {metadata['uterine_contractions']} bpm
- Light Decelerations: {metadata['light_decelerations']} bpm
- Severe Decelerations: {metadata['severe_decelerations']} bpm
- Prolongued Decelerations: {metadata['prolongued_decelerations']} bpm
- Abnormal Short Term Variability: {metadata['abnormal_short_term_variability']} ms
- Mean Value of Short Term Variability: {metadata['mean_value_of_short_term_variability']} ms
- Percentage of Time with Abnormal Long Term Variability: {metadata['percentage_of_time_with_abnormal_long_term_variability']}%
- Mean Value of Long Term Variability: {metadata['mean_value_of_long_term_variability']} ms
- Histogram Width: {metadata['histogram_width']}
- Histogram Min: {metadata['histogram_min']}
- Histogram Max: {metadata['histogram_max']}
- Histogram Number of Peaks: {metadata['histogram_number_of_peaks']}
- Histogram Number of Zeroes: {metadata['histogram_number_of_zeroes']}
- Histogram Mode: {metadata['histogram_mode']}
- Histogram Mean: {metadata['histogram_mean']}
- Histogram Median: {metadata['histogram_median']}
- Histogram Variance: {metadata['histogram_variance']}
- Histogram Tendency: {metadata['histogram_tendency']}

Use this metadata to enhance your analysis by considering:
1. Fetal health metrics and their implications
2. Variability in fetal heart rate and its significance
3. Histogram data to understand the distribution of measurements
4. Any correlations between metadata and brain development"""
        base_prompt += metadata_info

    base_prompt += "\n\nReturn your analysis as one word, either: normal, benign, malignant"

    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": f"{bucket_path}/{image_path}"
                        }
                    },
                    {
                        "text": base_prompt
                    }
                ]
            },
            {
                "role": "model",
                "parts": [
                    {
                        "text": f"""{label}

Detailed Analysis:

1. Classification: {label}

2. Brain Region Analysis:
   • Overall brain structure and development
   • Presence of any abnormalities or unusual growths
   • Tissue patterns and densities

3. Symmetry and Morphology:
   • Brain hemisphere symmetry
   • Midline integrity
   • Ventricular system assessment

4. Lesions and Masses:
   • Presence of focal lesions
   • Evidence of calcification or hemorrhage
   • Space-occupying masses

5. Tissue Characteristics:
   • Echogenicity patterns
   • Tissue integrity
   • Any signs of inflammation or necrosis

6. Mass Effects:
   • Compression or displacement of structures
   • Impact on adjacent tissues
   • Overall brain shape alterations

7. Gestational Assessment:
   • Age-appropriate development
   • Brain size and shape appropriateness

8. Confidence Level: High/Medium/Low

9. Key Distinguishing Features:
   • [List specific features that led to classification]

10. Additional Observations:
    • [Any other relevant findings]"""
                    }
                ]
            }
        ]
    }

def generate_jsonl(folder_path, output_file, bucket_path, matched_data_csv, split):
    # Read metadata from matched_data.csv
    params_df = pd.read_csv(matched_data_csv)
    params_dict = dict(zip(params_df['image_filename'], params_df.to_dict('records')))

    jsonl_data = []
    split_path = Path(folder_path) / split
    if split_path.exists():
        for label in ['normal', 'benign', 'malignant']:
            label_path = split_path / label
            if label_path.exists():
                # Get all PNG files and sort them naturally
                image_files = sorted(label_path.glob('*.png'), key=natural_sort_key)
                for image_file in image_files:
                    image_path = f"{split}/{label}/{image_file.name}"
                    # Get metadata for this image if available
                    metadata = params_dict.get(image_path)
                    jsonl_data.append(create_jsonl_example(image_path, label, bucket_path, metadata, metadata))

    # Write JSONL data to a file
    with open(output_file, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')

    print(f"JSONL file created successfully at {output_file}.")
    print(f"Total examples: {len(jsonl_data)}")

def main():
    # Define paths
    folder_path = "partitioned_dataset"
    bucket_path = "gs://fetus-ultrasound-with-metadata/partitioned_dataset"
    matched_data_csv = "partitioned_dataset/matched_data.csv"
    
    # Process each split
    for split in ['train', 'val', 'test']:
        output_file = f"jsonl/{split}_dataset.jsonl"
        print(f"\nProcessing {split} split...")
        generate_jsonl(folder_path, output_file, bucket_path, matched_data_csv, split)

if __name__ == "__main__":
    main() 