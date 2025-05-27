"""
Convert test dataset to batch prediction format.

@author: Abhinav Raghavendra
@year: 2025
"""

import json
import pandas as pd

def create_dynamic_prompt(metadata):
    """Create a dynamic prompt based on the image's metadata and ellipse parameters."""
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

    # Add ellipse parameters if available
    if all(key in metadata for key in ['ellipse_center_x', 'ellipse_center_y', 'ellipse_axis_x', 'ellipse_axis_y', 'ellipse_angle']):
        ellipse_info = f"""
Brain Region Measurements:
The green ellipse surrounding the brain region has the following characteristics:
- Center: ({metadata['ellipse_center_x']}, {metadata['ellipse_center_y']}) pixels
- Major and Minor Axes: ({metadata['ellipse_axis_x']}, {metadata['ellipse_axis_y']}) pixels
- Rotation Angle: {metadata['ellipse_angle']} degrees

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

    # Add metadata if available
    metadata_fields = [
        'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions',
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
        'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
        'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
        'histogram_tendency'
    ]
    
    if all(field in metadata for field in metadata_fields):
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
    return base_prompt

def convert_to_batch_format(input_file, output_file, ground_truth_file):
    """Convert the test dataset to batch prediction format."""
    # Read the matched data CSV for ground truth and metadata
    matched_data = pd.read_csv('partitioned_dataset/matched_data.csv')
    matched_data = matched_data[matched_data['image_filename'].str.startswith('test/')]
    
    # Create a mapping of image filenames to their categories and metadata
    ground_truth_map = dict(zip(matched_data['image_filename'], matched_data['category']))
    metadata_map = dict(zip(matched_data['image_filename'], matched_data.to_dict('records')))
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out, open(ground_truth_file, 'w') as f_truth:
        for line in f_in:
            data = json.loads(line)
            # Extract the user message which contains the image
            user_message = data['contents'][0]
            
            # Get the image filename from the fileUri
            image_path = user_message['parts'][0]['fileData']['fileUri'].split('/')[-1]
            
            # Get metadata for this image
            metadata = metadata_map.get(image_path, {})
            
            # Create dynamic prompt with this image's metadata
            dynamic_prompt = create_dynamic_prompt(metadata)
            
            # Create the batch prediction format with the dynamic prompt
            batch_format = {
                "request": {
                    "contents": [{
                        "role": "user",
                        "parts": [
                            user_message['parts'][0],  # Image
                            {"text": dynamic_prompt}  # Dynamic prompt with ellipse and metadata
                        ]
                    }]
                }
            }
            
            # Write to output file
            f_out.write(json.dumps(batch_format) + '\n')
            
            # Get ground truth from the matched data
            if image_path in ground_truth_map:
                ground_truth = ground_truth_map[image_path]
                f_truth.write(json.dumps({"ground_truth": ground_truth}) + '\n')

def main():
    input_file = "jsonl/test_dataset.jsonl"
    output_file = "batch_prediction_input.jsonl"
    ground_truth_file = "ground_truth.jsonl"
    
    print("Converting test dataset to batch prediction format...")
    convert_to_batch_format(input_file, output_file, ground_truth_file)
    print(f"Conversion complete. Output written to {output_file} and ground truth to {ground_truth_file}")

if __name__ == "__main__":
    main() 