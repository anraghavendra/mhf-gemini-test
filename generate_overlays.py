import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil

def create_ellipse_overlay(image, ellipse_params):
    """Create an overlay with the ellipse drawn on the original image."""
    overlay = image.copy()
    
    # Draw the ellipse
    cv2.ellipse(overlay,
                (int(ellipse_params['center_x']), int(ellipse_params['center_y'])),
                (int(ellipse_params['axis_x']), int(ellipse_params['axis_y'])),
                ellipse_params['angle'],
                0, 360, (0, 255, 0), 2)  # Green color, 2px thickness
    
    # Blend the overlay with the original image
    result = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    return result

def process_annotations(annotation_dir, output_base_path):
    """Process annotation images and generate overlays."""
    # Create output directory
    output_dir = "overlayed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters and data collection
    total_processed = 0
    category_counts = {}
    processed_data = []
    
    # Read the original matched_data.csv to preserve all metadata
    original_metadata = pd.read_csv(Path(annotation_dir) / 'matched_data.csv')
    metadata_dict = dict(zip(original_metadata['image_filename'], original_metadata.to_dict('records')))
    
    # Process each category
    for category in os.listdir(annotation_dir):
        category_path = Path(annotation_dir) / category
        if not category_path.is_dir():
            continue
            
        # Create category output directory
        category_output = Path(output_dir) / category
        os.makedirs(category_output, exist_ok=True)
        
        print(f"\nProcessing {category} category...")
        category_processed = 0
        
        # Process each annotation image
        for fname in os.listdir(category_path):
            if not fname.endswith("_Annotation.png"):
                continue
                
            # Read annotation image
            img_path = category_path / fname
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            
            # Threshold to binary
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                print(f"No contours found in: {fname}")
                continue
            
            # Fit ellipse to the largest contour
            cnt = max(contours, key=cv2.contourArea)
            
            if len(cnt) < 5:
                print(f"Not enough points to fit ellipse in: {fname}")
                continue
            
            # Fit ellipse
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (axis_x, axis_y), angle = ellipse
            
            # Get the original image
            original_img = fname.replace("_Annotation.png", ".png")
            original_path = category_path / original_img
            
            if not original_path.exists():
                print(f"Original image not found: {original_path}")
                continue
            
            # Read original image
            original = cv2.imread(str(original_path))
            if original is None:
                print(f"Could not read original image: {original_path}")
                continue
            
            # Create overlay
            ellipse_params = {
                'center_x': center_x,
                'center_y': center_y,
                'axis_x': axis_x/2,  # OpenCV returns full length, divide by 2 for radius
                'axis_y': axis_y/2,
                'angle': angle
            }
            
            overlay = create_ellipse_overlay(original, ellipse_params)
            
            # Save overlay
            overlay_filename = f"overlay_{original_img}"
            overlay_path = category_output / overlay_filename
            cv2.imwrite(str(overlay_path), overlay)
            
            # Get original metadata for this image
            original_metadata = metadata_dict.get(original_img, {})
            
            # Add data for CSV, preserving all original metadata
            data_entry = {
                'image_number': int(original_img.split('_')[0]),
                'image_filename': overlay_filename,
                'category': category,
                'fetal_health': 1.0 if category == 'normal' else (2.0 if category == 'benign' else 3.0),
                'ellipse_center_x': center_x,
                'ellipse_center_y': center_y,
                'ellipse_axis_x': axis_x/2,
                'ellipse_axis_y': axis_y/2,
                'ellipse_angle': angle,
                'has_annotation': True
            }
            
            # Add all original metadata fields
            for key, value in original_metadata.items():
                if key not in data_entry:
                    data_entry[key] = value
            
            processed_data.append(data_entry)
            
            category_processed += 1
            total_processed += 1
            print(f"Processed: {fname}")
        
        category_counts[category] = category_processed
        print(f"Completed {category}: {category_processed} images processed")
    
    # Create and save the updated CSV
    df = pd.DataFrame(processed_data)
    df = df.sort_values('image_number')
    df.to_csv(Path(output_dir) / 'matched_data.csv', index=False)
    
    return total_processed, category_counts

def main():
    # Base paths
    annotation_dir = "matched_dataset"
    output_base = "overlayed_dataset"
    
    if not os.path.exists(annotation_dir):
        print(f"Error: Annotation directory not found at {annotation_dir}")
        return
    
    print("Starting overlay generation...")
    total_processed, category_counts = process_annotations(annotation_dir, output_base)
    
    print("\nOverlay generation complete!")
    print(f"Total images processed: {total_processed}")
    print("\nImages processed by category:")
    for category, count in category_counts.items():
        print(f"{category}: {count} images")
    print(f"\nOverlays have been saved to: {output_base}")

if __name__ == "__main__":
    main() 