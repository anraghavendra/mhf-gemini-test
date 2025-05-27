"""
Create a balanced dataset by selecting the highest quality images from normal and benign categories
to match the malignant category (smallest), while keeping all malignant images.

@author: Abhinav Raghavendra
@year: 2025
"""

import os
import shutil
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def calculate_image_quality(image_path):
    """
    Calculate image quality score based on actual image characteristics.
    Higher score indicates better quality.
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate quality metrics
    metrics = {}
    
    # 1. Resolution score (normalized by typical ultrasound resolution)
    height, width = gray.shape
    resolution_score = min((width * height) / (800 * 600), 1.0)  # Normalize to typical ultrasound resolution
    metrics['resolution'] = resolution_score
    
    # 2. Sharpness score using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize to typical sharpness range
    metrics['sharpness'] = sharpness_score
    
    # 3. Contrast score using standard deviation
    contrast_score = min(np.std(gray) / 100, 1.0)  # Normalize to typical contrast range
    metrics['contrast'] = contrast_score
    
    # 4. Noise score using median filter
    median = cv2.medianBlur(gray, 3)
    noise = cv2.absdiff(gray, median)
    noise_score = 1.0 - min(np.mean(noise) / 50, 1.0)  # Lower noise is better
    metrics['noise'] = noise_score
    
    # Calculate final quality score with weights
    weights = {
        'resolution': 0.3,
        'sharpness': 0.3,
        'contrast': 0.2,
        'noise': 0.2
    }
    
    final_score = sum(metrics[metric] * weight for metric, weight in weights.items())
    return final_score

def create_balanced_dataset(source_dir, target_dir):
    # Create target directory structure
    for split in ['train', 'val', 'test']:
        for category in ['normal', 'benign', 'malignant']:
            target_path = Path(target_dir) / split / category
            target_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Get all images for each category
        categories = {}
        for category in ['normal', 'benign', 'malignant']:
            source_path = Path(source_dir) / split / category
            if source_path.exists():
                images = list(source_path.glob('*.png'))
                categories[category] = images
                print(f"{category}: {len(images)} images")
        
        # Find the smallest category (malignant)
        category_counts = {cat: len(imgs) for cat, imgs in categories.items()}
        smallest_count = min(category_counts.values())
        print(f"Reducing normal and benign categories to match malignant category: {smallest_count} images")
        
        # Process each category
        for category, images in categories.items():
            if category in ['normal', 'benign']:
                # For normal and benign categories, select highest quality images up to smallest_count
                image_scores = [(img, calculate_image_quality(img)) for img in images]
                image_scores.sort(key=lambda x: x[1], reverse=True)
                selected_images = [img for img, _ in image_scores[:smallest_count]]
                print(f"Selected {len(selected_images)} highest quality images from {category} category")
            else:
                # For malignant category, keep all images
                selected_images = images
                print(f"Keeping all {len(selected_images)} images from {category} category")
            
            # Copy selected images
            for img in selected_images:
                target_path = Path(target_dir) / split / category / img.name
                shutil.copy2(img, target_path)

def main():
    source_dir = "partitioned_dataset"
    target_dir = "balanced_dataset"
    
    # Remove existing balanced dataset if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # Create balanced dataset
    create_balanced_dataset(source_dir, target_dir)
    
    print("\nBalanced dataset creation complete!")

if __name__ == "__main__":
    main() 