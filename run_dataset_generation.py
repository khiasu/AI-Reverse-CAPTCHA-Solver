"""
Simple script to run CAPTCHA dataset generation
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.generate_captcha_dataset import CaptchaDatasetGenerator

def main():
    """
    Generate CAPTCHA dataset with default settings
    """
    print("=== AI Reverse CAPTCHA Solver - Dataset Generation ===\n")
    
    # Create generator with default settings
    generator = CaptchaDatasetGenerator(
        output_dir="dataset/images",
        csv_file="dataset/labels.csv"
    )
    
    # Generate 5000 samples
    generator.generate_dataset(num_samples=5000)
    
    # Verify the dataset
    generator.verify_dataset(sample_size=10)
    
    print("\n=== Dataset generation complete! ===")
    print("Next steps:")
    print("1. Check the 'dataset/images' folder for generated CAPTCHA images")
    print("2. Check 'dataset/labels.csv' for the image-label mappings")
    print("3. Ready for model training phase!")

if __name__ == "__main__":
    main()
