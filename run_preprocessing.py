"""
Simple script to run CAPTCHA data preprocessing
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.preprocess_captcha_data import CaptchaPreprocessor

def main():
    """
    Run CAPTCHA data preprocessing with default settings
    """
    print("=== AI Reverse CAPTCHA Solver - Data Preprocessing ===\n")
    
    # Create preprocessor
    preprocessor = CaptchaPreprocessor(
        images_dir="dataset/images",
        labels_file="dataset/labels.csv",
        output_dir="dataset/preprocessed",
        image_width=100,
        image_height=40
    )
    
    # Preprocess dataset with integer encoding (more memory efficient)
    data = preprocessor.preprocess_dataset(
        encoding_type="integer",
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(data, filename="preprocessed_data.npz")
    
    # Also create one-hot encoded version for comparison
    print("\nCreating one-hot encoded version...")
    data_onehot = preprocessor.preprocess_dataset(
        encoding_type="onehot",
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    preprocessor.save_preprocessed_data(data_onehot, filename="preprocessed_data_onehot.npz")
    
    print("\n=== Preprocessing complete! ===")
    print("Generated files:")
    print("1. dataset/preprocessed/preprocessed_data.npz (integer encoded)")
    print("2. dataset/preprocessed/preprocessed_data_onehot.npz (one-hot encoded)")
    print("3. dataset/preprocessed/metadata.pkl (preprocessing metadata)")
    print("\nReady for model training phase!")

if __name__ == "__main__":
    main()
