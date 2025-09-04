"""
CAPTCHA Dataset Generator
Generates synthetic CAPTCHA images for training AI models.
"""

import os
import random
import string
import csv
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import argparse


class CaptchaDatasetGenerator:
    """
    Generates CAPTCHA dataset with various distortions and noise patterns.
    """
    
    def __init__(self, output_dir="dataset/images", csv_file="dataset/labels.csv"):
        """
        Initialize the CAPTCHA generator.
        
        Args:
            output_dir (str): Directory to save generated images
            csv_file (str): Path to CSV file for labels
        """
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.characters = string.ascii_uppercase + string.digits
        self.captcha_length = 5
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        
        # Initialize CAPTCHA generator with varied fonts
        self.image_captcha = ImageCaptcha(
            width=200, 
            height=80,
            fonts=[
                # Will use default fonts if custom fonts not available
            ]
        )
    
    def generate_random_text(self):
        """
        Generate random alphanumeric text for CAPTCHA.
        
        Returns:
            str: Random 5-character string
        """
        return ''.join(random.choices(self.characters, k=self.captcha_length))
    
    def add_noise(self, image):
        """
        Add random noise to the image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Image with added noise
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add random noise
        noise_factor = random.uniform(0.1, 0.3)
        noise = np.random.randint(0, int(255 * noise_factor), img_array.shape, dtype=np.uint8)
        
        # Apply noise
        noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    def add_distortions(self, image):
        """
        Apply random distortions to the image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Distorted image
        """
        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, fillcolor='white')
        
        # Random blur
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Random brightness adjustment
        if random.random() < 0.4:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # Random contrast adjustment
        if random.random() < 0.4:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        return image
    
    def generate_single_captcha(self, text, filename):
        """
        Generate a single CAPTCHA image with distortions.
        
        Args:
            text (str): Text to generate CAPTCHA for
            filename (str): Output filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate base CAPTCHA image
            image_data = self.image_captcha.generate(text)
            image = Image.open(image_data)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply random distortions and noise
            image = self.add_distortions(image)
            image = self.add_noise(image)
            
            # Save the image
            filepath = os.path.join(self.output_dir, filename)
            image.save(filepath, 'PNG')
            
            return True
            
        except Exception as e:
            print(f"Error generating CAPTCHA for '{text}': {e}")
            return False
    
    def generate_dataset(self, num_samples=5000, batch_size=100):
        """
        Generate the complete CAPTCHA dataset.
        
        Args:
            num_samples (int): Number of samples to generate
            batch_size (int): Batch size for progress reporting
        """
        print(f"Generating {num_samples} CAPTCHA samples...")
        print(f"Output directory: {self.output_dir}")
        print(f"Labels file: {self.csv_file}")
        
        # Prepare CSV file
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])  # Header
            
            successful_generations = 0
            
            for i in range(num_samples):
                # Generate random text
                text = self.generate_random_text()
                filename = f"captcha_{i:06d}.png"
                
                # Generate CAPTCHA image
                if self.generate_single_captcha(text, filename):
                    # Write to CSV
                    writer.writerow([filename, text])
                    successful_generations += 1
                
                # Progress reporting
                if (i + 1) % batch_size == 0:
                    print(f"Generated {i + 1}/{num_samples} samples "
                          f"({successful_generations} successful)")
        
        print(f"\nDataset generation complete!")
        print(f"Successfully generated: {successful_generations}/{num_samples} samples")
        print(f"Success rate: {(successful_generations/num_samples)*100:.1f}%")
    
    def verify_dataset(self, sample_size=10):
        """
        Verify the generated dataset by checking a few samples.
        
        Args:
            sample_size (int): Number of samples to verify
        """
        print(f"\nVerifying dataset (checking {sample_size} samples)...")
        
        try:
            # Read CSV file
            with open(self.csv_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
            
            if len(rows) == 0:
                print("No data found in CSV file!")
                return
            
            # Check random samples
            sample_rows = random.sample(rows, min(sample_size, len(rows)))
            
            for row in sample_rows:
                filename = row['filename']
                label = row['label']
                filepath = os.path.join(self.output_dir, filename)
                
                if os.path.exists(filepath):
                    print(f"✓ {filename} -> {label}")
                else:
                    print(f"✗ {filename} -> {label} (file not found)")
            
            print(f"\nTotal samples in dataset: {len(rows)}")
            
        except Exception as e:
            print(f"Error verifying dataset: {e}")


def main():
    """
    Main function to run the dataset generator.
    """
    parser = argparse.ArgumentParser(description='Generate CAPTCHA dataset')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of samples to generate (default: 5000)')
    parser.add_argument('--output-dir', type=str, default='dataset/images',
                       help='Output directory for images')
    parser.add_argument('--csv-file', type=str, default='dataset/labels.csv',
                       help='CSV file for labels')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset after generation')
    
    args = parser.parse_args()
    
    # Create generator
    generator = CaptchaDatasetGenerator(
        output_dir=args.output_dir,
        csv_file=args.csv_file
    )
    
    # Generate dataset
    generator.generate_dataset(num_samples=args.samples)
    
    # Verify if requested
    if args.verify:
        generator.verify_dataset()


if __name__ == "__main__":
    main()
