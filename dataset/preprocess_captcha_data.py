"""
CAPTCHA Data Preprocessing Pipeline
Preprocesses CAPTCHA images and labels for deep learning training.
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import argparse
from typing import Tuple, List, Dict
import pickle


class CaptchaPreprocessor:
    """
    Preprocesses CAPTCHA images and labels for machine learning training.
    """
    
    def __init__(self, 
                 images_dir: str = "dataset/images",
                 labels_file: str = "dataset/labels.csv",
                 output_dir: str = "dataset/preprocessed",
                 image_width: int = 100,
                 image_height: int = 40,
                 captcha_length: int = 5):
        """
        Initialize the preprocessor.
        
        Args:
            images_dir: Directory containing CAPTCHA images
            labels_file: CSV file with image-label mappings
            output_dir: Directory to save preprocessed data
            image_width: Target width for resized images
            image_height: Target height for resized images
            captcha_length: Number of characters in each CAPTCHA
        """
        self.images_dir = images_dir
        self.labels_file = labels_file
        self.output_dir = output_dir
        self.image_width = image_width
        self.image_height = image_height
        self.captcha_length = captcha_length
        
        # Character set (A-Z, 0-9)
        self.characters = string.ascii_uppercase + string.digits
        self.num_classes = len(self.characters)
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.characters)}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Preprocessor initialized:")
        print(f"  - Image dimensions: {image_width}x{image_height}")
        print(f"  - CAPTCHA length: {captcha_length}")
        print(f"  - Character set: {self.characters}")
        print(f"  - Number of classes: {self.num_classes}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to target dimensions
        resized = cv2.resize(gray, (self.image_width, self.image_height))
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def encode_label_integer(self, label: str) -> np.ndarray:
        """
        Encode label as integer array.
        
        Args:
            label: CAPTCHA text (e.g., "ABC12")
            
        Returns:
            Integer encoded array
        """
        if len(label) != self.captcha_length:
            raise ValueError(f"Label length {len(label)} != expected {self.captcha_length}")
        
        encoded = np.zeros(self.captcha_length, dtype=np.int32)
        for i, char in enumerate(label):
            if char not in self.char_to_idx:
                raise ValueError(f"Unknown character: {char}")
            encoded[i] = self.char_to_idx[char]
        
        return encoded
    
    def encode_label_onehot(self, label: str) -> np.ndarray:
        """
        Encode label as one-hot array.
        
        Args:
            label: CAPTCHA text (e.g., "ABC12")
            
        Returns:
            One-hot encoded array of shape (captcha_length, num_classes)
        """
        if len(label) != self.captcha_length:
            raise ValueError(f"Label length {len(label)} != expected {self.captcha_length}")
        
        encoded = np.zeros((self.captcha_length, self.num_classes), dtype=np.float32)
        for i, char in enumerate(label):
            if char not in self.char_to_idx:
                raise ValueError(f"Unknown character: {char}")
            encoded[i, self.char_to_idx[char]] = 1.0
        
        return encoded
    
    def decode_prediction(self, prediction: np.ndarray) -> str:
        """
        Decode prediction back to text.
        
        Args:
            prediction: Model prediction (either integer indices or probabilities)
            
        Returns:
            Decoded text string
        """
        if prediction.ndim == 1:
            # Integer encoded
            indices = prediction
        else:
            # One-hot or probability encoded
            indices = np.argmax(prediction, axis=-1)
        
        decoded = ''.join([self.idx_to_char[idx] for idx in indices])
        return decoded
    
    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """
        Load image paths and labels from CSV file.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        print(f"Loading dataset from {self.labels_file}...")
        
        # Read CSV file
        df = pd.read_csv(self.labels_file)
        
        image_paths = []
        labels = []
        
        for _, row in df.iterrows():
            filename = row['filename']
            label = row['label']
            
            image_path = os.path.join(self.images_dir, filename)
            
            # Check if image file exists
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(label)
            else:
                print(f"Warning: Image not found: {image_path}")
        
        print(f"Loaded {len(image_paths)} valid samples")
        return image_paths, labels
    
    def preprocess_dataset(self, 
                          encoding_type: str = "integer",
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Preprocess the entire dataset.
        
        Args:
            encoding_type: "integer" or "onehot"
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing preprocessed arrays
        """
        print("Starting dataset preprocessing...")
        
        # Load dataset
        image_paths, labels = self.load_dataset()
        
        if len(image_paths) == 0:
            raise ValueError("No valid images found!")
        
        # Process images
        print("Processing images...")
        images = []
        valid_labels = []
        
        for i, (image_path, label) in enumerate(zip(image_paths, labels)):
            try:
                # Load and preprocess image
                image = self.load_image(image_path)
                images.append(image)
                valid_labels.append(label)
                
                # Progress reporting
                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Successfully processed {len(images)} images")
        
        # Convert to numpy arrays
        X = np.array(images)
        
        # Encode labels
        print(f"Encoding labels using {encoding_type} encoding...")
        if encoding_type == "integer":
            y = np.array([self.encode_label_integer(label) for label in valid_labels])
        elif encoding_type == "onehot":
            y = np.array([self.encode_label_onehot(label) for label in valid_labels])
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        # Split dataset
        print("Splitting dataset...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=None
        )
        
        print(f"Dataset splits:")
        print(f"  - Training: {X_train.shape[0]} samples")
        print(f"  - Validation: {X_val.shape[0]} samples")
        print(f"  - Testing: {X_test.shape[0]} samples")
        
        # Prepare output dictionary
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'encoding_type': encoding_type,
            'image_shape': (self.image_height, self.image_width),
            'captcha_length': self.captcha_length,
            'num_classes': self.num_classes,
            'characters': self.characters,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }
        
        return data
    
    def save_preprocessed_data(self, data: Dict[str, np.ndarray], filename: str = "preprocessed_data.npz"):
        """
        Save preprocessed data to disk.
        
        Args:
            data: Preprocessed data dictionary
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"Saving preprocessed data to {output_path}...")
        
        # Save as compressed numpy archive
        np.savez_compressed(output_path, **data)
        
        # Also save metadata as pickle for easy loading
        metadata = {
            'encoding_type': data['encoding_type'],
            'image_shape': data['image_shape'],
            'captcha_length': data['captcha_length'],
            'num_classes': data['num_classes'],
            'characters': data['characters'],
            'char_to_idx': data['char_to_idx'],
            'idx_to_char': data['idx_to_char']
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Data saved successfully!")
        print(f"  - Main data: {output_path}")
        print(f"  - Metadata: {metadata_path}")
    
    def load_preprocessed_data(self, filename: str = "preprocessed_data.npz") -> Dict[str, np.ndarray]:
        """
        Load preprocessed data from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Preprocessed data dictionary
        """
        data_path = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Preprocessed data not found: {data_path}")
        
        print(f"Loading preprocessed data from {data_path}...")
        
        # Load numpy archive
        loaded = np.load(data_path, allow_pickle=True)
        
        # Convert to dictionary
        data = {key: loaded[key] for key in loaded.files}
        
        print(f"Data loaded successfully!")
        print(f"  - Training samples: {data['X_train'].shape[0]}")
        print(f"  - Validation samples: {data['X_val'].shape[0]}")
        print(f"  - Test samples: {data['X_test'].shape[0]}")
        
        return data
    
    def visualize_samples(self, data: Dict[str, np.ndarray], num_samples: int = 5):
        """
        Visualize some preprocessed samples.
        
        Args:
            data: Preprocessed data dictionary
            num_samples: Number of samples to show
        """
        try:
            import matplotlib.pyplot as plt
            
            X_train = data['X_train']
            y_train = data['y_train']
            encoding_type = data['encoding_type']
            
            fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
            
            for i in range(num_samples):
                if i < len(X_train):
                    # Show image
                    axes[i].imshow(X_train[i], cmap='gray')
                    axes[i].axis('off')
                    
                    # Decode label
                    if encoding_type == 'integer':
                        label = self.decode_prediction(y_train[i])
                    else:  # onehot
                        label = self.decode_prediction(y_train[i])
                    
                    axes[i].set_title(f'Label: {label}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'sample_images.png'), dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Sample visualization saved to {self.output_dir}/sample_images.png")
            
        except ImportError:
            print("Matplotlib not available for visualization")


def main():
    """
    Main function to run preprocessing.
    """
    parser = argparse.ArgumentParser(description='Preprocess CAPTCHA dataset')
    parser.add_argument('--images-dir', type=str, default='dataset/images',
                       help='Directory containing CAPTCHA images')
    parser.add_argument('--labels-file', type=str, default='dataset/labels.csv',
                       help='CSV file with labels')
    parser.add_argument('--output-dir', type=str, default='dataset/preprocessed',
                       help='Output directory for preprocessed data')
    parser.add_argument('--width', type=int, default=100,
                       help='Target image width')
    parser.add_argument('--height', type=int, default=40,
                       help='Target image height')
    parser.add_argument('--encoding', type=str, choices=['integer', 'onehot'], default='integer',
                       help='Label encoding type')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (fraction)')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation set size (fraction)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create sample visualizations')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = CaptchaPreprocessor(
        images_dir=args.images_dir,
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        image_width=args.width,
        image_height=args.height
    )
    
    # Preprocess dataset
    data = preprocessor.preprocess_dataset(
        encoding_type=args.encoding,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(data)
    
    # Visualize samples if requested
    if args.visualize:
        preprocessor.visualize_samples(data)
    
    print("\n=== Preprocessing complete! ===")
    print("Next steps:")
    print("1. Check the 'dataset/preprocessed' folder for processed data")
    print("2. Use the saved .npz file for model training")
    print("3. Ready for Phase 3: Model Training!")


if __name__ == "__main__":
    main()
