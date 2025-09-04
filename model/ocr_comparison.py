"""
OCR Baseline Comparison
Compares CNN model performance against Tesseract OCR on CAPTCHA dataset.
"""

import os
import numpy as np
import pandas as pd
import cv2
import pytesseract
from typing import List, Tuple, Dict, Optional
import argparse
from datetime import datetime
import re
from tabulate import tabulate
import matplotlib.pyplot as plt


class OCRComparison:
    """
    Comparison system between CNN model and Tesseract OCR for CAPTCHA recognition.
    """
    
    def __init__(self, 
                 images_dir: str,
                 labels_csv_path: str,
                 captcha_length: int = 5):
        """
        Initialize the OCR comparison system.
        
        Args:
            images_dir: Directory containing CAPTCHA images
            labels_csv_path: Path to ground truth labels CSV
            captcha_length: Number of characters in CAPTCHA
        """
        self.images_dir = images_dir
        self.labels_csv_path = labels_csv_path
        self.captcha_length = captcha_length
        
        # Load ground truth
        self.ground_truth = self.load_ground_truth()
        
        print(f"OCR Comparison System initialized:")
        print(f"  - Images directory: {images_dir}")
        print(f"  - Labels file: {labels_csv_path}")
        print(f"  - Ground truth samples: {len(self.ground_truth)}")
        print(f"  - CAPTCHA length: {captcha_length}")
    
    def load_ground_truth(self) -> Dict[str, str]:
        """
        Load ground truth labels from CSV file.
        
        Returns:
            Dictionary mapping filename to ground truth label
        """
        if not os.path.exists(self.labels_csv_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_csv_path}")
        
        try:
            df = pd.read_csv(self.labels_csv_path)
            ground_truth = dict(zip(df['filename'], df['label']))
            print(f"Loaded {len(ground_truth)} ground truth labels")
            return ground_truth
            
        except Exception as e:
            raise RuntimeError(f"Error loading ground truth: {e}")
    
    def preprocess_image_for_ocr(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR performance.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply various preprocessing techniques for better OCR
        # 1. Resize to larger size for better OCR accuracy
        height, width = gray.shape
        scale_factor = 3
        resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        
        # 2. Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # 3. Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def run_tesseract_ocr(self, image_path: str) -> Tuple[str, float]:
        """
        Run Tesseract OCR on a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_text, confidence)
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image_for_ocr(image_path)
            
            # Configure Tesseract for alphanumeric characters only
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Run OCR
            ocr_result = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Clean up result
            predicted_text = re.sub(r'[^A-Z0-9]', '', ocr_result.upper().strip())
            
            # Pad or truncate to expected length
            if len(predicted_text) < self.captcha_length:
                predicted_text = predicted_text.ljust(self.captcha_length, '?')
            elif len(predicted_text) > self.captcha_length:
                predicted_text = predicted_text[:self.captcha_length]
            
            # Get confidence (simplified - Tesseract confidence is complex)
            try:
                data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            except:
                avg_confidence = 0.5  # Default confidence if extraction fails
            
            return predicted_text, float(avg_confidence)
            
        except Exception as e:
            print(f"  OCR error for {image_path}: {e}")
            return "ERROR", 0.0
    
    def run_ocr_batch(self, image_paths: List[str]) -> List[Tuple[str, str, float]]:
        """
        Run OCR on multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of tuples (filename, predicted_text, confidence)
        """
        results = []
        
        print(f"Running Tesseract OCR on {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                filename = os.path.basename(image_path)
                predicted_text, confidence = self.run_tesseract_ocr(image_path)
                results.append((filename, predicted_text, confidence))
                
                # Progress update
                if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                    print(f"  Processed {i + 1}/{len(image_paths)} images")
                
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                results.append((os.path.basename(image_path), "ERROR", 0.0))
        
        return results
    
    def load_cnn_results(self, cnn_results_csv: str) -> Dict[str, Tuple[str, float]]:
        """
        Load CNN model results from CSV file.
        
        Args:
            cnn_results_csv: Path to CNN results CSV file
            
        Returns:
            Dictionary mapping filename to (prediction, confidence)
        """
        if not os.path.exists(cnn_results_csv):
            print(f"Warning: CNN results file not found: {cnn_results_csv}")
            return {}
        
        try:
            df = pd.read_csv(cnn_results_csv)
            cnn_results = {}
            
            for _, row in df.iterrows():
                filename = row['filename']
                prediction = row['prediction']
                confidence = row['overall_confidence']
                cnn_results[filename] = (prediction, confidence)
            
            print(f"Loaded {len(cnn_results)} CNN results")
            return cnn_results
            
        except Exception as e:
            print(f"Error loading CNN results: {e}")
            return {}
    
    def calculate_metrics(self, predictions: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """
        Calculate accuracy metrics for predictions.
        
        Args:
            predictions: List of (filename, predicted_text, confidence) tuples
            
        Returns:
            Dictionary of metrics
        """
        total_samples = 0
        correct_sequences = 0
        correct_characters = 0
        total_characters = 0
        
        character_accuracies = [0] * self.captcha_length
        character_counts = [0] * self.captcha_length
        
        for filename, predicted_text, confidence in predictions:
            if filename in self.ground_truth and predicted_text != "ERROR":
                true_text = self.ground_truth[filename]
                total_samples += 1
                
                # Sequence accuracy
                if predicted_text == true_text:
                    correct_sequences += 1
                
                # Character-wise accuracy
                for i, (pred_char, true_char) in enumerate(zip(predicted_text, true_text)):
                    if i < self.captcha_length:
                        total_characters += 1
                        character_counts[i] += 1
                        
                        if pred_char == true_char:
                            correct_characters += 1
                            character_accuracies[i] += 1
        
        # Calculate metrics
        metrics = {}
        
        if total_samples > 0:
            metrics['sequence_accuracy'] = correct_sequences / total_samples
            metrics['character_accuracy'] = correct_characters / total_characters
            metrics['total_samples'] = total_samples
            
            # Individual character accuracies
            for i in range(self.captcha_length):
                if character_counts[i] > 0:
                    metrics[f'char_{i+1}_accuracy'] = character_accuracies[i] / character_counts[i]
        
        return metrics
    
    def create_comparison_table(self, 
                              ocr_metrics: Dict[str, float], 
                              cnn_metrics: Dict[str, float] = None) -> str:
        """
        Create a formatted comparison table.
        
        Args:
            ocr_metrics: OCR performance metrics
            cnn_metrics: CNN performance metrics (optional)
            
        Returns:
            Formatted table string
        """
        # Prepare data for table
        table_data = []
        
        # Overall metrics
        table_data.append([
            "Sequence Accuracy", 
            f"{ocr_metrics.get('sequence_accuracy', 0):.1%}",
            f"{cnn_metrics.get('sequence_accuracy', 0):.1%}" if cnn_metrics else "N/A"
        ])
        
        table_data.append([
            "Character Accuracy", 
            f"{ocr_metrics.get('character_accuracy', 0):.1%}",
            f"{cnn_metrics.get('character_accuracy', 0):.1%}" if cnn_metrics else "N/A"
        ])
        
        # Character-wise accuracies
        for i in range(self.captcha_length):
            ocr_char_acc = ocr_metrics.get(f'char_{i+1}_accuracy', 0)
            cnn_char_acc = cnn_metrics.get(f'char_{i+1}_accuracy', 0) if cnn_metrics else 0
            
            table_data.append([
                f"Character {i+1} Accuracy", 
                f"{ocr_char_acc:.1%}",
                f"{cnn_char_acc:.1%}" if cnn_metrics else "N/A"
            ])
        
        # Sample count
        table_data.append([
            "Total Samples", 
            str(ocr_metrics.get('total_samples', 0)),
            str(cnn_metrics.get('total_samples', 0)) if cnn_metrics else "N/A"
        ])
        
        # Create table
        headers = ["Metric", "Tesseract OCR", "CNN Model"] if cnn_metrics else ["Metric", "Tesseract OCR"]
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        
        return table_str
    
    def save_comparison_results(self, 
                              ocr_predictions: List[Tuple[str, str, float]], 
                              cnn_results: Dict[str, Tuple[str, float]], 
                              output_path: str) -> None:
        """
        Save detailed comparison results to CSV.
        
        Args:
            ocr_predictions: OCR prediction results
            cnn_results: CNN prediction results
            output_path: Path to save CSV file
        """
        results_data = []
        
        for filename, ocr_pred, ocr_conf in ocr_predictions:
            # Get ground truth
            true_text = self.ground_truth.get(filename, "N/A")
            
            # Get CNN results
            cnn_pred, cnn_conf = cnn_results.get(filename, ("N/A", 0.0))
            
            # Calculate correctness
            ocr_correct = (ocr_pred == true_text) if true_text != "N/A" else None
            cnn_correct = (cnn_pred == true_text) if true_text != "N/A" and cnn_pred != "N/A" else None
            
            row = {
                'filename': filename,
                'ground_truth': true_text,
                'ocr_prediction': ocr_pred,
                'ocr_confidence': round(ocr_conf, 4),
                'ocr_correct': ocr_correct,
                'cnn_prediction': cnn_pred,
                'cnn_confidence': round(cnn_conf, 4),
                'cnn_correct': cnn_correct
            }
            
            results_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(results_data)
        df.to_csv(output_path, index=False)
        
        print(f"Detailed comparison results saved to: {output_path}")
    
    def run_comparison(self, 
                      cnn_results_csv: str = None, 
                      output_dir: str = "results",
                      max_images: int = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run complete OCR vs CNN comparison.
        
        Args:
            cnn_results_csv: Path to CNN results CSV (optional)
            output_dir: Directory to save results
            max_images: Maximum number of images to process
            
        Returns:
            Tuple of (ocr_metrics, cnn_metrics)
        """
        print(f"=== OCR vs CNN Comparison ===")
        print(f"Images directory: {self.images_dir}")
        print(f"CNN results: {cnn_results_csv}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for filename in os.listdir(self.images_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(self.images_dir, filename))
        
        # Limit number of images if specified
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]
            print(f"Limited to {max_images} images for testing")
        
        print(f"Found {len(image_files)} images to process")
        
        if not image_files:
            print("No images found!")
            return {}, {}
        
        # Run OCR predictions
        start_time = datetime.now()
        ocr_predictions = self.run_ocr_batch(image_files)
        ocr_time = datetime.now() - start_time
        
        print(f"\nOCR completed in {ocr_time}")
        print(f"Average OCR time per image: {ocr_time.total_seconds() / len(image_files):.3f} seconds")
        
        # Calculate OCR metrics
        ocr_metrics = self.calculate_metrics(ocr_predictions)
        
        # Load CNN results if available
        cnn_results = {}
        cnn_metrics = {}
        
        if cnn_results_csv:
            cnn_results = self.load_cnn_results(cnn_results_csv)
            
            # Extract CNN metrics from loaded results
            if cnn_results:
                cnn_predictions = []
                for filename, (prediction, confidence) in cnn_results.items():
                    cnn_predictions.append((filename, prediction, confidence))
                cnn_metrics = self.calculate_metrics(cnn_predictions)
        
        # Create and display comparison table
        comparison_table = self.create_comparison_table(ocr_metrics, cnn_metrics)
        print(f"\n=== Performance Comparison ===")
        print(comparison_table)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_csv = os.path.join(output_dir, f"ocr_comparison_{timestamp}.csv")
        self.save_comparison_results(ocr_predictions, cnn_results, results_csv)
        
        # Print summary
        print(f"\n=== Summary ===")
        print(f"OCR Sequence Accuracy: {ocr_metrics.get('sequence_accuracy', 0):.1%}")
        if cnn_metrics:
            print(f"CNN Sequence Accuracy: {cnn_metrics.get('sequence_accuracy', 0):.1%}")
            improvement = cnn_metrics.get('sequence_accuracy', 0) - ocr_metrics.get('sequence_accuracy', 0)
            print(f"CNN Improvement: {improvement:.1%}")
        
        return ocr_metrics, cnn_metrics


def main():
    """
    Main function for OCR comparison.
    """
    parser = argparse.ArgumentParser(description='OCR vs CNN Comparison')
    parser.add_argument('--images', type=str, default='dataset/images',
                       help='Directory containing CAPTCHA images')
    parser.add_argument('--labels', type=str, default='dataset/labels.csv',
                       help='Path to ground truth labels CSV')
    parser.add_argument('--cnn-results', type=str, default=None,
                       help='Path to CNN results CSV file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--max-images', type=int, default=200,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    try:
        # Check Tesseract installation
        try:
            pytesseract.get_tesseract_version()
            print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print(f"ERROR: Tesseract not found. Please install Tesseract OCR.")
            print(f"Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print(f"Error: {e}")
            return
        
        # Create comparison system
        comparison_system = OCRComparison(
            images_dir=args.images,
            labels_csv_path=args.labels,
            captcha_length=5
        )
        
        # Run comparison
        ocr_metrics, cnn_metrics = comparison_system.run_comparison(
            cnn_results_csv=args.cnn_results,
            output_dir=args.output,
            max_images=args.max_images
        )
        
        print(f"\n🔍 OCR Comparison completed successfully!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Install Tesseract OCR and pytesseract")
        print("2. Check images directory path")
        print("3. Ensure labels CSV exists")


if __name__ == "__main__":
    main()
