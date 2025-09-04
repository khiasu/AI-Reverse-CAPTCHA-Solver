"""
Simple script to run OCR vs CNN comparison on CAPTCHA dataset.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.ocr_comparison import OCRComparison


def main():
    """
    Run OCR vs CNN comparison with default settings.
    """
    print("=== OCR vs CNN Comparison ===")
    print("This script compares Tesseract OCR performance against our trained CNN model.")
    print()
    
    # Configuration
    images_dir = "dataset/images"
    labels_csv = "dataset/labels.csv"
    cnn_results_csv = None  # Will look for latest inference results
    output_dir = "results"
    max_images = 200  # Limit for reasonable comparison time
    
    print(f"Configuration:")
    print(f"  - Images directory: {images_dir}")
    print(f"  - Labels file: {labels_csv}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Max images: {max_images}")
    print()
    
    # Check for Tesseract installation
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR found: {version}")
    except Exception as e:
        print(f"❌ ERROR: Tesseract OCR not found!")
        print(f"Please install Tesseract OCR:")
        print(f"  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print(f"  Then: pip install pytesseract")
        print(f"Error: {e}")
        return
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        print("Please generate dataset first:")
        print("  python run_dataset_generation.py")
        return
    
    # Check if labels file exists
    if not os.path.exists(labels_csv):
        print(f"ERROR: Labels file not found: {labels_csv}")
        print("Please generate dataset first:")
        print("  python run_dataset_generation.py")
        return
    
    # Look for CNN results file
    if os.path.exists(output_dir):
        result_files = [f for f in os.listdir(output_dir) if f.startswith("inference_results_") and f.endswith(".csv")]
        if result_files:
            # Use the most recent CNN results file
            result_files.sort(reverse=True)
            cnn_results_csv = os.path.join(output_dir, result_files[0])
            print(f"✓ Found CNN results: {cnn_results_csv}")
        else:
            print(f"⚠️  No CNN results found in {output_dir}")
            print(f"   Run inference first for full comparison:")
            print(f"   python run_inference.py")
    
    try:
        # Create comparison system
        print("\nInitializing OCR comparison system...")
        comparison_system = OCRComparison(
            images_dir=images_dir,
            labels_csv_path=labels_csv,
            captcha_length=5
        )
        
        # Run comparison
        print("\nStarting OCR vs CNN comparison...")
        ocr_metrics, cnn_metrics = comparison_system.run_comparison(
            cnn_results_csv=cnn_results_csv,
            output_dir=output_dir,
            max_images=max_images
        )
        
        print("\n" + "="*60)
        print("📊 OCR COMPARISON COMPLETED SUCCESSFULLY! 📊")
        print("="*60)
        
        # Performance summary
        print(f"\nPerformance Summary:")
        print(f"  📈 Tesseract OCR:")
        print(f"     - Sequence Accuracy: {ocr_metrics.get('sequence_accuracy', 0):.1%}")
        print(f"     - Character Accuracy: {ocr_metrics.get('character_accuracy', 0):.1%}")
        
        if cnn_metrics:
            print(f"  🧠 CNN Model:")
            print(f"     - Sequence Accuracy: {cnn_metrics.get('sequence_accuracy', 0):.1%}")
            print(f"     - Character Accuracy: {cnn_metrics.get('character_accuracy', 0):.1%}")
            
            # Calculate improvement
            seq_improvement = cnn_metrics.get('sequence_accuracy', 0) - ocr_metrics.get('sequence_accuracy', 0)
            char_improvement = cnn_metrics.get('character_accuracy', 0) - ocr_metrics.get('character_accuracy', 0)
            
            print(f"  🚀 CNN Improvement:")
            print(f"     - Sequence: {seq_improvement:+.1%}")
            print(f"     - Character: {char_improvement:+.1%}")
        
        print(f"\nGenerated Files:")
        print(f"✓ {output_dir}/ocr_comparison_*.csv (detailed comparison)")
        
        print(f"\nKey Insights:")
        if cnn_metrics and cnn_metrics.get('sequence_accuracy', 0) > ocr_metrics.get('sequence_accuracy', 0):
            print(f"  🎯 CNN model significantly outperforms traditional OCR")
            print(f"  🔬 Deep learning approach handles CAPTCHA distortions better")
        elif not cnn_metrics:
            print(f"  📋 OCR baseline established - run CNN inference for comparison")
        else:
            print(f"  🤔 Results show interesting performance characteristics")
        
        print(f"\nNext Steps:")
        if not cnn_metrics:
            print(f"  1. Run CNN inference: python run_inference.py")
            print(f"  2. Re-run comparison for full analysis")
        else:
            print(f"  1. Analyze detailed results in CSV file")
            print(f"  2. Ready for demo app development!")
        
    except Exception as e:
        print(f"\nERROR during comparison: {e}")
        print("\nTroubleshooting:")
        print("1. Install Tesseract OCR and pytesseract")
        print("2. Check if dataset exists: run_dataset_generation.py")
        print("3. Verify all dependencies are installed")
        return


if __name__ == "__main__":
    main()
