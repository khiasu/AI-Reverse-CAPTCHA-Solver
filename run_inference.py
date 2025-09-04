"""
Simple script to run CAPTCHA inference on test images.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.captcha_inference import CaptchaInference


def main():
    """
    Run CAPTCHA inference with default settings.
    """
    print("=== CAPTCHA Inference Pipeline ===")
    print("This script will run inference on CAPTCHA images using the trained model.")
    print()
    
    # Configuration
    model_path = "model/best_model.h5"
    images_dir = "dataset/images"
    labels_csv = "dataset/labels.csv"
    output_dir = "results"
    max_images = 100  # Limit for quick testing
    
    print(f"Configuration:")
    print(f"  - Model: {model_path}")
    print(f"  - Images directory: {images_dir}")
    print(f"  - Labels file: {labels_csv}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Max images (testing): {max_images}")
    print()
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please train the model first:")
        print("  python run_model_training.py")
        return
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        print("Please generate dataset first:")
        print("  python run_dataset_generation.py")
        return
    
    try:
        # Create inference system
        print("Initializing inference system...")
        inference_system = CaptchaInference(
            model_path=model_path,
            image_height=40,
            image_width=100,
            captcha_length=5
        )
        
        # Run inference
        print("\nStarting inference...")
        metrics = inference_system.run_inference(
            images_dir=images_dir,
            labels_csv_path=labels_csv,
            output_dir=output_dir,
            max_images=max_images
        )
        
        print("\n" + "="*50)
        print("🎯 INFERENCE COMPLETED SUCCESSFULLY! 🎯")
        print("="*50)
        
        if metrics:
            print(f"\nPerformance Summary:")
            print(f"  - Sequence Accuracy: {metrics.get('sequence_accuracy', 0):.1%}")
            print(f"  - Character Accuracy: {metrics.get('character_accuracy', 0):.1%}")
            print(f"  - Total Samples: {metrics.get('total_samples', 0)}")
            
            print(f"\nCharacter-wise Performance:")
            for i in range(5):
                char_acc = metrics.get(f'char_{i+1}_accuracy', 0)
                print(f"  - Position {i+1}: {char_acc:.1%}")
        
        print(f"\nGenerated Files:")
        print(f"✓ {output_dir}/inference_results_*.csv (detailed results)")
        
        print(f"\nNext Steps:")
        print(f"  1. Review results CSV for detailed analysis")
        print(f"  2. Run on full dataset: remove --max-images limit")
        print(f"  3. Ready for demo app development!")
        
    except Exception as e:
        print(f"\nERROR during inference: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure model is trained: run_model_training.py")
        print("2. Check if images exist: run_dataset_generation.py")
        print("3. Verify TensorFlow installation")
        return


if __name__ == "__main__":
    main()
