"""
Simple script to run CAPTCHA CNN model training.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.captcha_cnn_model import CaptchaCNNModel
import numpy as np


def main():
    """
    Run CAPTCHA model training with default settings.
    """
    print("=== CAPTCHA CNN Model Training ===")
    print("This script will train a CNN model for CAPTCHA recognition.")
    print()
    
    # Configuration
    data_file = "dataset/preprocessed/preprocessed_data_onehot.npz"
    model_dir = "model"
    epochs = 50
    batch_size = 32
    patience = 10
    dropout_rate = 0.5
    
    print(f"Configuration:")
    print(f"  - Data file: {data_file}")
    print(f"  - Model directory: {model_dir}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - Dropout rate: {dropout_rate}")
    print()
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        print("Please run preprocessing first:")
        print("  python run_preprocessing.py")
        return
    
    try:
        # Load preprocessed data
        print(f"Loading data from: {data_file}")
        data = np.load(data_file, allow_pickle=True)
        data_dict = {key: data[key] for key in data.files}
        
        print(f"Data loaded successfully:")
        print(f"  - Training samples: {data_dict['X_train'].shape[0]}")
        print(f"  - Validation samples: {data_dict['X_val'].shape[0]}")
        print(f"  - Test samples: {data_dict['X_test'].shape[0]}")
        print(f"  - Image shape: {data_dict['image_shape']}")
        print(f"  - Encoding type: {data_dict['encoding_type']}")
        print()
        
        # Create model trainer
        model_trainer = CaptchaCNNModel(
            image_height=int(data_dict['image_shape'][0]),
            image_width=int(data_dict['image_shape'][1]),
            captcha_length=int(data_dict['captcha_length']),
            num_classes=int(data_dict['num_classes']),
            model_dir=model_dir
        )
        
        # Build and train model
        print("Building CNN model...")
        model_trainer.build_model(dropout_rate=dropout_rate)
        
        print("\nStarting training...")
        history = model_trainer.train(
            data=data_dict,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        # Plot training curves
        print("\nGenerating training curves...")
        model_trainer.plot_training_curves()
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = model_trainer.evaluate_model(data_dict)
        
        # Save training summary
        model_trainer.save_training_summary(metrics)
        
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*50)
        print("\nGenerated files:")
        print(f"✓ {model_dir}/model.h5 (final trained model)")
        print(f"✓ {model_dir}/best_model.h5 (best model checkpoint)")
        print(f"✓ {model_dir}/training_curves.png (training plots)")
        print(f"✓ {model_dir}/training_log.csv (detailed training log)")
        print(f"✓ {model_dir}/training_summary.pkl (training metadata)")
        
        print(f"\nFinal Results:")
        print(f"  - Sequence Accuracy: {metrics.get('sequence_accuracy', 0):.1%}")
        print(f"  - Average Character Accuracy: {np.mean([metrics.get(f'char_{i+1}_accuracy', 0) for i in range(5)]):.1%}")
        
        print(f"\nNext Steps:")
        print(f"  1. Review training curves in: {model_dir}/training_curves.png")
        print(f"  2. Use model.h5 for inference and demo")
        print(f"  3. Ready for Phase 4: Demo App Development!")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure TensorFlow is installed: pip install tensorflow")
        print("2. Check if preprocessed data exists: run_preprocessing.py")
        print("3. For GPU training, ensure CUDA is properly configured")
        return


if __name__ == "__main__":
    main()
