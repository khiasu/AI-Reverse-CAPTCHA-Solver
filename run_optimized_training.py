"""
Enhanced CAPTCHA CNN model training with advanced optimizations.
Includes data augmentation, mixed precision, and model quantization.
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.captcha_cnn_model import CaptchaCNNModel
from model.optimized_training import OptimizedTrainer


def main():
    """Run optimized CAPTCHA model training."""
    parser = argparse.ArgumentParser(description='Enhanced CAPTCHA CNN Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision')
    parser.add_argument('--quantize', action='store_true', help='Apply post-training quantization')
    
    args = parser.parse_args()
    
    print("=== ENHANCED CAPTCHA CNN TRAINING ===")
    print("🚀 Advanced optimizations enabled!")
    print()
    
    # Configuration
    data_file = "dataset/preprocessed/preprocessed_data_onehot.npz"
    model_dir = "model"
    
    print(f"Configuration:")
    print(f"  📁 Data file: {data_file}")
    print(f"  📁 Model directory: {model_dir}")
    print(f"  🔄 Epochs: {args.epochs}")
    print(f"  📦 Batch size: {args.batch_size}")
    print(f"  ⏹️  Early stopping patience: {args.patience}")
    print(f"  🔄 Data augmentation: {'❌ Disabled' if args.no_augmentation else '✅ Enabled'}")
    print(f"  ⚡ Mixed precision: {'❌ Disabled' if args.no_mixed_precision else '✅ Enabled'}")
    print(f"  🗜️  Model quantization: {'✅ Enabled' if args.quantize else '❌ Disabled'}")
    print()
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"❌ ERROR: Data file not found: {data_file}")
        print("Please run preprocessing first:")
        print("  python run_preprocessing.py")
        return
    
    try:
        # Load preprocessed data
        print(f"📂 Loading data from: {data_file}")
        data = np.load(data_file, allow_pickle=True)
        data_dict = {key: data[key] for key in data.files}
        
        print(f"✅ Data loaded successfully:")
        print(f"   🎯 Training samples: {data_dict['X_train'].shape[0]:,}")
        print(f"   🔍 Validation samples: {data_dict['X_val'].shape[0]:,}")
        print(f"   📊 Test samples: {data_dict['X_test'].shape[0]:,}")
        print(f"   📐 Image shape: {data_dict['image_shape']}")
        print(f"   🔤 Captcha length: {data_dict['captcha_length']}")
        print(f"   🎲 Number of classes: {data_dict['num_classes']}")
        print()
        
        # Create model
        print("🏗️  Building CNN architecture...")
        model_trainer = CaptchaCNNModel(
            image_height=int(data_dict['image_shape'][0]),
            image_width=int(data_dict['image_shape'][1]),
            captcha_length=int(data_dict['captcha_length']),
            num_classes=int(data_dict['num_classes']),
            model_dir=model_dir
        )
        
        # Build model
        model_trainer.build_model(dropout_rate=0.5)
        model = model_trainer.model
        
        print("✅ CNN model built successfully!")
        print(f"   📊 Total parameters: {model.count_params():,}")
        print()
        
        # Create optimized trainer
        print("⚙️  Initializing optimized trainer...")
        optimizer = OptimizedTrainer(model, model_dir)
        
        # Train model with optimizations
        print("🚀 Starting optimized training...")
        print("=" * 60)
        
        history = optimizer.train_with_augmentation(
            data_dict=data_dict,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_augmentation=not args.no_augmentation,
            use_mixed_precision=not args.no_mixed_precision
        )
        
        print("=" * 60)
        print("✅ Training completed!")
        print()
        
        # Save final model
        print("💾 Saving final model...")
        final_model_path = os.path.join(model_dir, 'model_final.keras')
        model.save(final_model_path)
        print(f"✅ Final model saved: {final_model_path}")
        
        # Evaluate model
        print("📊 Evaluating model performance...")
        model_trainer.model = model  # Update trainer's model reference
        metrics = model_trainer.evaluate_model(data_dict)
        
        # Plot training curves
        print("📈 Generating training curves...")
        model_trainer.plot_training_curves(history)
        
        # Apply quantization if requested
        if args.quantize:
            print("🗜️  Applying post-training quantization...")
            best_model_path = os.path.join(model_dir, 'best_model.keras')
            if os.path.exists(best_model_path):
                optimizer.quantize_model(best_model_path)
            else:
                optimizer.quantize_model(final_model_path)
        
        # Benchmark model
        print("⚡ Benchmarking inference speed...")
        sample_data = data_dict['X_test'][:10].astype(np.float32) / 255.0
        if os.path.exists(os.path.join(model_dir, 'best_model.keras')):
            optimizer.benchmark_model(os.path.join(model_dir, 'best_model.keras'), sample_data)
        else:
            optimizer.benchmark_model(final_model_path, sample_data)
        
        # Display final results
        print("\n" + "=" * 60)
        print("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        
        print("\n📁 Generated files:")
        files_generated = [
            ("best_model.keras", "Best model checkpoint (.keras format)"),
            ("model_final.keras", "Final trained model"),
            ("training_curves.png", "Training visualization plots"),
            ("training_log.csv", "Detailed training metrics")
        ]
        
        if args.quantize:
            files_generated.append(("model_quantized.tflite", "Quantized model for deployment"))
        
        for filename, description in files_generated:
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"✅ {filepath} ({size_mb:.2f} MB) - {description}")
        
        print(f"\n📊 Final Performance Metrics:")
        if metrics:
            sequence_acc = metrics.get('sequence_accuracy', 0)
            char_accuracies = [metrics.get(f'char_{i+1}_accuracy', 0) for i in range(5)]
            avg_char_acc = np.mean(char_accuracies)
            
            print(f"   🎯 Full Sequence Accuracy: {sequence_acc:.1%}")
            print(f"   🔤 Average Character Accuracy: {avg_char_acc:.1%}")
            print(f"   📍 Individual Character Accuracies:")
            for i, acc in enumerate(char_accuracies):
                print(f"      Char {i+1}: {acc:.1%}")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. 📈 Review training curves: {model_dir}/training_curves.png")
        print(f"   2. 🖥️  Use model for inference and demo")
        print(f"   3. 🚀 Deploy to production with the optimized model!")
        
        # Training recommendations
        if not args.no_augmentation and avg_char_acc < 0.7:
            print(f"\n💡 Performance Tips:")
            print(f"   • Consider increasing training epochs")
            print(f"   • Try adjusting learning rate schedule")
            print(f"   • Collect more diverse training data")
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure TensorFlow is installed: pip install tensorflow")
        print("   2. Check preprocessed data exists: python run_preprocessing.py")
        print("   3. For GPU training, ensure CUDA is properly configured")
        print("   4. Try reducing batch size if running out of memory")
        return


if __name__ == "__main__":
    main()
