"""
Advanced training optimizations for CAPTCHA CNN model.
Includes data augmentation, learning rate scheduling, and model optimizations.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, 
    EarlyStopping, 
    ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

class OptimizedTrainer:
    """Enhanced training with advanced optimization techniques."""
    
    def __init__(self, model, model_dir="model"):
        self.model = model
        self.model_dir = model_dir
        
        # Suppress TensorFlow warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        tf.get_logger().setLevel('ERROR')
    
    def create_data_augmentation(self):
        """Create data augmentation generator for training data."""
        return ImageDataGenerator(
            rotation_range=5,           # Slight rotation
            width_shift_range=0.1,      # Horizontal shifts
            height_shift_range=0.1,     # Vertical shifts
            shear_range=0.1,           # Shear transformations
            zoom_range=0.1,            # Zoom in/out
            brightness_range=[0.8, 1.2], # Brightness variation
            fill_mode='nearest'        # Fill pixels after transformation
        )
    
    def create_callbacks(self, patience=15):
        """Create optimized callbacks for training."""
        callbacks = []
        
        # Model checkpoint - save best model (use .keras format)
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.model_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Learning rate reduction
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Custom learning rate scheduler
        def lr_schedule(epoch):
            """Learning rate schedule with warmup and cosine decay."""
            initial_lr = 1e-3
            if epoch < 5:
                # Warmup phase
                return initial_lr * (epoch + 1) / 5
            else:
                # Cosine decay
                decay_epochs = 45
                progress = min((epoch - 5) / decay_epochs, 1.0)
                return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
        callbacks.append(lr_scheduler)
        
        return callbacks
    
    def optimize_model_compilation(self):
        """Compile model with optimized settings."""
        # Use advanced Adam optimizer
        optimizer = Adam(
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0  # Gradient clipping
        )
        
        # Define losses and metrics for each character position
        losses = {f'char_{i+1}': 'categorical_crossentropy' for i in range(5)}
        metrics = {f'char_{i+1}': ['accuracy'] for i in range(5)}
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics
        )
    
    def create_mixed_precision_strategy(self):
        """Enable mixed precision for faster training."""
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision enabled (FP16)")
            return True
        except:
            print("⚠ Mixed precision not available, using FP32")
            return False
    
    def train_with_augmentation(self, data_dict, epochs=50, batch_size=32, 
                               use_augmentation=True, use_mixed_precision=True):
        """Train model with all optimizations."""
        
        print("=== OPTIMIZED TRAINING CONFIGURATION ===")
        
        # Enable mixed precision if available
        if use_mixed_precision:
            self.create_mixed_precision_strategy()
        
        # Optimize model compilation
        self.optimize_model_compilation()
        
        # Prepare data
        X_train = data_dict['X_train'].astype(np.float32) / 255.0
        X_val = data_dict['X_val'].astype(np.float32) / 255.0
        
        # Prepare target data for multi-output model
        y_train = {f'char_{i+1}': data_dict['y_train'][:, i, :] for i in range(5)}
        y_val = {f'char_{i+1}': data_dict['y_val'][:, i, :] for i in range(5)}
        
        # Create callbacks
        callbacks = self.create_callbacks(patience=15)
        
        # Training with/without data augmentation
        if use_augmentation:
            print("🚀 Training with data augmentation")
            datagen = self.create_data_augmentation()
            
            # Fit data generator
            datagen.fit(X_train)
            
            # Calculate steps per epoch
            steps_per_epoch = len(X_train) // batch_size
            
            # Train with generator
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            print("🚀 Training without data augmentation")
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def quantize_model(self, model_path):
        """Apply post-training quantization for deployment."""
        try:
            # Load the trained model
            model = tf.keras.models.load_model(model_path)
            
            # Convert to TensorFlow Lite with quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Enable dynamic range quantization
            converter.target_spec.supported_types = [tf.float16]
            
            quantized_model = converter.convert()
            
            # Save quantized model
            quantized_path = os.path.join(self.model_dir, 'model_quantized.tflite')
            with open(quantized_path, 'wb') as f:
                f.write(quantized_model)
            
            print(f"✓ Quantized model saved: {quantized_path}")
            
            # Get file sizes for comparison
            original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)  # MB
            compression_ratio = original_size / quantized_size
            
            print(f"📊 Model Compression:")
            print(f"   Original: {original_size:.2f} MB")
            print(f"   Quantized: {quantized_size:.2f} MB")
            print(f"   Compression: {compression_ratio:.1f}x smaller")
            
            return quantized_path
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            return None
    
    def benchmark_model(self, model_path, sample_data):
        """Benchmark model inference speed."""
        try:
            model = tf.keras.models.load_model(model_path)
            
            # Warmup
            for _ in range(5):
                _ = model.predict(sample_data[:1], verbose=0)
            
            # Benchmark
            import time
            start_time = time.time()
            
            num_iterations = 100
            for _ in range(num_iterations):
                _ = model.predict(sample_data[:1], verbose=0)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            
            print(f"⚡ Inference Speed: {avg_time:.2f}ms per image")
            return avg_time
            
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
            return None
