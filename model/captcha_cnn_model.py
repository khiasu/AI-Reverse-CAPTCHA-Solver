"""
CAPTCHA CNN Model Training
Implements a Convolutional Neural Network for multi-character CAPTCHA recognition.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt
import pickle
from typing import Dict, Tuple, List
import argparse
from datetime import datetime


class CaptchaCNNModel:
    """
    CNN model for CAPTCHA recognition with multi-character output.
    """
    
    def __init__(self, 
                 image_height: int = 40,
                 image_width: int = 100,
                 captcha_length: int = 5,
                 num_classes: int = 36,
                 model_dir: str = "model"):
        """
        Initialize the CNN model.
        
        Args:
            image_height: Height of input images
            image_width: Width of input images
            captcha_length: Number of characters in CAPTCHA
            num_classes: Number of possible characters (A-Z, 0-9 = 36)
            model_dir: Directory to save model files
        """
        self.image_height = image_height
        self.image_width = image_width
        self.captcha_length = captcha_length
        self.num_classes = num_classes
        self.model_dir = model_dir
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.history = None
        
        print(f"CNN Model initialized:")
        print(f"  - Input shape: ({image_height}, {image_width}, 1)")
        print(f"  - Output shape: {captcha_length} characters, {num_classes} classes each")
        print(f"  - Model directory: {model_dir}")
    
    def build_model(self, dropout_rate: float = 0.5) -> keras.Model:
        """
        Build the CNN architecture.
        
        Args:
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        print("Building CNN architecture...")
        
        # Input layer
        input_layer = layers.Input(shape=(self.image_height, self.image_width, 1), name='input_image')
        
        # First Convolutional Block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(input_layer)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(dropout_rate * 0.5, name='dropout1')(x)
        
        # Second Convolutional Block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(dropout_rate * 0.5, name='dropout2')(x)
        
        # Third Convolutional Block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(dropout_rate * 0.7, name='dropout3')(x)
        
        # Fourth Convolutional Block (deeper for better feature extraction)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
        x = layers.BatchNormalization(name='bn4')(x)
        x = layers.MaxPooling2D((2, 2), name='pool4')(x)
        x = layers.Dropout(dropout_rate, name='dropout4')(x)
        
        # Flatten for dense layers
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu', name='dense1')(x)
        x = layers.BatchNormalization(name='bn_dense1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense1')(x)
        
        x = layers.Dense(256, activation='relu', name='dense2')(x)
        x = layers.BatchNormalization(name='bn_dense2')(x)
        x = layers.Dropout(dropout_rate, name='dropout_dense2')(x)
        
        # Output layers - one for each character position
        outputs = []
        for i in range(self.captcha_length):
            output = layers.Dense(self.num_classes, activation='softmax', name=f'char_{i+1}')(x)
            outputs.append(output)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=outputs, name='captcha_cnn')
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print(f"Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def prepare_data(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray]]:
        """
        Prepare data for training.
        
        Args:
            data: Preprocessed data dictionary
            
        Returns:
            Tuple of (X_train, y_train_list, X_val, y_val_list)
        """
        print("Preparing data for training...")
        
        X_train = data['X_train']
        X_val = data['X_val']
        
        # Add channel dimension for grayscale images
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        
        # Handle different encoding types
        if data['encoding_type'] == 'onehot':
            # Data is already one-hot encoded
            y_train = data['y_train']  # Shape: (samples, captcha_length, num_classes)
            y_val = data['y_val']
        else:
            # Convert integer encoding to one-hot
            y_train_int = data['y_train']  # Shape: (samples, captcha_length)
            y_val_int = data['y_val']
            
            # Convert to one-hot
            y_train = np.zeros((y_train_int.shape[0], self.captcha_length, self.num_classes))
            y_val = np.zeros((y_val_int.shape[0], self.captcha_length, self.num_classes))
            
            for i in range(self.captcha_length):
                y_train[:, i, :] = keras.utils.to_categorical(y_train_int[:, i], self.num_classes)
                y_val[:, i, :] = keras.utils.to_categorical(y_val_int[:, i], self.num_classes)
        
        # Split into separate outputs for each character position
        y_train_list = [y_train[:, i, :] for i in range(self.captcha_length)]
        y_val_list = [y_val[:, i, :] for i in range(self.captcha_length)]
        
        print(f"Data prepared:")
        print(f"  - Training: X={X_train.shape}, y={len(y_train_list)} outputs of {y_train_list[0].shape}")
        print(f"  - Validation: X={X_val.shape}, y={len(y_val_list)} outputs of {y_val_list[0].shape}")
        
        return X_train, y_train_list, X_val, y_val_list
    
    def create_callbacks(self, patience: int = 10) -> List[callbacks.Callback]:
        """
        Create training callbacks.
        
        Args:
            patience: Early stopping patience
            
        Returns:
            List of callbacks
        """
        callback_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                filename=os.path.join(self.model_dir, 'training_log.csv'),
                append=False
            )
        ]
        
        return callback_list
    
    def train(self, 
              data: Dict[str, np.ndarray],
              epochs: int = 50,
              batch_size: int = 32,
              patience: int = 10) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            data: Preprocessed data dictionary
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        print(f"Starting training for {epochs} epochs...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X_train, y_train_list, X_val, y_val_list = self.prepare_data(data)
        
        # Create callbacks
        callback_list = self.create_callbacks(patience)
        
        # Train model
        start_time = datetime.now()
        
        self.history = self.model.fit(
            X_train, y_train_list,
            validation_data=(X_val, y_val_list),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\nTraining completed in {training_time}")
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, 'model.h5')
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return self.history
    
    def plot_training_curves(self, save_path: str = None) -> None:
        """
        Plot and save training/validation curves.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        if save_path is None:
            save_path = os.path.join(self.model_dir, 'training_curves.png')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Average accuracy across all character positions
        train_acc_keys = [key for key in self.history.history.keys() if 'accuracy' in key and 'val_' not in key]
        val_acc_keys = [key for key in self.history.history.keys() if 'val_' in key and 'accuracy' in key]
        
        if train_acc_keys and val_acc_keys:
            # Calculate average accuracy
            train_acc_avg = np.mean([self.history.history[key] for key in train_acc_keys], axis=0)
            val_acc_avg = np.mean([self.history.history[key] for key in val_acc_keys], axis=0)
            
            axes[0, 1].plot(train_acc_avg, label='Training Accuracy', color='blue')
            axes[0, 1].plot(val_acc_avg, label='Validation Accuracy', color='red')
            axes[0, 1].set_title('Average Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'], label='Learning Rate', color='green')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Individual character accuracies
        if len(train_acc_keys) >= 2:
            for i, (train_key, val_key) in enumerate(zip(train_acc_keys[:3], val_acc_keys[:3])):
                axes[1, 1].plot(self.history.history[train_key], 
                               label=f'Char {i+1} Train', alpha=0.7)
                axes[1, 1].plot(self.history.history[val_key], 
                               label=f'Char {i+1} Val', alpha=0.7, linestyle='--')
            
            axes[1, 1].set_title('Individual Character Accuracies')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Individual Character\nAccuracies Not Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to: {save_path}")
    
    def evaluate_model(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            data: Preprocessed data dictionary
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Train or load a model first.")
        
        print("Evaluating model on test data...")
        
        X_test = data['X_test']
        X_test = np.expand_dims(X_test, axis=-1)
        
        # Prepare test labels
        if data['encoding_type'] == 'onehot':
            y_test = data['y_test']
        else:
            y_test_int = data['y_test']
            y_test = np.zeros((y_test_int.shape[0], self.captcha_length, self.num_classes))
            for i in range(self.captcha_length):
                y_test[:, i, :] = keras.utils.to_categorical(y_test_int[:, i], self.num_classes)
        
        y_test_list = [y_test[:, i, :] for i in range(self.captcha_length)]
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test_list, verbose=1)
        
        # Parse results
        metrics = {}
        metrics['overall_loss'] = results[0]
        
        # Individual character accuracies
        for i in range(self.captcha_length):
            metrics[f'char_{i+1}_accuracy'] = results[i + 1 + self.captcha_length]
        
        # Calculate sequence accuracy (all characters correct)
        predictions = self.model.predict(X_test)
        pred_chars = np.array([np.argmax(pred, axis=1) for pred in predictions]).T
        true_chars = np.array([np.argmax(y_test[:, i, :], axis=1) for i in range(self.captcha_length)]).T
        
        sequence_accuracy = np.mean(np.all(pred_chars == true_chars, axis=1))
        metrics['sequence_accuracy'] = sequence_accuracy
        
        print(f"\nEvaluation Results:")
        print(f"  - Overall Loss: {metrics['overall_loss']:.4f}")
        print(f"  - Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
        for i in range(self.captcha_length):
            print(f"  - Character {i+1} Accuracy: {metrics[f'char_{i+1}_accuracy']:.4f}")
        
        return metrics
    
    def load_model(self, model_path: str) -> None:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    
    def save_training_summary(self, metrics: Dict[str, float] = None) -> None:
        """
        Save training summary and metadata.
        
        Args:
            metrics: Evaluation metrics to include
        """
        summary = {
            'model_architecture': {
                'input_shape': (self.image_height, self.image_width, 1),
                'captcha_length': self.captcha_length,
                'num_classes': self.num_classes,
                'total_parameters': self.model.count_params() if self.model else None
            },
            'training_completed': datetime.now().isoformat(),
            'metrics': metrics or {}
        }
        
        if self.history:
            summary['training_history'] = {
                'epochs_completed': len(self.history.history['loss']),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'best_val_loss': float(min(self.history.history['val_loss']))
            }
        
        summary_path = os.path.join(self.model_dir, 'training_summary.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"Training summary saved to: {summary_path}")


def main():
    """
    Main function for training the CAPTCHA CNN model.
    """
    parser = argparse.ArgumentParser(description='Train CAPTCHA CNN model')
    parser.add_argument('--data-file', type=str, default='dataset/preprocessed/preprocessed_data_onehot.npz',
                       help='Path to preprocessed data file')
    parser.add_argument('--model-dir', type=str, default='model',
                       help='Directory to save model files')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate for regularization')
    
    args = parser.parse_args()
    
    # Check for GPU availability
    print("GPU availability:")
    print(f"  - TensorFlow version: {tf.__version__}")
    print(f"  - GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    if tf.config.list_physical_devices('GPU'):
        print(f"  - GPU devices: {tf.config.list_physical_devices('GPU')}")
    else:
        print("  - No GPU detected, using CPU")
    
    # Load preprocessed data
    print(f"\nLoading data from: {args.data_file}")
    data = np.load(args.data_file, allow_pickle=True)
    data_dict = {key: data[key] for key in data.files}
    
    # Create model
    model_trainer = CaptchaCNNModel(
        image_height=int(data_dict['image_shape'][0]),
        image_width=int(data_dict['image_shape'][1]),
        captcha_length=int(data_dict['captcha_length']),
        num_classes=int(data_dict['num_classes']),
        model_dir=args.model_dir
    )
    
    # Build and train model
    model_trainer.build_model(dropout_rate=args.dropout)
    history = model_trainer.train(
        data=data_dict,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )
    
    # Plot training curves
    model_trainer.plot_training_curves()
    
    # Evaluate model
    metrics = model_trainer.evaluate_model(data_dict)
    
    # Save training summary
    model_trainer.save_training_summary(metrics)
    
    print("\n=== Training Complete! ===")
    print("Generated files:")
    print(f"1. {args.model_dir}/model.h5 (final model)")
    print(f"2. {args.model_dir}/best_model.h5 (best model during training)")
    print(f"3. {args.model_dir}/training_curves.png (training plots)")
    print(f"4. {args.model_dir}/training_summary.pkl (training metadata)")
    print("\nReady for inference and demo phase!")


if __name__ == "__main__":
    main()
