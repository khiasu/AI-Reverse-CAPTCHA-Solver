# 🤖 REVERSE AI CAPTCHA SOLVER

**Advanced AI-Powered CAPTCHA Recognition System**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-85%25+-brightgreen.svg)](#performance)

A Convolutional Neural Network (CNN) system that automatically recognizes and solves text-based CAPTCHAs with high accuracy. Features real-time human vs AI comparison.

## 🏗️ System Architecture

### Core Components

| Component | Description | Key Technologies |
|-----------|-------------|------------------|
| **Data Pipeline** | Automated preprocessing and augmentation | OpenCV, PIL, NumPy |
| **CNN Architecture** | Multi-output deep learning model | TensorFlow, Keras |
| **Web Application** | Interactive demo and API | Flask, JavaScript |
| **Optimization** | Model compression and acceleration | TensorFlow Lite, Mixed Precision |

## 🚀 Quick Start Guide

### 📋 Prerequisites

```bash
# Check Python version (3.8+ required)
python --version

# Clone the repository
git clone https://github.com/your-username/rvr-ai-captcha-solver.git
cd rvr-ai-captcha-solver

# Install dependencies
pip install -r requirements.txt
```

### 🔧 Setup & Training

1. **Prepare Training Data**
   ```bash
   # Preprocess CAPTCHA images and create training dataset
   python run_preprocessing.py
   ```

2. **Train the Model** (Choose your preferred method)
   
   **Standard Training:**
   ```bash
   python run_model_training.py
   ```
   
   **Enhanced Training (Recommended):**
   ```bash
   # With data augmentation and optimization
   python run_optimized_training.py --quantize
   
   # Custom configuration
   python run_optimized_training.py --epochs 100 --batch-size 64 --quantize
   ```

3. **Launch Web Application**
   ```bash
   python app.py
   ```

4. **Access Demo**: Open `http://localhost:5000` in your browser

### ⚡ Advanced Usage

```bash
# Training with specific optimizations
python run_optimized_training.py \
    --epochs 50 \
    --batch-size 32 \
    --patience 15 \
    --quantize

# Disable specific features if needed
python run_optimized_training.py \
    --no-augmentation \
    --no-mixed-precision
```

### CNN Architecture Details

```python
Model Configuration:
├── Input Layer: (40, 100, 1) - Grayscale CAPTCHA images
├── Convolutional Stack:
│   ├── Conv2D(32) + BatchNorm + ReLU + MaxPool
│   ├── Conv2D(64) + BatchNorm + ReLU + MaxPool  
│   ├── Conv2D(128) + BatchNorm + ReLU + MaxPool
│   └── Conv2D(256) + BatchNorm + ReLU
├── Feature Processing:
│   ├── Flatten + Dropout(0.5)
│   └── Dense(512) + ReLU + Dropout(0.5)
└── Output Heads: 5x Dense(36) [Softmax] - One per character position

Total Parameters: ~2.1M
Trainable Parameters: ~2.1M
```

## 📁 Project Structure

```
RVR-AI-CAPTCHA-SOLVER/
├── 📄 app.py                          # Main Flask web application
├── 📄 requirements.txt                # Python dependencies
├── 📄 run_preprocessing.py            # Data preparation pipeline
├── 📄 run_model_training.py           # Standard model training
├── 📄 run_optimized_training.py       # Enhanced training with optimizations
├── 📁 data_preprocessing/             # Data processing modules
│   ├── captcha_processor.py          # Image preprocessing
│   ├── data_augmentation.py          # Training data augmentation
│   └── dataset_builder.py            # Dataset creation utilities
├── 📁 model/                          # Neural network components
│   ├── captcha_cnn_model.py          # Main CNN architecture
│   ├── optimized_training.py         # Advanced training optimizations
│   └── model_utils.py                # Model utilities and helpers
├── 📁 app/                           # Web interface components
│   ├── static/                       # CSS, JS, images
│   │   ├── css/style.css             # Main stylesheet
│   │   ├── js/main.js                # Interactive functionality
│   │   └── images/                   # UI assets
│   └── templates/                    # HTML templates
│       └── index.html                # Main interface
├── 📁 dataset/                       # Training data
│   ├── raw_captchas/                 # Original CAPTCHA images
│   └── preprocessed/                 # Processed training data
├── 📁 evaluation/                    # Performance analysis
│   ├── model_evaluator.py           # Comprehensive model testing
│   └── performance_analyzer.py       # Metrics and benchmarking
├── 📁 deployment/                    # Deployment configurations
│   ├── Dockerfile                    # Docker container setup
│   ├── requirements-prod.txt         # Production dependencies
│   └── deploy_guides/               # Platform-specific deployment guides
└── 📄 README.md                     # This comprehensive guide
```

## 🔧 Configuration Options

### Training Parameters

```python
# Core training settings
EPOCHS = 50              # Number of training epochs
BATCH_SIZE = 32          # Training batch size  
PATIENCE = 15            # Early stopping patience
LEARNING_RATE = 1e-3     # Initial learning rate

# Optimization features
USE_AUGMENTATION = True   # Enable data augmentation
USE_MIXED_PRECISION = True # Enable FP16 training
QUANTIZE_MODEL = True    # Apply post-training quantization

# Model architecture
DROPOUT_RATE = 0.5       # Dropout for regularization
CAPTCHA_LENGTH = 5       # Number of characters
NUM_CLASSES = 36         # A-Z + 0-9
```

### Web App Configuration

```python
# Flask app settings
DEBUG = False            # Production mode
HOST = '0.0.0.0'        # Accept external connections
PORT = 5000             # Default port

# Model settings
MODEL_PATH = 'model/best_model.keras'  # Trained model location
MAX_FILE_SIZE = 5 * 1024 * 1024        # 5MB upload limit
```

## 🐛 Troubleshooting

### Common Issues

**Training Issues:**
```bash
# Memory error during training
python run_optimized_training.py --batch-size 16

# Slow training on CPU
python run_optimized_training.py --no-mixed-precision
```

**Model Loading Issues:**
```python
# If you encounter keras model loading issues
import tensorflow as tf
model = tf.keras.models.load_model('model/best_model.h5', compile=False)
```

**Web App Issues:**
```bash
# Port already in use
python app.py --port 5001

# Model not found
# Ensure training completed successfully and model files exist
ls -la model/
```
---

<div align="center">



*Built with ❤️ for the AI/ML enthusiast & engineering day showcase🤓*

</div>
