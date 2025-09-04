# AI Reverse CAPTCHA Solver 🤖

**Advanced CNN-based system that automatically solves CAPTCHAs humans struggle with**

Built for engineering day demo - showcasing the power of deep learning over traditional OCR methods.

## 🎯 Project Overview

This project demonstrates how modern AI can outperform humans at solving distorted CAPTCHAs through:
- **Synthetic Dataset Generation** (5,000+ samples with realistic distortions)
- **Advanced CNN Architecture** (Multi-character recognition with confidence scoring)
- **Performance Benchmarking** (AI vs Traditional OCR comparison)
- **Interactive Demo** (Modern web interface for live testing)

## 📁 Project Structure

```
RVR AI CAPTCHA SOLVER/
├── 📂 dataset/                    # Dataset generation and preprocessing
│   ├── __init__.py
│   ├── generate_captcha_dataset.py    # CAPTCHA image generation
│   ├── preprocess_captcha_data.py     # Data preprocessing pipeline
│   ├── images/                        # Generated CAPTCHA images (5000+)
│   ├── preprocessed/                  # Processed data (.npz files)
│   └── labels.csv                     # Ground truth labels
├── 📂 model/                      # AI model and inference
│   ├── __init__.py
│   ├── captcha_cnn_model.py          # CNN architecture and training
│   ├── captcha_inference.py          # Inference system
│   ├── ocr_comparison.py             # OCR baseline comparison
│   ├── model.h5                      # Trained model weights
│   ├── best_model.h5                 # Best checkpoint
│   └── training_curves.png           # Training visualization
├── 📂 app/                        # Demo web interface
│   ├── __init__.py
│   └── captcha_demo.py               # Streamlit demo app
├── 📂 results/                    # Output results and analysis
│   ├── inference_results_*.csv       # Prediction results
│   └── ocr_comparison_*.csv          # OCR vs CNN comparison
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # This file
└── 🚀 Run Scripts:
    ├── run_dataset_generation.py     # Generate CAPTCHA dataset
    ├── run_preprocessing.py          # Preprocess images
    ├── run_model_training.py         # Train CNN model
    ├── run_inference.py              # Run predictions
    ├── run_ocr_comparison.py         # Compare with OCR
    └── run_demo.py                   # Launch web demo
```

## 🚀 Quick Start

### Option A: Complete Setup (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python run_dataset_generation.py

# 3. Preprocess data
python run_preprocessing.py

# 4. Train model (use Google Colab for GPU - see training section)
python run_model_training.py

# 5. Run inference
python run_inference.py

# 6. Compare with OCR (requires Tesseract installation)
python run_ocr_comparison.py

# 7. Launch interactive demo
python run_demo.py
```

### Option B: Demo Only (Pre-trained Model)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample dataset
python run_dataset_generation.py

# 3. Place pre-trained model in model/ directory as 'best_model.h5'

# 4. Launch demo
python run_demo.py
```

## Phase 1: Dataset Creation ✅

### Features
- Generates 5,000+ synthetic CAPTCHA images
- 5-character alphanumeric CAPTCHAs (A-Z, 0-9)
- Random distortions: rotation, blur, brightness, contrast
- Noise injection for realism
- Modular and reusable code structure

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate dataset:**
   ```bash
   python run_dataset_generation.py
   ```

3. **Custom generation:**
   ```bash
   python dataset/generate_captcha_dataset.py --samples 10000 --verify
   ```

### Dataset Details
- **Image format:** PNG, 200x80 pixels
- **Characters:** Uppercase letters (A-Z) + digits (0-9)
- **Length:** 5 characters per CAPTCHA
- **Distortions:** Random rotation (±15°), blur, brightness/contrast adjustment
- **Noise:** Random pixel noise (10-30% intensity)

## Phase 2: Data Preprocessing 

**Features:**
- **OpenCV Pipeline:** Grayscale conversion, normalization (0-1), resizing (100x40)
- **Label Encoding:** Integer and one-hot encoding support
- **Dataset Splitting:** 70% train, 10% validation, 20% test
- **Efficient Storage:** Compressed .npz format for fast loading
- **Flexible:** Configurable image dimensions and encoding types

**Quick Start:**
```bash
python run_preprocessing.py
```

**Output:**
- **Files:** `dataset/preprocessed/preprocessed_data_*.npz`
- **Size:** ~15MB per encoding type
- **Splits:** 70% train, 10% validation, 20% test
- **Storage:** Compressed .npz format (~15MB)

## Tech Stack
- **Python 3.x**
- **Libraries:** captcha, Pillow, NumPy, OpenCV, TensorFlow, Streamlit
- **Training:** Google Colab (GPU)
- **Inference:** CPU-optimized for modest hardware

## Phase 4: Inference & Testing ✅

**Inference Features:**
- **Model Loading:** Automatic loading of trained CNN model (model.h5)
- **Image Processing:** Same preprocessing pipeline as training
- **Predictions:** Multi-character CAPTCHA text prediction with confidence scores
- **Accuracy Analysis:** Character-wise and sequence-level accuracy metrics
- **Batch Processing:** Efficient processing of multiple images
- **Results Export:** Detailed CSV output with predictions and ground truth comparison

**Key Capabilities:**
- **Confidence Scoring:** Individual character and overall confidence metrics
- **Ground Truth Comparison:** Automatic accuracy calculation when labels available
- **Performance Metrics:** Sequence accuracy, character accuracy, position-wise analysis
- **Error Handling:** Robust processing with detailed error reporting
- **Flexible Input:** Support for various image formats (PNG, JPG, etc.)

**Quick Start:**
```bash
python run_inference.py
```

**Output:**
- **Results CSV:** `results/inference_results_*.csv` (detailed predictions)
- **Metrics:** Sequence accuracy, character accuracy, confidence scores
- **Analysis:** Character-wise performance breakdown

## Phase 5: OCR Baseline Comparison ✅

**Comparison Features:**
- **Tesseract OCR Integration:** Complete pytesseract implementation with optimized preprocessing
- **Image Enhancement:** Upscaling, denoising, thresholding, and morphological operations for OCR
- **Performance Benchmarking:** Side-by-side accuracy comparison between CNN and traditional OCR
- **Detailed Analysis:** Character-wise and sequence-level accuracy metrics for both approaches
- **Formatted Results:** Professional comparison tables with improvement calculations

**OCR Preprocessing Pipeline:**
- **Image Upscaling:** 3x scale factor for better character recognition
- **Noise Reduction:** Gaussian blur and morphological operations
- **Binarization:** OTSU thresholding for clean text extraction
- **Character Filtering:** Alphanumeric whitelist (A-Z, 0-9) matching CAPTCHA format

**Quick Start:**
```bash
# Install Tesseract OCR first (Windows: https://github.com/UB-Mannheim/tesseract/wiki)
pip install pytesseract tabulate
python run_ocr_comparison.py
```

**Output:**
- **Comparison Table:** Formatted performance comparison (CNN vs OCR)
- **Detailed CSV:** `results/ocr_comparison_*.csv` (per-image analysis)
- **Metrics:** Sequence accuracy, character accuracy, confidence scores
- **Improvement Analysis:** Quantified CNN performance gains over traditional OCR

## Phase 6: Frontend Demo App ✅

**Modern Streamlit Interface:**
- **Sleek Design:** Industry-standard UI with gradient backgrounds, modern typography, and smooth animations
- **Dual Input Methods:** Upload custom images or select from test dataset samples
- **Real-time AI Prediction:** Instant CAPTCHA solving with confidence scores and timing metrics
- **Interactive Features:** Human vs AI challenge with built-in stopwatch and performance comparison
- **Professional Layout:** Clean card-based design with responsive columns and modern CSS styling

**Key Features:**
- **Image Upload:** Drag-and-drop interface for custom CAPTCHA images
- **Test Dataset Integration:** Random selection from generated CAPTCHA samples
- **AI Performance Display:** Prediction results with character-wise confidence visualization
- **Speed Comparison:** Millisecond-precision timing for AI vs human performance
- **Accuracy Analysis:** Real-time correctness validation against ground truth
- **Interactive Stopwatch:** Human challenge mode with start/stop timer functionality

**UI/UX Design:**
- **Modern Gradients:** Professional color schemes with smooth transitions
- **Typography:** Inter font family for clean, readable text
- **Responsive Layout:** Optimized for desktop and mobile viewing
- **Visual Feedback:** Success/error states with appropriate color coding
- **Performance Metrics:** Real-time charts and progress bars

**Quick Start:**
```bash
streamlit run app/captcha_demo.py
# or
python run_demo.py
```

**Demo Features:**
- **Live Prediction:** Real-time CAPTCHA solving with sub-second response times
- **Confidence Visualization:** Character-by-character confidence bars
- **Human Challenge:** Interactive stopwatch for speed comparison
- **Performance Analytics:** Detailed metrics and accuracy analysis

## 🎓 Training Instructions

### Google Colab Training (Recommended for GPU)

1. **Upload project to Google Drive:**
   ```bash
   # Zip the entire project folder
   # Upload to Google Drive
   ```

2. **Open Google Colab and mount Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Navigate to project
   %cd /content/drive/MyDrive/RVR_AI_CAPTCHA_SOLVER
   
   # Install dependencies
   !pip install -r requirements.txt
   ```

3. **Run training pipeline:**
   ```python
   # Generate dataset
   !python run_dataset_generation.py
   
   # Preprocess data
   !python run_preprocessing.py
   
   # Train model (will use GPU automatically)
   !python run_model_training.py
   ```

4. **Download trained model:**
   ```python
   # Download model files to local machine
   from google.colab import files
   files.download('model/best_model.h5')
   files.download('model/training_curves.png')
   ```

### Local Training (CPU Only)
```bash
# For modest hardware - will be slower but works
python run_model_training.py --epochs 20 --batch-size 16
```

## 💻 Local Inference Setup

### System Requirements
- **Minimum:** Intel i3 or equivalent, 4GB RAM
- **Recommended:** Intel i5 or equivalent, 8GB RAM
- **OS:** Windows 10/11, macOS, or Linux
- **Python:** 3.8 or higher

### CPU-Only Inference
The system is optimized for CPU inference on modest hardware:
```bash
# Install CPU-only TensorFlow (lighter)
pip install tensorflow-cpu==2.13.0

# Or use regular TensorFlow (works on CPU too)
pip install -r requirements.txt

# Run inference (automatically uses CPU)
python run_inference.py
```

## 🔧 Tesseract OCR Setup

### Windows
1. Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR\`
3. Add to PATH or set environment variable
4. Install Python wrapper: `pip install pytesseract`

### macOS
```bash
brew install tesseract
pip install pytesseract
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract
```

## 📊 Expected Results

### Model Performance
- **Training Time:** 30-60 minutes on GPU (Colab)
- **Sequence Accuracy:** 70-90% (all 5 characters correct)
- **Character Accuracy:** 85-95% (individual characters)
- **Inference Speed:** 50-200ms per image (CPU)

### OCR Comparison
- **CNN Model:** 70-90% sequence accuracy
- **Tesseract OCR:** 10-40% sequence accuracy
- **Speed:** CNN is typically 2-5x faster than OCR

## 🚨 Troubleshooting

### Common Issues

1. **"Model not found" error:**
   ```bash
   # Ensure model is trained first
   python run_model_training.py
   # Or place pre-trained model.h5 in model/ directory
   ```

2. **"No dataset found" error:**
   ```bash
   # Generate dataset first
   python run_dataset_generation.py
   ```

3. **Tesseract not found:**
   ```bash
   # Install Tesseract OCR (see setup section above)
   # Or skip OCR comparison
   ```

4. **Memory issues on modest hardware:**
   ```bash
   # Reduce batch size
   python run_model_training.py --batch-size 8
   # Or use fewer images
   python run_inference.py --max-images 100
   ```

5. **Streamlit app won't start:**
   ```bash
   # Install Streamlit
   pip install streamlit
   # Check port availability
   streamlit run app/captcha_demo.py --server.port 8502
   ```

## 🎯 Demo Day Checklist

### Before Presentation:
- [ ] ✅ Dataset generated (5,000 images)
- [ ] ✅ Model trained (best_model.h5 exists)
- [ ] ✅ Demo app tested (`python run_demo.py`)
- [ ] ✅ OCR comparison ready (Tesseract installed)
- [ ] ✅ Sample predictions verified

### During Demo:
1. **Show dataset:** Display generated CAPTCHA samples
2. **Launch demo:** `python run_demo.py` → Opens in browser
3. **Live prediction:** Upload/select CAPTCHA → AI solves instantly
4. **Human challenge:** Let audience try to beat the AI
5. **Show comparison:** Display CNN vs OCR performance table
6. **Highlight speed:** Emphasize sub-second AI response times

### Key Talking Points:
- 🤖 **AI vs Human:** "AI solves in 100ms, humans take 5-10 seconds"
- 📊 **Accuracy:** "90% AI accuracy vs 30% traditional OCR"
- 🚀 **Technology:** "Deep learning CNN with 4 convolutional layers"
- 💻 **Efficiency:** "Runs on modest laptop, no cloud required"
- 🎯 **Real-world:** "Demonstrates AI superiority over traditional methods"

## Project Complete! 🎉

All phases successfully implemented:
✅ **Phase 1:** Dataset Generation (5,000 synthetic CAPTCHAs)
✅ **Phase 2:** Data Preprocessing (OpenCV pipeline)  
✅ **Phase 3:** Model Training (CNN architecture)
✅ **Phase 4:** Inference & Testing (Complete prediction system)
✅ **Phase 5:** OCR Baseline Comparison (CNN vs Tesseract benchmarking)
✅ **Phase 6:** Frontend Demo App (Modern Streamlit interface)
✅ **Phase 7:** Final Packaging (Complete setup and documentation)

---
*Built for engineering day demo - easy setup and impressive results!*
