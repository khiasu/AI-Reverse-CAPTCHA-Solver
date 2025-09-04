# Model Directory

This directory contains the AI model components and inference system.

## Structure
- `captcha_cnn_model.py` - CNN architecture and training pipeline
- `captcha_inference.py` - Inference system with confidence scoring
- `ocr_comparison.py` - OCR baseline comparison
- `best_model.h5` - Trained model weights (generated after training)
- `training_curves.png` - Training visualization (generated after training)

## Usage
```bash
# Train model (use Google Colab for GPU)
python run_model_training.py

# Run inference
python run_inference.py

# Compare with OCR
python run_ocr_comparison.py
```

**Note:** Model weights (.h5 files) are excluded from Git due to size. Train the model or download pre-trained weights to use the inference system.
