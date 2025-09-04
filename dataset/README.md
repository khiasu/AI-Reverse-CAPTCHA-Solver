# Dataset Directory

This directory contains the CAPTCHA dataset and preprocessing components.

## Structure
- `images/` - Generated CAPTCHA images (5000+ samples)
- `preprocessed/` - Processed data files (.npz format)
- `labels.csv` - Ground truth labels for all images
- `generate_captcha_dataset.py` - Dataset generation script
- `preprocess_captcha_data.py` - Data preprocessing pipeline

## Usage
```bash
# Generate dataset
python run_dataset_generation.py

# Preprocess data
python run_preprocessing.py
```

**Note:** The actual image files and preprocessed data are excluded from Git due to size constraints. Run the generation scripts to create them locally.
