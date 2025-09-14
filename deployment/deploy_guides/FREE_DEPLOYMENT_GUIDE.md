# 🚀 Free Deployment Guide for RVR AI CAPTCHA SOLVER

This comprehensive guide covers deploying your CAPTCHA recognition system on free platforms. Choose the option that best fits your needs!

## 🎯 Quick Platform Comparison

| Platform | Best For | Free Tier | Difficulty | Notes |
|----------|----------|-----------|------------|--------|
| **Vercel** | Static frontend + API | Generous | ⭐⭐ | Best for web demos |
| **Railway** | Full-stack apps | 500 hours/month | ⭐⭐⭐ | Excellent for ML models |
| **Render** | Web services | 750 hours/month | ⭐⭐⭐ | Good performance |
| **Hugging Face Spaces** | ML demos | Unlimited | ⭐⭐ | Perfect for AI showcases |
| **Google Cloud Run** | Serverless containers | 2M requests/month | ⭐⭐⭐⭐ | Production-ready |

## 🔥 Option 1: Vercel (Recommended for Demos)

**Perfect for:** Quick demos, portfolio showcases
**Pros:** Lightning fast, great UI, simple deployment
**Cons:** Limited to serverless functions

### Step-by-Step Deployment

#### 1. Prepare Your Project

```bash
# Create Vercel-compatible structure
mkdir vercel-deploy
cd vercel-deploy

# Copy essential files
cp -r app/static ./static
cp -r app/templates ./templates
cp app.py ./api/predict.py  # Convert to Vercel function
```

#### 2. Create Vercel Configuration

Create `vercel.json`:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/predict.py",
      "use": "@vercel/python"
    },
    {
      "src": "static/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/predict.py"
    },
    {
      "src": "/(.*)",
      "dest": "/static/$1"
    }
  ],
  "env": {
    "PYTHON_VERSION": "3.9"
  }
}
```

#### 3. Create Serverless Function

Create `api/predict.py`:
```python
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify

# Load model (download from your trained model)
model = None

def load_model():
    global model
    if model is None:
        # Download model from GitHub releases or cloud storage
        model_url = "https://your-storage-url/model.h5"
        model = tf.keras.models.load_model(model_url)
    return model

def handler(request):
    if request.method == 'POST':
        try:
            # Get image from request
            file = request.files['file']
            
            # Process image
            image = Image.open(file.stream).convert('L')
            image = image.resize((100, 40))
            image_array = np.array(image).reshape(1, 40, 100, 1) / 255.0
            
            # Load model and predict
            model = load_model()
            predictions = model.predict(image_array)
            
            # Process predictions
            predicted_text = ""
            confidence_scores = []
            
            for i in range(5):
                char_pred = predictions[i][0]
                char_idx = np.argmax(char_pred)
                confidence = float(np.max(char_pred))
                confidence_scores.append(confidence)
                
                # Convert index to character
                if char_idx < 10:
                    char = str(char_idx)
                else:
                    char = chr(char_idx - 10 + ord('A'))
                predicted_text += char
            
            return jsonify({
                'predicted_text': predicted_text,
                'confidence_scores': confidence_scores,
                'average_confidence': float(np.mean(confidence_scores))
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'message': 'CAPTCHA Recognition API'}), 200
```

#### 4. Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod

# Your app will be live at: https://your-project.vercel.app
```

---

## 🚂 Option 2: Railway (Best for Full ML Apps)

**Perfect for:** Full-featured applications with persistent storage
**Pros:** Supports large ML models, databases, persistent storage
**Cons:** Limited free hours (500/month)

### Step-by-Step Deployment

#### 1. Prepare Railway Configuration

Create `railway.toml`:
```toml
[build]
builder = "nixpacks"

[deploy]
healthcheckPath = "/health"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[env]
PORT = "8000"
PYTHON_VERSION = "3.9"
```

#### 2. Create Production Requirements

Create `requirements-prod.txt`:
```txt
flask==2.3.3
tensorflow-cpu==2.13.0  # Use CPU version for cost efficiency
opencv-python-headless==4.8.0.76
pillow==10.0.0
numpy==1.24.3
gunicorn==21.2.0
```

#### 3. Create Production App

Create `app_prod.py`:
```python
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time

app = Flask(__name__)

# Configure for production
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Load model on startup
MODEL_PATH = os.path.join('model', 'best_model.h5')
if os.path.exists(MODEL_PATH):
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print("Warning: Model not found, using dummy predictions")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'healthy', 'model_loaded': model is not None}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process image
        image = Image.open(file.stream).convert('L')
        image = image.resize((100, 40))
        image_array = np.array(image).reshape(1, 40, 100, 1) / 255.0
        
        if model is None:
            # Return dummy prediction for demo
            return jsonify({
                'predicted_text': 'DEMO1',
                'confidence_scores': [0.95, 0.92, 0.88, 0.91, 0.94],
                'average_confidence': 0.92,
                'prediction_time': time.time() - start_time,
                'note': 'Demo mode - train your model first!'
            })
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Process results
        predicted_text = ""
        confidence_scores = []
        
        for i in range(5):
            char_pred = predictions[i][0]
            char_idx = np.argmax(char_pred)
            confidence = float(np.max(char_pred))
            confidence_scores.append(confidence)
            
            if char_idx < 10:
                char = str(char_idx)
            else:
                char = chr(char_idx - 10 + ord('A'))
            predicted_text += char
        
        prediction_time = time.time() - start_time
        
        return jsonify({
            'predicted_text': predicted_text,
            'confidence_scores': confidence_scores,
            'average_confidence': float(np.mean(confidence_scores)),
            'prediction_time': prediction_time
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

#### 4. Create Procfile

Create `Procfile`:
```
web: gunicorn app_prod:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

#### 5. Deploy to Railway

```bash
# Install Railway CLI
curl -fsSL https://railway.app/install.sh | sh

# Login
railway login

# Initialize project
railway init

# Deploy
railway up

# Your app will be live at: https://your-project.railway.app
```

---

## 🤗 Option 3: Hugging Face Spaces (Perfect for AI Demos)

**Perfect for:** AI/ML showcases, academic projects
**Pros:** Free, unlimited usage, great for ML community
**Cons:** Public only (no private deployments on free tier)

### Step-by-Step Deployment

#### 1. Create Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Gradio" as the SDK
4. Name your space (e.g., "captcha-solver-demo")

#### 2. Create Gradio App

Create `app.py`:
```python
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Download model if not exists
MODEL_URL = "https://huggingface.co/your-username/captcha-model/resolve/main/model.h5"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def solve_captcha(image):
    """Solve CAPTCHA using trained CNN model."""
    try:
        # Preprocess image
        if image is None:
            return "Please upload an image", 0.0, "No image provided"
        
        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((100, 40))
        image_array = np.array(image).reshape(1, 40, 100, 1) / 255.0
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Process results
        predicted_text = ""
        confidence_scores = []
        
        for i in range(5):
            char_pred = predictions[i][0]
            char_idx = np.argmax(char_pred)
            confidence = float(np.max(char_pred))
            confidence_scores.append(confidence)
            
            if char_idx < 10:
                char = str(char_idx)
            else:
                char = chr(char_idx - 10 + ord('A'))
            predicted_text += char
        
        avg_confidence = float(np.mean(confidence_scores))
        
        # Create detailed output
        details = f"Individual character confidences:\\n"
        for i, (char, conf) in enumerate(zip(predicted_text, confidence_scores)):
            details += f"Position {i+1}: '{char}' ({conf:.1%})\\n"
        
        return predicted_text, avg_confidence, details
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0, "Processing failed"

# Create Gradio interface
demo = gr.Interface(
    fn=solve_captcha,
    inputs=[
        gr.Image(type="pil", label="Upload CAPTCHA Image")
    ],
    outputs=[
        gr.Textbox(label="Predicted Text", show_label=True),
        gr.Number(label="Average Confidence", show_label=True),
        gr.Textbox(label="Detailed Results", lines=6, show_label=True)
    ],
    title="🤖 AI CAPTCHA Solver",
    description="""
    ## Advanced CAPTCHA Recognition System
    
    Upload a 5-character CAPTCHA image and watch the AI solve it instantly!
    
    **Features:**
    - Real-time CNN-based recognition
    - Character-level confidence scoring
    - Handles alphanumeric CAPTCHAs (A-Z, 0-9)
    
    **Accuracy:** ~90% on standard text-based CAPTCHAs
    """,
    examples=[
        ["examples/captcha1.png"],
        ["examples/captcha2.png"],
        ["examples/captcha3.png"]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
```

#### 3. Create Requirements

Create `requirements.txt`:
```txt
gradio==4.7.1
tensorflow==2.13.0
numpy==1.24.3
pillow==10.0.0
requests==2.31.0
```

#### 4. Create README

Create `README.md`:
```markdown
---
title: AI CAPTCHA Solver
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
---

# AI CAPTCHA Solver

An advanced CNN-based CAPTCHA recognition system that can solve text-based CAPTCHAs with high accuracy.

## Features
- Real-time CAPTCHA solving
- Character-level confidence scoring
- Supports A-Z and 0-9 characters
- ~90% accuracy on standard CAPTCHAs

Built with TensorFlow and Gradio.
```

#### 5. Deploy

```bash
# Clone your space repository
git clone https://huggingface.co/spaces/your-username/captcha-solver-demo
cd captcha-solver-demo

# Add your files
cp app.py requirements.txt README.md ./

# Commit and push
git add .
git commit -m "Initial CAPTCHA solver deployment"
git push

# Your space will be live at: https://huggingface.co/spaces/your-username/captcha-solver-demo
```

---

## ☁️ Option 4: Google Cloud Run (Production Ready)

**Perfect for:** Production applications, scalable deployments
**Pros:** Excellent performance, auto-scaling, pay-per-use
**Cons:** Requires Google Cloud account, more complex setup

### Step-by-Step Deployment

#### 1. Create Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Set environment variable for port
ENV PORT=8080

# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app_prod:app
```

#### 2. Create Cloud Run Configuration

Create `cloudbuild.yaml`:
```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/captcha-solver', '.']
  
  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/captcha-solver']
  
  # Deploy container image to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'captcha-solver'
      - '--image'
      - 'gcr.io/$PROJECT_ID/captcha-solver'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '2'
      - '--max-instances'
      - '10'
```

#### 3. Deploy to Cloud Run

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Get service URL
gcloud run services describe captcha-solver --region=us-central1 --format="value(status.url)"
```

---

## 🔧 Production Tips & Optimizations

### Model Optimization for Deployment

```python
# Optimize model for production
import tensorflow as tf

def optimize_for_production(model_path, output_path):
    """Optimize model for deployment."""
    model = tf.keras.models.load_model(model_path)
    
    # Convert to SavedModel format (better for serving)
    tf.saved_model.save(model, output_path)
    
    # Create TensorFlow Lite version
    converter = tf.lite.TFLiteConverter.from_saved_model(output_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(f"{output_path}/model.tflite", "wb") as f:
        f.write(tflite_model)
    
    print(f"Optimized models saved to {output_path}")

# Usage
optimize_for_production("model/best_model.h5", "model/production")
```

### Environment Variables Setup

Create `.env` file:
```bash
# Model Configuration
MODEL_PATH=model/best_model.h5
MAX_FILE_SIZE=5242880  # 5MB in bytes
PREDICTION_TIMEOUT=30  # seconds

# Application Settings
DEBUG=false
SECRET_KEY=your-secret-key-here
UPLOAD_FOLDER=uploads

# Performance Settings
WORKERS=1
THREADS=8
TIMEOUT=120

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

### Health Checks & Monitoring

```python
@app.route('/health')
def health_check():
    """Comprehensive health check endpoint."""
    checks = {
        'status': 'healthy',
        'timestamp': time.time(),
        'model_loaded': model is not None,
        'memory_usage': get_memory_usage(),
        'disk_space': get_disk_space()
    }
    
    if model is None:
        checks['status'] = 'degraded'
        return jsonify(checks), 503
    
    return jsonify(checks), 200

@app.route('/metrics')
def metrics():
    """Basic metrics endpoint."""
    return jsonify({
        'predictions_total': prediction_counter,
        'average_prediction_time': avg_prediction_time,
        'model_accuracy': model_accuracy,
        'uptime': time.time() - start_time
    })
```

---

## 🚀 Quick Start Commands

Choose your deployment platform and run:

**Vercel:**
```bash
npm install -g vercel
vercel login
vercel --prod
```

**Railway:**
```bash
curl -fsSL https://railway.app/install.sh | sh
railway login
railway init
railway up
```

**Hugging Face:**
```bash
git clone https://huggingface.co/spaces/your-username/your-space
# Add your files and push
```

**Google Cloud Run:**
```bash
gcloud builds submit --config cloudbuild.yaml
```

## 🎯 Next Steps

1. **Test your deployment** with sample CAPTCHA images
2. **Monitor performance** using platform-specific analytics
3. **Set up custom domain** (optional, may require paid plan)
4. **Implement CI/CD** for automatic deployments
5. **Add monitoring & logging** for production use

## 💡 Pro Tips

- **Start with Vercel or Hugging Face** for quick demos
- **Use Railway or Cloud Run** for production applications
- **Optimize your model size** before deployment
- **Implement proper error handling** for better UX
- **Add rate limiting** to prevent abuse
- **Monitor costs** on paid platforms

---

**🎉 Congratulations!** Your AI CAPTCHA solver is now live on the internet! Share your deployment URL and showcase your AI engineering skills!

Need help? Check the troubleshooting section or open an issue in the repository.
