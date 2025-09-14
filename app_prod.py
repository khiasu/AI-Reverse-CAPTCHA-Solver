"""
Production-ready Flask application for RVR AI CAPTCHA SOLVER.
Includes monitoring, logging, error handling, and optimization.
"""

import os
import time
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter('captcha_predictions_total', 'Total CAPTCHA predictions')
PREDICTION_HISTOGRAM = Histogram('captcha_prediction_duration_seconds', 'CAPTCHA prediction duration')
ERROR_COUNTER = Counter('captcha_errors_total', 'Total CAPTCHA prediction errors', ['error_type'])

class CaptchaPredictor:
    """Thread-safe CAPTCHA prediction service."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: Optional[tf.keras.Model] = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model."""
        try:
            if os.path.exists(self.model_path):
                logger.info("Loading CAPTCHA model", model_path=self.model_path)
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                self.model_loaded = True
                logger.info("Model loaded successfully", 
                           model_params=self.model.count_params() if self.model else 0)
                return True
            else:
                logger.warning("Model file not found", model_path=self.model_path)
                return False
        except Exception as e:
            logger.error("Failed to load model", error=str(e), traceback=traceback.format_exc())
            return False
    
    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Make prediction on preprocessed image."""
        if not self.model_loaded or self.model is None:
            raise ValueError("Model not loaded")
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        
        # Process results
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
        
        return {
            'predicted_text': predicted_text,
            'confidence_scores': confidence_scores,
            'average_confidence': float(np.mean(confidence_scores))
        }

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config.update(
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5MB max file size
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
    UPLOAD_FOLDER='uploads',
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif', 'bmp'},
    MODEL_PATH=os.environ.get('MODEL_PATH', 'model/best_model.h5'),
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
predictor = CaptchaPredictor(app.config['MODEL_PATH'])

# Application metrics
app_start_time = time.time()
prediction_count = 0
total_prediction_time = 0.0

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for prediction."""
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to expected dimensions
    image = image.resize((100, 40), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(image).reshape(1, 40, 100, 1).astype(np.float32) / 255.0
    
    return image_array

@app.route('/')
def index():
    """Main application page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    uptime = time.time() - app_start_time
    
    health_status = {
        'status': 'healthy' if predictor.model_loaded else 'degraded',
        'timestamp': datetime.utcnow().isoformat(),
        'uptime_seconds': uptime,
        'model_loaded': predictor.model_loaded,
        'model_path': app.config['MODEL_PATH'],
        'predictions_served': prediction_count,
        'average_prediction_time': total_prediction_time / max(prediction_count, 1)
    }
    
    status_code = 200 if predictor.model_loaded else 503
    return jsonify(health_status), status_code

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/predict', methods=['POST'])
def predict():
    """CAPTCHA prediction endpoint."""
    global prediction_count, total_prediction_time
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info("Prediction request started", request_id=request_id)
    
    try:
        # Validate request
        if 'file' not in request.files:
            ERROR_COUNTER.labels(error_type='no_file').inc()
            return jsonify({'error': 'No file uploaded', 'request_id': request_id}), 400
        
        file = request.files['file']
        if file.filename == '':
            ERROR_COUNTER.labels(error_type='empty_filename').inc()
            return jsonify({'error': 'No file selected', 'request_id': request_id}), 400
        
        if not allowed_file(file.filename):
            ERROR_COUNTER.labels(error_type='invalid_file_type').inc()
            return jsonify({'error': 'Invalid file type', 'request_id': request_id}), 400
        
        # Check if model is loaded
        if not predictor.model_loaded:
            ERROR_COUNTER.labels(error_type='model_not_loaded').inc()
            logger.warning("Prediction attempted with no model loaded", request_id=request_id)
            # Return demo prediction
            return jsonify({
                'predicted_text': 'DEMO1',
                'confidence_scores': [0.95, 0.92, 0.88, 0.91, 0.94],
                'average_confidence': 0.92,
                'prediction_time': time.time() - start_time,
                'request_id': request_id,
                'note': 'Demo mode - model not loaded'
            })
        
        # Process image
        try:
            image = Image.open(file.stream)
            image_array = preprocess_image(image)
        except Exception as e:
            ERROR_COUNTER.labels(error_type='image_processing').inc()
            logger.error("Image processing failed", 
                        request_id=request_id, error=str(e))
            return jsonify({'error': 'Invalid image format', 'request_id': request_id}), 400
        
        # Make prediction
        try:
            prediction_result = predictor.predict(image_array)
        except Exception as e:
            ERROR_COUNTER.labels(error_type='prediction').inc()
            logger.error("Prediction failed", 
                        request_id=request_id, error=str(e))
            return jsonify({'error': 'Prediction failed', 'request_id': request_id}), 500
        
        # Calculate timing
        prediction_time = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_HISTOGRAM.observe(prediction_time)
        prediction_count += 1
        total_prediction_time += prediction_time
        
        # Prepare response
        response = {
            **prediction_result,
            'prediction_time': prediction_time,
            'request_id': request_id
        }
        
        logger.info("Prediction completed successfully", 
                   request_id=request_id,
                   predicted_text=prediction_result['predicted_text'],
                   avg_confidence=prediction_result['average_confidence'],
                   prediction_time=prediction_time)
        
        return jsonify(response)
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type='unexpected').inc()
        logger.error("Unexpected error in prediction", 
                    request_id=request_id, 
                    error=str(e), 
                    traceback=traceback.format_exc())
        return jsonify({
            'error': 'Internal server error', 
            'request_id': request_id
        }), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('app/static', filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    ERROR_COUNTER.labels(error_type='file_too_large').inc()
    return jsonify({'error': 'File too large (max 5MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    ERROR_COUNTER.labels(error_type='server_error').inc()
    logger.error("Internal server error", error=str(e))
    return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def log_request():
    """Log incoming requests."""
    logger.info("Request received", 
               method=request.method, 
               path=request.path,
               remote_addr=request.remote_addr,
               user_agent=request.headers.get('User-Agent', 'Unknown'))

if __name__ == '__main__':
    # Configure logging level
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(level=getattr(logging, log_level))
    
    # Run application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info("Starting CAPTCHA solver application", 
               port=port, 
               debug=debug,
               model_loaded=predictor.model_loaded)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
