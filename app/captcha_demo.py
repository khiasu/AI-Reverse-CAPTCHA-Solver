"""
CAPTCHA Solver Demo App
Modern Streamlit interface for AI CAPTCHA recognition with human vs AI comparison.
"""

import streamlit as st
import os
import sys
import time
import random
import pandas as pd
from PIL import Image
import numpy as np
from datetime import datetime
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.captcha_inference import CaptchaInference


# Page configuration
st.set_page_config(
    page_title="AI CAPTCHA Solver",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, sleek design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .demo-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 1.5rem;
    }
    
    .demo-card h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.4rem;
    }
    
    /* Results styling */
    .result-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
    }
    
    .result-success h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.2em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .confidence-bar {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #0abde3);
        height: 20px;
        border-radius: 8px;
        transition: width 0.8s ease;
    }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        min-width: 120px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Stopwatch styling */
    .stopwatch {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.3);
    }
    
    .stopwatch-time {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    
    /* Comparison styling */
    .vs-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .vs-circle {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        color: #2c3e50;
        box-shadow: 0 5px 15px rgba(252, 182, 159, 0.4);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .metric-container {
            flex-direction: column;
            gap: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained CAPTCHA model."""
    # Try both model paths with absolute paths
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path1 = os.path.join(base_dir, "model", "best_model.h5")
    model_path2 = os.path.join(base_dir, "model", "model.h5")
    
    if os.path.exists(model_path1):
        return CaptchaInference(model_path1)
    elif os.path.exists(model_path2):
        return CaptchaInference(model_path2)
    return None


@st.cache_data
def load_test_images():
    """Load sample images from the test dataset."""
    images_dir = "dataset/images"
    labels_csv = "dataset/labels.csv"
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_csv):
        return [], {}
    
    # Load labels
    try:
        df = pd.read_csv(labels_csv)
        labels = dict(zip(df['filename'], df['label']))
    except:
        labels = {}
    
    # Get sample images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
    sample_images = random.sample(image_files, min(20, len(image_files)))
    
    return sample_images, labels


def format_time(seconds):
    """Format time in a readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI CAPTCHA Solver</h1>
        <p>Advanced CNN model that solves CAPTCHAs humans struggle with</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ **Model not found!** Please train the model first by running `python run_model_training.py`")
        st.stop()
    
    # Load test images
    test_images, labels = load_test_images()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="demo-card">', unsafe_allow_html=True)
        st.markdown("### 📸 Choose CAPTCHA Image")
        
        # Image selection tabs
        tab1, tab2 = st.tabs(["📁 Upload Image", "🎲 Test Dataset"])
        
        selected_image = None
        image_source = None
        true_label = None
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload a CAPTCHA image",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                help="Upload your own CAPTCHA image to test the AI model"
            )
            
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                image_source = "uploaded"
                st.success("✅ Image uploaded successfully!")
        
        with tab2:
            if test_images:
                selected_file = st.selectbox(
                    "Choose from test dataset",
                    [""] + test_images,
                    help="Select a CAPTCHA from our test dataset"
                )
                
                if selected_file:
                    image_path = os.path.join("dataset/images", selected_file)
                    selected_image = Image.open(image_path)
                    image_source = "dataset"
                    true_label = labels.get(selected_file, "Unknown")
                    st.success(f"✅ Selected: {selected_file}")
            else:
                st.warning("⚠️ No test images found. Generate dataset first with `python run_dataset_generation.py`")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected image
        if selected_image:
            st.markdown('<div class="demo-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ CAPTCHA Image")
            
            # Center the image
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(selected_image, caption="CAPTCHA to solve", use_column_width=True)
            
            if true_label:
                st.info(f"🎯 **Ground Truth:** {true_label}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="demo-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 AI Solver")
        
        if selected_image:
            if st.button("🤖 Solve CAPTCHA", use_container_width=True):
                with st.spinner("🧠 AI is thinking..."):
                    # Save uploaded image temporarily if needed
                    if image_source == "uploaded":
                        temp_path = "temp_captcha.png"
                        selected_image.save(temp_path)
                        image_path = temp_path
                    else:
                        image_path = os.path.join("dataset/images", selected_file)
                    
                    # Predict with timing
                    start_time = time.time()
                    try:
                        predicted_text, char_confidences, overall_confidence = model.predict_single_image(image_path)
                        prediction_time = time.time() - start_time
                        
                        # Clean up temp file
                        if image_source == "uploaded" and os.path.exists("temp_captcha.png"):
                            os.remove("temp_captcha.png")
                        
                        # Store results in session state
                        st.session_state.prediction = predicted_text
                        st.session_state.confidence = overall_confidence
                        st.session_state.char_confidences = char_confidences
                        st.session_state.prediction_time = prediction_time
                        st.session_state.true_label = true_label
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
        else:
            st.info("👆 Select an image first")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Results section
    if hasattr(st.session_state, 'prediction'):
        st.markdown('<div class="demo-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 AI Results")
        
        # Main prediction result
        st.markdown(f"""
        <div class="result-success">
            <h2>{st.session_state.prediction}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-item">
                <p class="metric-value">{st.session_state.confidence:.1%}</p>
                <p class="metric-label">Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-item">
                <p class="metric-value">{format_time(st.session_state.prediction_time)}</p>
                <p class="metric-label">AI Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.true_label:
                is_correct = st.session_state.prediction == st.session_state.true_label
                accuracy_text = "✅ Correct" if is_correct else "❌ Wrong"
                st.markdown(f"""
                <div class="metric-item">
                    <p class="metric-value" style="font-size: 1.5rem;">{accuracy_text}</p>
                    <p class="metric-label">Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-item">
                    <p class="metric-value">N/A</p>
                    <p class="metric-label">Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence bar
        st.markdown("#### Character Confidence")
        confidence_html = ""
        for i, conf in enumerate(st.session_state.char_confidences):
            char = st.session_state.prediction[i] if i < len(st.session_state.prediction) else "?"
            confidence_html += f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                    <span><strong>{char}</strong></span>
                    <span>{conf:.1%}</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf*100}%;"></div>
                </div>
            </div>
            """
        st.markdown(confidence_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Human vs AI Challenge
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("### 👨‍💻 Human vs AI Challenge")
    
    if selected_image:
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown("**🤖 AI Performance**")
            if hasattr(st.session_state, 'prediction'):
                st.success(f"**Result:** {st.session_state.prediction}")
                st.info(f"**Time:** {format_time(st.session_state.prediction_time)}")
                st.info(f"**Confidence:** {st.session_state.confidence:.1%}")
            else:
                st.info("Run AI solver first")
        
        with col2:
            st.markdown('<div class="vs-circle">VS</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("**👨‍💻 Your Turn**")
            
            # Initialize stopwatch state
            if 'stopwatch_running' not in st.session_state:
                st.session_state.stopwatch_running = False
                st.session_state.start_time = None
                st.session_state.human_time = None
            
            # Stopwatch display
            if st.session_state.stopwatch_running and st.session_state.start_time:
                current_time = time.time() - st.session_state.start_time
                st.markdown(f"""
                <div class="stopwatch">
                    <div class="stopwatch-time">{format_time(current_time)}</div>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.human_time:
                st.markdown(f"""
                <div class="stopwatch">
                    <div class="stopwatch-time">{format_time(st.session_state.human_time)}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stopwatch">
                    <div class="stopwatch-time">0.00s</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Human input
            human_answer = st.text_input(
                "Type what you see:",
                max_chars=5,
                placeholder="Enter 5 characters",
                key="human_input"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("⏱️ Start Timer", use_container_width=True):
                    st.session_state.stopwatch_running = True
                    st.session_state.start_time = time.time()
                    st.session_state.human_time = None
                    st.rerun()
            
            with col_btn2:
                if st.button("⏹️ Stop Timer", use_container_width=True):
                    if st.session_state.stopwatch_running and st.session_state.start_time:
                        st.session_state.human_time = time.time() - st.session_state.start_time
                        st.session_state.stopwatch_running = False
                        st.rerun()
            
            # Show human results
            if human_answer and len(human_answer) == 5:
                if st.session_state.true_label:
                    is_correct = human_answer.upper() == st.session_state.true_label
                    if is_correct:
                        st.success("✅ Correct!")
                    else:
                        st.error("❌ Incorrect")
                
                if st.session_state.human_time:
                    st.info(f"**Your Time:** {format_time(st.session_state.human_time)}")
    else:
        st.info("👆 Select a CAPTCHA image to start the challenge")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance comparison
    if (hasattr(st.session_state, 'prediction') and 
        st.session_state.human_time and 
        human_answer and len(human_answer) == 5):
        
        st.markdown('<div class="demo-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Performance Comparison")
        
        # Speed comparison
        speed_ratio = st.session_state.human_time / st.session_state.prediction_time
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚡ Speed Analysis")
            st.markdown(f"**AI Time:** {format_time(st.session_state.prediction_time)}")
            st.markdown(f"**Human Time:** {format_time(st.session_state.human_time)}")
            st.markdown(f"**AI is {speed_ratio:.1f}x faster!**")
        
        with col2:
            if st.session_state.true_label:
                st.markdown("#### 🎯 Accuracy Analysis")
                ai_correct = st.session_state.prediction == st.session_state.true_label
                human_correct = human_answer.upper() == st.session_state.true_label
                
                st.markdown(f"**AI Result:** {'✅ Correct' if ai_correct else '❌ Wrong'}")
                st.markdown(f"**Human Result:** {'✅ Correct' if human_correct else '❌ Wrong'}")
                
                if ai_correct and human_correct:
                    st.success("🎉 Both got it right!")
                elif ai_correct and not human_correct:
                    st.info("🤖 AI wins this round!")
                elif not ai_correct and human_correct:
                    st.info("👨‍💻 Human wins this round!")
                else:
                    st.warning("😅 Both got it wrong!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p>🚀 Built with TensorFlow, OpenCV, and Streamlit | 
        <strong>AI CAPTCHA Solver</strong> - Engineering Day Demo</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
