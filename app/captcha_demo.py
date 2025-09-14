"""
Modern AI CAPTCHA Solver - Minimalist UI
Top-tier industry standard design with responsive layout
"""

import streamlit as st
import os
import sys
import time
import random
import pandas as pd
from PIL import Image
import numpy as np
import io
import base64
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing OpenCV first to get better error messages
try:
    import cv2
except ImportError:
    import subprocess
    import sys
    import streamlit as st
    
    st.error("""
    ## 🚨 OpenCV Not Found
    
    The application requires OpenCV to run. Attempting to install it now...
    """)
    
    try:
        # Try installing opencv-python-headless
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        import cv2
        st.rerun()  # Restart the app after successful installation
    except Exception as e:
        st.error(f"""
        ## ❌ Installation Failed
        
        Could not install OpenCV automatically. Please run:
        
        ```bash
        pip install opencv-python-headless
        ```
        
        Error details: {str(e)}
        """)
        st.stop()

try:
    from model.captcha_inference import CaptchaInference
except ImportError as e:
    import streamlit as st
    st.error(f"""
    ## 🚨 Import Error
    
    Could not import CAPTCHA inference module: {str(e)}
    
    Please ensure:
    1. All requirements are installed: `pip install -r requirements.txt`
    2. The model files are in the correct location
    """)
    st.stop()

# ================================
# MODERN DESIGN SYSTEM
# ================================

def get_modern_css(theme='dark'):
    """Generate modern CSS with theme support"""
    
    # Color palettes
    if theme == 'dark':
        colors = {
            'bg-primary': '#0A0A0B',
            'bg-secondary': '#151516', 
            'bg-tertiary': '#1E1E20',
            'bg-card': '#242427',
            'bg-hover': '#2A2A2E',
            'text-primary': '#FFFFFF',
            'text-secondary': '#B8B8B8',
            'text-muted': '#6B6B73',
            'accent-primary': '#8B5FBF',
            'accent-secondary': '#6B46C1',
            'accent-light': '#A78BFA',
            'border': '#2D2D31',
            'border-hover': '#404046',
            'shadow': 'rgba(0, 0, 0, 0.3)',
            'shadow-strong': 'rgba(0, 0, 0, 0.5)'
        }
    else:  # light theme
        colors = {
            'bg-primary': '#FAFAFA',
            'bg-secondary': '#FFFFFF',
            'bg-tertiary': '#F5F5F7',
            'bg-card': '#FFFFFF',
            'bg-hover': '#F0F0F2',
            'text-primary': '#1A1A1B',
            'text-secondary': '#525259',
            'text-muted': '#8E8E93',
            'accent-primary': '#8B5FBF',
            'accent-secondary': '#6B46C1',
            'accent-light': '#A78BFA',
            'border': '#E5E5E7',
            'border-hover': '#D1D1D6',
            'shadow': 'rgba(0, 0, 0, 0.1)',
            'shadow-strong': 'rgba(0, 0, 0, 0.15)'
        }
    
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Custom Properties */
    :root {{
        --bg-primary: {colors['bg-primary']};
        --bg-secondary: {colors['bg-secondary']};
        --bg-tertiary: {colors['bg-tertiary']};
        --bg-card: {colors['bg-card']};
        --bg-hover: {colors['bg-hover']};
        --text-primary: {colors['text-primary']};
        --text-secondary: {colors['text-secondary']};
        --text-muted: {colors['text-muted']};
        --accent-primary: {colors['accent-primary']};
        --accent-secondary: {colors['accent-secondary']};
        --accent-light: {colors['accent-light']};
        --border: {colors['border']};
        --border-hover: {colors['border-hover']};
        --shadow: {colors['shadow']};
        --shadow-strong: {colors['shadow-strong']};
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --spacing-xs: 0.5rem;
        --spacing-sm: 1rem;
        --spacing-md: 1.5rem;
        --spacing-lg: 2rem;
        --spacing-xl: 3rem;
    }}
    
    /* Global Reset & Base Styles */
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    html, body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
        background: var(--bg-primary);
        color: var(--text-primary);
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    
    .stApp {{
        background: var(--bg-primary);
        min-height: 100vh;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--border-hover);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-muted);
    }}
    
    /* Layout Components */
    .main-container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: var(--spacing-md);
    }}
    
    .header {{
        text-align: center;
        margin-bottom: var(--spacing-xl);
        padding: var(--spacing-lg) 0;
    }}
    
    .header h1 {{
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: var(--spacing-sm);
    }}
    
    .header p {{
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }}
    
    /* Cards */
    .card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        margin-bottom: var(--spacing-md);
        box-shadow: 0 2px 12px var(--shadow);
        transition: all 0.3s ease;
    }}
    
    .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 32px var(--shadow-strong);
        border-color: var(--border-hover);
    }}
    
    .card-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--spacing-sm);
        display: flex;
        align-items: center;
        gap: var(--spacing-xs);
    }}
    
    .card-content {{
        color: var(--text-secondary);
        line-height: 1.7;
    }}
    
    /* Results Display */
    .prediction-result {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        text-align: center;
        margin: var(--spacing-md) 0;
    }}
    
    .prediction-text {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-primary);
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.2em;
        margin: var(--spacing-md) 0;
    }}
    
    /* Stats */
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: var(--spacing-md);
        margin-top: var(--spacing-md);
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .main-container {{
            padding: var(--spacing-sm);
        }}
        
        .header h1 {{
            font-size: 2rem;
        }}
        
        .card {{
            padding: var(--spacing-md);
        }}
        
        .stats-grid {{
            grid-template-columns: repeat(2, 1fr);
            gap: var(--spacing-sm);
        }}
    }}
    
    @media (max-width: 480px) {{
        .stats-grid {{
            grid-template-columns: 1fr;
        }}
        
        .header p {{
            font-size: 1rem;
        }}
        
        .prediction-text {{
            font-size: 1.5rem;
        }}
    }}
    
    /* Loading Animation */
    .loading {{
        display: inline-flex;
        align-items: center;
        gap: var(--spacing-xs);
    }}
    
    .spinner {{
        width: 20px;
        height: 20px;
        border: 2px solid var(--border);
        border-top: 2px solid var(--accent-primary);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Fade animations */
    .fade-in {{
        animation: fadeIn 0.6s ease-out;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Enhanced Streamlit component styling */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-weight: 500 !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 12px var(--shadow) !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(139, 95, 191, 0.5) !important;
        background: linear-gradient(135deg, var(--accent-light), var(--accent-primary)) !important;
    }}
    
    .stButton > button[data-testid="button-secondary"] {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-hover) !important;
    }}
    
    .stButton > button[data-testid="button-secondary"]:hover {{
        background: var(--bg-hover) !important;
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--bg-card);
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
        padding: 4px;
        margin-bottom: var(--spacing-lg);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: var(--radius-sm);
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: var(--accent-primary);
        color: white;
    }}
    
    .stFileUploader {{
        background: var(--bg-tertiary);
        border: 2px dashed var(--border);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .stFileUploader::before {{
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, var(--accent-primary), var(--accent-light));
        border-radius: var(--radius-lg);
        opacity: 0;
        transition: opacity 0.4s ease;
        z-index: -1;
    }}
    
    .stFileUploader:hover {{
        border-color: transparent;
        background: var(--bg-hover);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px var(--shadow-strong);
    }}
    
    .stFileUploader:hover::before {{
        opacity: 1;
    }}
    
    .stMetric {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: var(--spacing-md);
        margin: var(--spacing-xs) 0;
        transition: all 0.3s ease;
        position: relative;
    }}
    
    .stMetric:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 16px var(--shadow);
        border-color: var(--border-hover);
    }}
    
    .stExpander {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        margin: var(--spacing-xs) 0;
        transition: all 0.3s ease;
    }}
    
    .stExpander:hover {{
        border-color: var(--border-hover);
        box-shadow: 0 2px 8px var(--shadow);
    }}
    
    /* Image styling */
    .stImage > img {{
        border-radius: var(--radius-md);
        box-shadow: 0 4px 16px var(--shadow);
        transition: all 0.3s ease;
    }}
    
    .stImage > img:hover {{
        transform: scale(1.02);
        box-shadow: 0 8px 24px var(--shadow-strong);
    }}
</style>
"""

# ================================
# MODERN APPLICATION LOGIC
# ================================

def init_session_state():
    """Initialize session state variables"""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'timer_start' not in st.session_state:
        st.session_state.timer_start = None
    if 'human_answer' not in st.session_state:
        st.session_state.human_answer = ""
    if 'challenge_active' not in st.session_state:
        st.session_state.challenge_active = False
    if 'ai_result' not in st.session_state:
        st.session_state.ai_result = None

def toggle_theme():
    """Toggle between dark and light themes"""
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
    st.rerun()

def get_random_sample_image():
    """Get a random sample image from the dataset"""
    dataset_dir = "dataset/images"
    if os.path.exists(dataset_dir):
        images = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            return os.path.join(dataset_dir, random.choice(images))
    return None

def main():
    """Main application function"""
    st.set_page_config(
        page_title="AI CAPTCHA Solver",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None
    )
    
    # Initialize session state
    init_session_state()
    
    # Apply CSS based on current theme
    st.markdown(get_modern_css(st.session_state.theme), unsafe_allow_html=True)
    
    # Theme toggle button (fixed position)
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_text = "Dark" if st.session_state.theme == 'light' else "Light"
    
    # Create columns for the theme toggle
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", help="Toggle theme"):
            toggle_theme()
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header fade-in">
        <h1>AI CAPTCHA Solver</h1>
        <p>Advanced neural network powered CAPTCHA recognition with real-time analysis and confidence scoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize inference system
    try:
        model_path = "model/best_model.h5"
        inference_system = CaptchaInference(model_path)
        model_available = True
    except Exception as e:
        st.error(f"⚠️ Model initialization failed: {str(e)}")
        st.info("🎭 Running in demo mode with mock predictions")
        try:
            inference_system = CaptchaInference("dummy_path")
            model_available = False
        except:
            st.error("Failed to initialize inference system")
            return
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Statistics", "ℹ️ About"])
    
    with tab1:
        # Clean upload section with smooth interactions
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            uploaded_file = st.file_uploader(
                "📤 Upload CAPTCHA Image",
                type=['png', 'jpg', 'jpeg'],
                help="Drag and drop or click to upload"
            )
            
        with col2:
            if st.button("🎲 Try Random Sample", use_container_width=True, type="secondary"):
                sample_path = get_random_sample_image()
                if sample_path:
                    st.session_state.current_image = sample_path
                    st.session_state.uploaded_file = sample_path
        
        # Process image if available
        image_to_process = None
        image_source = None
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_upload.png", "wb") as f:
                f.write(uploaded_file.getvalue())
            image_to_process = "temp_upload.png"
            image_source = "uploaded"
        elif hasattr(st.session_state, 'uploaded_file') and st.session_state.uploaded_file:
            image_to_process = st.session_state.uploaded_file
            image_source = "sample"
        
        if image_to_process:
            # Load image for display
            try:
                display_image = Image.open(image_to_process)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return
            
            st.markdown("---")
            
            # Main image display - centered
            col_spacer1, col_img, col_spacer2 = st.columns([1, 2, 1])
            with col_img:
                st.image(display_image, width=350, caption="🖼️ CAPTCHA Challenge")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Action buttons row
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🤖 AI Solve", type="primary", use_container_width=True):
                    st.session_state.ai_analyzing = True
                    
            with col2:
                if not st.session_state.challenge_active:
                    if st.button("⏱️ Human Challenge", type="secondary", use_container_width=True):
                        st.session_state.challenge_active = True
                        st.session_state.timer_start = time.time()
                        st.session_state.human_answer = ""
                        st.rerun()
                else:
                    if st.button("✅ Submit Answer", type="secondary", use_container_width=True):
                        if st.session_state.human_answer:
                            human_time = time.time() - st.session_state.timer_start
                            st.session_state.human_time = human_time
                            st.session_state.challenge_active = False
                            st.rerun()
            
            with col3:
                if st.button("🔄 New Image", use_container_width=True):
                    sample_path = get_random_sample_image()
                    if sample_path:
                        st.session_state.current_image = sample_path
                        st.session_state.uploaded_file = sample_path
                        st.session_state.challenge_active = False
                        st.session_state.ai_result = None
                        st.rerun()
            
            # Human challenge input
            if st.session_state.challenge_active:
                st.markdown("<br>", unsafe_allow_html=True)
                col_timer, col_input = st.columns([1, 2])
                
                with col_timer:
                    elapsed = time.time() - st.session_state.timer_start
                    st.metric("⏰ Your Time", f"{elapsed:.1f}s")
                    
                with col_input:
                    st.session_state.human_answer = st.text_input(
                        "Enter CAPTCHA text:",
                        value=st.session_state.human_answer,
                        max_chars=5,
                        placeholder="Type what you see..."
                    )
            
            # Process AI analysis
            if hasattr(st.session_state, 'ai_analyzing') and st.session_state.ai_analyzing:
                with st.spinner("🔄 AI Processing..."):
                    try:
                        start_time = time.time()
                        predicted_text, char_confidences, overall_confidence = inference_system.predict_single_image(image_to_process)
                        end_time = time.time()
                        
                        processing_time = (end_time - start_time) * 1000  # Convert to ms
                        
                        # Store AI result
                        st.session_state.ai_result = {
                            'text': predicted_text,
                            'confidence': overall_confidence,
                            'time_ms': processing_time,
                            'char_confidences': char_confidences
                        }
                        
                        # Add to prediction history
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'predicted_text': predicted_text,
                            'confidence': overall_confidence,
                            'processing_time': processing_time,
                            'source': image_source
                        })
                        
                        st.session_state.ai_analyzing = False
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ AI Analysis failed: {str(e)}")
                        st.session_state.ai_analyzing = False
            
            # Display results section
            if st.session_state.ai_result or hasattr(st.session_state, 'human_time'):
                st.markdown("---")
                st.markdown("### 🏆 Results")
                
                col_ai, col_vs, col_human = st.columns([2, 1, 2])
                
                # AI Results
                with col_ai:
                    if st.session_state.ai_result:
                        ai_data = st.session_state.ai_result
                        st.markdown("""
                        <div style='text-align: center; padding: 1rem; background: var(--bg-card); border-radius: var(--radius-lg); border: 1px solid var(--border);'>
                            <h4>🤖 AI Solution</h4>
                            <div style='font-size: 2rem; font-weight: bold; color: var(--accent-primary); font-family: monospace; margin: 1rem 0;'>{}</div>
                            <div style='color: var(--text-secondary);'>Time: {:.0f}ms | Confidence: {:.1f}%</div>
                        </div>
                        """.format(ai_data['text'], ai_data['time_ms'], ai_data['confidence']*100), unsafe_allow_html=True)
                
                # VS section
                with col_vs:
                    st.markdown("""
                    <div style='text-align: center; padding: 2rem 0;'>
                        <h2 style='color: var(--accent-primary);'>VS</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Human Results
                with col_human:
                    if hasattr(st.session_state, 'human_time'):
                        # Get ground truth for accuracy check
                        actual_answer = None
                        if image_to_process and 'captcha_' in os.path.basename(image_to_process):
                            try:
                                labels_path = os.path.join('dataset', 'labels.csv')
                                if os.path.exists(labels_path):
                                    import pandas as pd
                                    labels_df = pd.read_csv(labels_path)
                                    filename = os.path.basename(image_to_process)
                                    match = labels_df[labels_df['filename'] == filename]
                                    if not match.empty:
                                        actual_answer = match.iloc[0]['label']
                            except:
                                pass
                        
                        human_correct = st.session_state.human_answer.upper() == (actual_answer if actual_answer else "UNKNOWN")
                        accuracy_text = "✅ Correct" if human_correct else "❌ Incorrect"
                        
                        st.markdown("""
                        <div style='text-align: center; padding: 1rem; background: var(--bg-card); border-radius: var(--radius-lg); border: 1px solid var(--border);'>
                            <h4>👤 Your Solution</h4>
                            <div style='font-size: 2rem; font-weight: bold; color: var(--text-primary); font-family: monospace; margin: 1rem 0;'>{}</div>
                            <div style='color: var(--text-secondary);'>Time: {:.1f}s | {}</div>
                        </div>
                        """.format(st.session_state.human_answer.upper(), st.session_state.human_time, accuracy_text), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='text-align: center; padding: 1rem; background: var(--bg-tertiary); border-radius: var(--radius-lg); border: 2px dashed var(--border);'>
                            <h4>👤 Human Challenge</h4>
                            <p style='color: var(--text-muted); margin: 1rem 0;'>Start the challenge to compete!</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Winner announcement
                if st.session_state.ai_result and hasattr(st.session_state, 'human_time'):
                    ai_time_s = st.session_state.ai_result['time_ms'] / 1000
                    human_time_s = st.session_state.human_time
                    
                    if ai_time_s < human_time_s:
                        winner = "🤖 AI Wins!"
                        speed_diff = f"AI was {human_time_s/ai_time_s:.1f}x faster"
                    else:
                        winner = "👤 Human Wins!"
                        speed_diff = f"Human was {ai_time_s/human_time_s:.1f}x faster"
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, var(--accent-primary), var(--accent-light)); border-radius: var(--radius-lg); margin: 1rem 0; color: white;'>
                        <h3>{winner}</h3>
                        <p>{speed_diff}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Character analysis in expandable
                if st.session_state.ai_result and st.session_state.ai_result.get('char_confidences'):
                    with st.expander("🔍 Detailed Character Analysis", expanded=False):
                        char_confidences = st.session_state.ai_result['char_confidences']
                        predicted_text = st.session_state.ai_result['text']
                        
                        if len(char_confidences) == len(predicted_text):
                            char_cols = st.columns(len(predicted_text))
                            for i, (char, conf) in enumerate(zip(predicted_text, char_confidences)):
                                with char_cols[i]:
                                    st.metric(
                                        label=f"'{char}'",
                                        value=f"{conf*100:.1f}%"
                                    )
    
    with tab2:
        st.markdown("### 📈 Performance Statistics")
        
        if st.session_state.prediction_history:
            history = st.session_state.prediction_history
            
            # Clean summary stats
            total_predictions = len(history)
            avg_confidence = np.mean([h['confidence'] for h in history]) * 100
            avg_processing_time = np.mean([h['processing_time'] for h in history])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", total_predictions)
            with col2: 
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
            with col3:
                st.metric("Average Speed", f"{avg_processing_time:.0f}ms")
            
            # Recent predictions
            st.markdown("### Recent Predictions")
            
            for i, pred in enumerate(reversed(history[-10:])):
                with st.expander(f"Prediction {len(history)-i}: {pred['predicted_text']} ({pred['confidence']*100:.1f}%)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Time:** {pred['timestamp'].strftime('%H:%M:%S')}")
                    with col2:
                        st.write(f"**Source:** {pred['source'].title()}")
                    with col3:
                        st.write(f"**Processing:** {pred['processing_time']:.0f}ms")
        else:
            st.info("📊 No predictions yet. Analyze some CAPTCHAs to see statistics here!")
    
    with tab3:
        st.markdown("### 🏦 Engineering Day Model Exhibition")
        
        # Team information
        st.markdown("""
        **Team:** B.Tech CSE 5th Semester  
        **Backend:** Khiasuthong.T  
        **Frontend:** Yoihenba  
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clean project description
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            **What it does:**  
            Recognizes text in CAPTCHA images using deep learning. 
            The CNN model processes distorted images and predicts 
            5-character alphanumeric codes with confidence scores.
            
            **Technical Stack:**
            - TensorFlow/Keras for neural networks
            - OpenCV for image processing  
            - Streamlit for web interface
            - Custom dataset of 5,000 synthetic CAPTCHAs
            """)
        
        with col2:
            st.markdown("""
            **Performance:**
            - Speed: ~100ms per image
            - Accuracy: 70-90% on test data
            - Model: 2.1M parameters
            - Runs locally without internet
            
            **Features:**
            - Real-time prediction
            - Character-wise confidence
            - Dark/Light themes
            - Mobile responsive design
            """)
        
        # Clean dataset info
        with st.expander("📈 Dataset Details", expanded=False):
            dataset_dir = "dataset/images"
            labels_file = "dataset/labels.csv"
            
            if os.path.exists(dataset_dir) and os.path.exists(labels_file):
                num_images = len([f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                try:
                    df = pd.read_csv(labels_file)
                    num_labels = len(df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Images Generated", f"{num_images:,}")
                    with col2:
                        st.metric("Labels Created", f"{num_labels:,}")
                        
                    if len(df) > 0:
                        st.write("**Sample from dataset:**")
                        sample_df = df.sample(min(3, len(df)))
                        st.dataframe(sample_df, width=600)
                        
                except Exception as e:
                    st.warning(f"Could not read labels file: {e}")
            else:
                st.info("Dataset will be generated when you run the training pipeline.")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: var(--text-muted); border-top: 1px solid var(--border); margin-top: 3rem;">
        <p>🤖 AI CAPTCHA Solver | Modern Neural Network Architecture</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Built with TensorFlow, Streamlit & Modern Web Technologies</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

# ================================
# RUN APPLICATION
# ================================
if __name__ == "__main__":
    main()
