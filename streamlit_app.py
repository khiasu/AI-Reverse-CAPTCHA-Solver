"""
Streamlit Cloud entry point for AI CAPTCHA Solver Demo.
This file imports and runs the main demo app.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main demo
try:
    from app.captcha_demo import main
except ImportError:
    # Direct import for Streamlit Cloud
    import streamlit as st
    st.error("Import error. Please use app/captcha_demo.py as the main file instead.")
    st.stop()

if __name__ == "__main__":
    main()
