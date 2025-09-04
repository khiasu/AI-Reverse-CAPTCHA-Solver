"""
Streamlit Cloud entry point for AI CAPTCHA Solver Demo.
This file imports and runs the main demo app.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main demo
from app.captcha_demo import main

if __name__ == "__main__":
    main()
