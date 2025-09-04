"""
Simple script to launch the CAPTCHA Solver Demo App.
"""

import os
import sys
import subprocess

def main():
    """
    Launch the Streamlit demo app.
    """
    print("=== AI CAPTCHA Solver Demo ===")
    print("Launching modern web interface...")
    print()
    
    # Check if model exists
    model_paths = ["model/best_model.h5", "model/model.h5"]
    model_found = any(os.path.exists(path) for path in model_paths)
    
    if not model_found:
        print("⚠️  WARNING: No trained model found!")
        print("   Please train the model first:")
        print("   python run_model_training.py")
        print()
        print("   The demo will still launch but predictions won't work.")
        print()
    
    # Check if dataset exists
    if not os.path.exists("dataset/images"):
        print("⚠️  WARNING: No dataset found!")
        print("   Please generate dataset first:")
        print("   python run_dataset_generation.py")
        print()
        print("   The demo will still launch but test images won't be available.")
        print()
    
    # Launch Streamlit app
    app_path = "app/captcha_demo.py"
    
    if not os.path.exists(app_path):
        print(f"❌ ERROR: Demo app not found at {app_path}")
        return
    
    print("🚀 Starting Streamlit demo app...")
    print("   The app will open in your default web browser")
    print("   URL: http://localhost:8501")
    print()
    print("Features:")
    print("  📸 Upload your own CAPTCHA images")
    print("  🎲 Test with generated dataset samples")
    print("  🤖 AI prediction with confidence scores")
    print("  ⏱️  Human vs AI speed comparison")
    print("  📊 Real-time performance metrics")
    print()
    print("Press Ctrl+C to stop the demo")
    print("="*50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for trying the AI CAPTCHA Solver!")
    except Exception as e:
        print(f"\n❌ Error launching demo: {e}")
        print("\nTroubleshooting:")
        print("1. Install Streamlit: pip install streamlit")
        print("2. Check if all dependencies are installed")
        print("3. Ensure you're in the project root directory")


if __name__ == "__main__":
    main()
