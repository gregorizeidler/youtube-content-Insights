#!/usr/bin/env python3
"""
Quick launcher for YouTube Content Insights web application.
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_streamlit():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not available."""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit==1.28.1"])
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Streamlit")
        return False

def main():
    """Main launcher function."""
    print("🎥 YouTube Content Insights")
    print("Web Application Launcher")
    print("=" * 40)
    
    # Check if Streamlit is available
    if not check_streamlit():
        print("⚠️  Streamlit not found. Installing...")
        if not install_streamlit():
            print("❌ Cannot proceed without Streamlit")
            return
    
    # Launch Streamlit app
    print("🚀 Launching web application...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⚠️  To stop the app, press Ctrl+C in this terminal")
    print("=" * 40)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()
