#!/usr/bin/env python3
"""
NeuroDx-MultiModal Dashboard Launcher
Simple script to launch the Streamlit dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    
    print("🧠 Starting NeuroDx-MultiModal Dashboard...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully")
    
    # Check if plotly is installed
    try:
        import plotly
        print(f"✅ Plotly version: {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        print("✅ Plotly installed successfully")
    
    # Launch dashboard
    dashboard_path = Path(__file__).parent / "streamlit_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return
    
    print(f"🚀 Launching dashboard from: {dashboard_path}")
    print("📱 Dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()