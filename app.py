"""
Main entry point for the Finance Analyzer Streamlit application.
"""

import streamlit as st
import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from src.dashboard import main

if __name__ == "__main__":
    main()
