# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import json
import re
import PyPDF2
import docx
from io import BytesIO
import base64
import secrets
import time

# ---------------- CONFIG ---------------- #
DB_PATH = "heart_app.db"
MODEL_PATH = Path("models") / "rf.joblib"

# Try loading model; graceful fallback if missing
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

# ---------------- MODERN ADAPTIVE STYLING ---------------- #
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Custom Properties for Theme Adaptation */
    :root {
        --primary-color: #667eea;
        --primary-dark: #5a67d8;
        --secondary-color: #764ba2;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --text-primary: #1a202c;
        --text-secondary: #718096;
        --bg-primary: #ffffff;
        --bg-secondary: #f7fafc;
        --bg-tertiary: #edf2f7;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Dark theme detection and adaptation */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #f7fafc;
            --text-secondary: #a0aec0;
            --bg-primary: #1a202c;
            --bg-secondary: #2d3748;
            --bg-tertiary: #4a5568;
            --border-color: #4a5568;
        }
    }
    
    /* Override Streamlit's dark theme detection */
    .stApp[data-theme="dark"] {
        --text-primary: #f7fafc;
        --text-secondary: #a0aec0;
        --bg-primary: #1a202c;
        --bg-secondary: #2d3748;
        --bg-tertiary: #4a5568;
        --border-color: #4a5568;
    }
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main container */
    .stApp {
        background-color: var(--bg-secondary);
    }
    
    .main {
        padding-top: 0rem;
        background-color: var(--bg-secondary);
    }

    /* Navigation Pills Container */
    # .nav-pills-container {
    #     background: rgba(255, 255, 255, 0.02);
    #     backdrop-filter: blur(10px);
    #     border: 1px solid rgba(255, 255, 255, 0.08);
    #     border-radius: 20px;
    #     padding: 1rem;
    #     margin: 1rem 0 2rem 0;
    #     box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    # }
    
    /* Modern Navigation Header */
    .top-nav {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 0 -1rem;
        box-shadow: var(--shadow-lg);
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .nav-content {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .nav-logo {
        font-size: 2rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .nav-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .nav-subtitle {
        font-size: 0.875rem;
        color: rgba(255,255,255,0.8);
        font-weight: 400;
    }
    
    .nav-user {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: white;
    }
    
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 1.1rem;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .user-info {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    
    .user-name {
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .user-role {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
    }
    
    /* Navigation Pills */
    .nav-pills {
        display: flex;
        gap: 0.5rem;
        margin: 2rem 0;
        padding: 0.5rem;
        background: var(--bg-primary);
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        max-width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }
    
    .nav-pill {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        background: transparent;
        color: var(--text-secondary);
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-decoration: none;
    }
    
    .nav-pill:hover {
        background: var(--bg-tertiary);
        transform: translateY(-1px);
    }
    
    .nav-pill.active {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        box-shadow: var(--shadow-sm);
    }
    
    /* Page container */
    .page-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Modern Card Design */
    # .glass-card {
    #     background: var(--bg-primary);
    #     backdrop-filter: blur(20px);
    #     border-radius: 16px;
    #     padding: 2rem;
    #     box-shadow: var(--shadow-lg);
    #     border: 1px solid var(--border-color);
    #     margin-bottom: 2rem;
    #     transition: all 0.3s ease;
    # }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .compact-card {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .compact-card:hover {
        box-shadow: var(--shadow-md);
    }
    
    /* Professional Auth Pages - Split Screen Layout */
    # .auth-wrapper {
    #     min-height: 100vh;
    #     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    #     padding: 0;
    #     margin: 0;
    # }
    
    /* Override Streamlit's column gaps for auth page */
    # .auth-wrapper .stColumns {
    #     gap: 0 !important;
    # }
    
    # .auth-wrapper .stColumn {
    #     padding: 0 !important;
    # }
    
    .auth-welcome-board {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4rem 3rem;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        position: relative;
        overflow: hidden;
    }
    
    .auth-welcome-board::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        pointer-events: none;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-20px) rotate(120deg); }
        66% { transform: translateY(10px) rotate(240deg); }
    }
    
    .welcome-content {
        position: relative;
        z-index: 2;
        max-width: 500px;
    }
    
    .welcome-logo {
        font-size: 4rem;
        margin-bottom: 2rem;
        display: block;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
    }
    
    .welcome-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        line-height: 1.1;
        letter-spacing: -0.025em;
    }
    
    .welcome-subtitle {
        font-size: 1.25rem;
        line-height: 1.6;
        margin-bottom: 3rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .welcome-features {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .welcome-feature {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .welcome-feature-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem;
        border-radius: 50%;
        min-width: 2.5rem;
        height: 2.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    # .auth-forms-container {
    #     background: var(--bg-primary);
    #     padding: 4rem 3rem;
    #     min-height: 100vh;
    #     display: flex;
    #     align-items: center;
    #     justify-content: center;
    #     position: relative;
    # }
    
    .auth-container {
        max-width: 450px;
        width: 100%;
        background: var(--bg-primary);
        border-radius: 24px;
        padding: 0;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }
    
    .auth-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 24px 24px 0 0;
    }
    
    # .auth-card-body {
    #     padding: 3rem 2.5rem 2.5rem;
    #     position: relative;
    #     z-index: 1;
    # }
    
    /* Mobile responsive design */
    @media (max-width: 768px) {
        .auth-wrapper .stColumns {
            flex-direction: column !important;
        }
        
        .auth-welcome-board {
            padding: 3rem 2rem;
            text-align: center;
            min-height: 50vh;
        }
        
        .welcome-title {
            font-size: 2.5rem;
        }
        
        .welcome-subtitle {
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        .welcome-features {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 1rem;
        }
        
        .welcome-feature {
            flex-direction: column;
            margin-bottom: 1rem;
            text-align: center;
            min-width: 140px;
        }
        
        .welcome-feature-icon {
            margin-right: 0;
            margin-bottom: 0.5rem;
        }
        
        .auth-forms-container {
            padding: 2rem 1rem;
            min-height: auto;
        }
        
        .auth-card-body {
            padding: 2rem 1.5rem;
        }
    }
    
    .auth-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    # .auth-logo {
    #     font-size: 4rem;
    #     margin-bottom: 1.5rem;
    #     display: block;
    #     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    #     -webkit-background-clip: text;
    #     -webkit-text-fill-color: transparent;
    #     background-clip: text;
    #     filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    # }
    
    .auth-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        letter-spacing: -0.025em;
    }
    
    .auth-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.5;
        font-weight: 400;
    }
    
    .auth-divider {
        display: flex;
        align-items: center;
        margin: 2rem 0;
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    .auth-divider::before,
    .auth-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border-color);
    }
    
    .auth-divider span {
        padding: 0 1rem;
        background: var(--bg-primary);
    }
    
    /* Professional Form Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 2px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1), 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
        outline: none !important;
        transform: translateY(-1px) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9CA3AF !important;
        font-weight: 400 !important;
    }
    
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label {
        font-size: 0.925rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.75rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    .form-group {
        margin-bottom: 1.75rem;
    }
    
    .form-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 1.75rem;
    }
    
    /* Professional Auth Buttons */
    .auth-btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.25) !important;
        text-transform: none !important;
        min-height: 50px !important;
        line-height: 1.5 !important;
        letter-spacing: 0.025em !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .auth-btn-primary::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        transition: left 0.5s ease;
    }
    
    .auth-btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.35) !important;
        filter: brightness(105%) !important;
    }
    
    .auth-btn-primary:hover::before {
        left: 0;
    }
    
    .auth-btn-primary:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.25) !important;
    }
    
    .auth-btn-secondary {
        background: transparent !important;
        color: #667eea !important;
        border: 2px solid #667eea !important;
        border-radius: 16px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        min-height: 46px !important;
        width: 100% !important;
    }
    
    .auth-btn-secondary:hover {
        background: #667eea !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Professional Checkbox Styling */
    .stCheckbox {
        margin: 1.5rem 0 !important;
    }
    
    .stCheckbox > label {
        display: flex !important;
        align-items: center !important;
        font-size: 0.925rem !important;
        color: var(--text-secondary) !important;
        cursor: pointer !important;
    }
    
    .stCheckbox input[type="checkbox"] {
        margin-right: 0.75rem !important;
        transform: scale(1.2) !important;
        accent-color: #667eea !important;
    }
    
    /* Enhanced Button Styling for Main App */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.25) !important;
        text-transform: none !important;
        min-height: 44px !important;
        line-height: 1.4 !important;
        letter-spacing: 0.025em !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }
    
    /* Navigation button styling */
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        font-size: 0.9rem !important;
        padding: 0.75rem 1.5rem !important;
        min-height: 44px !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    /* Primary navigation button (active state) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        font-size: 0.9rem !important;
        padding: 0.75rem 1.5rem !important;
        min-height: 44px !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    /* Navigation button hover effects */
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        color: white !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        filter: brightness(110%) !important;
    }

    /* Navigation button active/pressed effects */
    .stButton > button[kind="secondary"]:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }

    .stButton > button[kind="primary"]:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }

    /* Special styling for Sign Out button */
    .stButton > button:has-text("🚪 Sign Out") {
        background: rgba(239, 68, 68, 0.1) !important;
        border-color: rgba(239, 68, 68, 0.3) !important;
        color: #ef4444 !important;
    }

    .stButton > button:has-text("🚪 Sign Out"):hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
        color: #dc2626 !important;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.35) !important;
        filter: brightness(105%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Auth form specific button styling */
    .auth-card-body .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.25) !important;
        text-transform: none !important;
        min-height: 50px !important;
        line-height: 1.5 !important;
        letter-spacing: 0.025em !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 90% !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .auth-card-body .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 90%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        transition: left 0.5s ease;
    }
    
    .auth-card-body .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.35) !important;
        filter: brightness(105%) !important;
    }
    
    .auth-card-body .stButton > button:hover::before {
        left: 0;
    }
    
    /* Focus states for better accessibility */
    .stTextInput > div > div > input:focus-visible,
    .stSelectbox > div > div > select:focus-visible {
        outline: 2px solid #667eea !important;
        outline-offset: 2px !important;
    }
    .auth-alert-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.25);
    }
    
    .auth-alert-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.25);
    }
    
    .auth-alert-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.25);
    }
    .st-emotion-cache-zuyloh {
    border: 1px solid rgba(250, 250, 250, 0.2);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    width: 100%;
    height: 100%;
    overflow: visible;
}
    
    /* Professional Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 16px;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        background: transparent;
        color: #64748b;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 0, 0, 0.1);
        color: #475569;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.25);
    }
    
    /* Professional Links */
    .auth-link {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.2s ease;
        border-bottom: 1px solid transparent;
    }
    
    .auth-link:hover {
        color: #5a67d8;
        border-bottom-color: #5a67d8;
    }
    
    /* Loading States */
    .auth-loading {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        color: var(--text-secondary);
        font-size: 0.925rem;
        font-weight: 500;
    }
    
    .auth-spinner {
        width: 20px;
        height: 20px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-left-color: #667eea;
        border-radius: 50%;
        animation: auth-spin 1s linear infinite;
    }
    
    @keyframes auth-spin {
        to { transform: rotate(360deg); }
    }
    
    /* Professional Helper Text */
    .auth-helper {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-align: center;
        margin-top: 1.5rem;
        line-height: 1.5;
    }
    
    .auth-footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(226, 232, 240, 0.6);
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed var(--border-color);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: var(--bg-secondary);
        transition: all 0.3s ease;
        cursor: pointer;
        margin-bottom: 2rem;
    }
    
    .upload-area:hover {
        border-color: var(--primary-color);
        background: rgba(102, 126, 234, 0.05);
    }
    
    .upload-area.has-file {
        border-color: var(--success-color);
        background: rgba(16, 185, 129, 0.05);
    }
    
    .upload-icon {
        font-size: 3rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        display: block;
    }
    
    .upload-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .upload-subtext {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Form Grid */
    .form-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .form-section {
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Page Headers */
    .page-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .page-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        font-size: 1.1rem;
    }
    
    /* Enhanced Results Display with Professional Styling */
    .result-card {
        background: var(--bg-primary);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        text-align: center;
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.05);
        border: 1px solid var(--border-color);
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(20px);
    }
    
    .result-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 32px 64px -12px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1);
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        border-radius: 24px 24px 0 0;
    }
    
    .result-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        opacity: 0.03;
        pointer-events: none;
        transition: opacity 0.4s ease;
    }
    
    .result-card:hover::after {
        opacity: 0.06;
    }
    
    .result-positive {
        border-color: rgba(239, 68, 68, 0.2);
    }
    
    .result-positive::before {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .result-positive::after {
        background: radial-gradient(circle, #ef4444 0%, transparent 70%);
    }
    
    .result-negative {
        border-color: rgba(16, 185, 129, 0.2);
    }
    
    .result-negative::before {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .result-negative::after {
        background: radial-gradient(circle, #10b981 0%, transparent 70%);
    }
    
    .result-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        display: block;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: var(--text-primary);
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    
    .result-positive .result-title {
        color: #dc2626;
    }
    
    .result-negative .result-title {
        color: #059669;
    }
    
    .result-probability {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .result-probability strong {
        color: var(--text-primary);
        font-weight: 700;
    }
    
    .probability-meter {
        width: 100%;
        height: 12px;
        background: var(--bg-tertiary);
        border-radius: 6px;
        margin: 1rem 0 2rem 0;
        overflow: hidden;
        position: relative;
    }
    
    .probability-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .probability-fill::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .probability-fill.high-risk {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .probability-fill.moderate-risk {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .probability-fill.low-risk {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .risk-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%);
        color: #059669;
        border: 2px solid rgba(16, 185, 129, 0.3);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.15) 100%);
        color: #d97706;
        border: 2px solid rgba(245, 158, 11, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        color: #dc2626;
        border: 2px solid rgba(239, 68, 68, 0.3);
    }
    
    .result-details {
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border-color);
    }
    
    .result-recommendations {
        background: var(--bg-tertiary);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        text-align: left;
    }
    
    .result-recommendations h4 {
        margin: 0 0 1rem 0;
        color: var(--text-primary);
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .result-recommendations ul {
        margin: 0;
        padding-left: 1.5rem;
        color: var(--text-secondary);
    }
    
    .result-recommendations li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Responsive Design for Result Cards */
    @media (max-width: 768px) {
        .result-card {
            padding: 2rem 1.5rem;
            margin-top: 1.5rem;
            border-radius: 20px;
        }
        
        .result-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .result-title {
            font-size: 1.5rem;
        }
        
        .result-probability {
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .probability-meter {
            height: 10px;
            margin: 0.75rem 0 1.5rem 0;
        }
        
        .risk-badge {
            padding: 0.6rem 1.2rem;
            font-size: 0.85rem;
        }
        
        .result-recommendations {
            padding: 1.25rem;
            margin-top: 1.25rem;
        }
        
        .result-recommendations h4 {
            font-size: 0.95rem;
        }
        
        .result-recommendations ul {
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        .result-card {
            padding: 1.5rem 1rem;
            margin: 1rem -0.5rem 0 -0.5rem;
            border-radius: 16px;
        }
        
        .result-icon {
            font-size: 3.5rem;
        }
        
        .result-title {
            font-size: 1.25rem;
            line-height: 1.3;
        }
        
        .result-probability {
            font-size: 0.95rem;
        }
        
        .probability-meter {
            height: 8px;
        }
        
        .risk-badge {
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
        }
        
        .result-recommendations {
            padding: 1rem;
            text-align: center;
        }
        
        .result-recommendations ul {
            text-align: left;
            font-size: 0.85rem;
        }
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .risk-moderate {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .risk-high {
        background: rgba(239, 68, 68, 0.1);
        color: var(--error-color);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Stats Dashboard */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    
    .stat-card {
        background: var(--bg-primary);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
    }
    
    .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        overflow: hidden !important;
    }
    
    /* Profile Page */
    .profile-section {
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 2rem;
        align-items: start;
    }
    
    .profile-avatar {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: white;
        font-weight: 700;
        margin: 0 auto 2rem;
        box-shadow: var(--shadow-lg);
    }
    
    .profile-info {
        display: grid;
        gap: 1.5rem;
    }
    
    .info-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: var(--bg-secondary);
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
    }
    
    .info-label {
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-value {
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        font-size: 0.95rem !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { -webkit-transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .form-grid {
            grid-template-columns: 1fr;
        }
        
        .nav-content {
            flex-direction: column;
            gap: 1rem;
        }
        
        .page-container {
            padding: 1rem 0.5rem;
        }
        
        .auth-wrapper {
            padding: 1rem 0.5rem;
        }
        
        .auth-container {
            margin: 0;
            border-radius: 20px;
            box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.15);
        }
        
        .auth-card-body {
            padding: 2.5rem 2rem 2rem;
        }
        
        .auth-title {
            font-size: 1.75rem;
        }
        
        .auth-subtitle {
            font-size: 0.95rem;
        }
        
        .form-row {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .profile-section {
            grid-template-columns: 1fr;
        }
        
        .nav-pills {
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .page-title {
            font-size: 2rem;
        }
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- DOCUMENT PROCESSING ---------------- #
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        file.seek(0)
        doc = docx.Document(BytesIO(file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_medical_features(text):
    """Extract medical features from text using pattern matching"""
    import difflib
    features = {}
    # Normalize text for easier parsing
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text_lower = text.lower()

    # Enhanced: Handle column-based format (Feature\nValue pattern)
    def extract_from_columns(lines):
        """Extract features from column-based layout where feature names and values are on separate lines"""
        column_features = {}
        i = 0
        while i < len(lines) - 1:
            current_line = lines[i].lower()
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            
            # Check if current line is a feature name and next line is likely a value
            if current_line and next_line:
                # Age
                if 'age' in current_line and next_line.isdigit():
                    column_features['age'] = int(next_line)
                    i += 2
                    continue
                
                # Blood Pressure
                elif ('blood pressure' in current_line or 'resting' in current_line) and next_line.isdigit():
                    column_features['resting_bp_s'] = int(next_line)
                    i += 2
                    continue
                
                # Cholesterol
                elif 'cholesterol' in current_line and next_line.isdigit():
                    column_features['cholesterol'] = int(next_line)
                    i += 2
                    continue
                
                # Max Heart Rate
                elif ('heart rate' in current_line or 'max heart' in current_line) and next_line.isdigit():
                    column_features['max_heart_rate'] = int(next_line)
                    i += 2
                    continue
                
                # Fasting Blood Sugar
                elif 'fasting blood sugar' in current_line:
                    if 'high' in next_line.lower() or 'yes' in next_line.lower():
                        column_features['diabetes_detected'] = True
                    elif 'low' in next_line.lower() or 'normal' in next_line.lower() or 'no' in next_line.lower():
                        column_features['diabetes_detected'] = False
                    i += 2
                    continue
                
                # Resting ECG
                elif 'resting ecg' in current_line or ('ecg' in current_line and 'resting' not in current_line):
                    if 'normal' in next_line.lower():
                        column_features['resting_ecg'] = 'Normal'
                    elif 'abnormal' in next_line.lower() or 'st-t' in next_line.lower():
                        column_features['resting_ecg'] = 'ST-T Wave Abnormality'
                    elif 'hypertrophy' in next_line.lower():
                        column_features['resting_ecg'] = 'Left Ventricular Hypertrophy'
                    i += 2
                    continue
                
                # Exercise Angina
                elif 'exercise angina' in current_line or ('angina' in current_line and 'exercise' not in current_line):
                    if 'yes' in next_line.lower():
                        column_features['exercise_angina'] = 'Yes'
                    elif 'no' in next_line.lower():
                        column_features['exercise_angina'] = 'No'
                    i += 2
                    continue
                
                # Oldpeak
                elif 'oldpeak' in current_line or 'st depression' in current_line:
                    try:
                        column_features['oldpeak'] = float(next_line)
                    except:
                        pass
                    i += 2
                    continue
                
                # ST Slope
                elif 'st slope' in current_line or ('slope' in current_line and 'st' not in current_line):
                    if 'upslop' in next_line.lower():
                        column_features['st_slope'] = 'Upsloping'
                    elif 'flat' in next_line.lower():
                        column_features['st_slope'] = 'Flat'
                    elif 'downslop' in next_line.lower():
                        column_features['st_slope'] = 'Downsloping'
                    i += 2
                    continue
                
                # Risk Score
                elif 'risk score' in current_line:
                    if next_line.isdigit():
                        column_features['risk_score_simple'] = int(next_line)
                    elif 'moderate' in next_line.lower():
                        column_features['risk_score_simple'] = 5  # Assume moderate = 5
                    elif 'high' in next_line.lower():
                        column_features['risk_score_simple'] = 8  # Assume high = 8
                    elif 'low' in next_line.lower():
                        column_features['risk_score_simple'] = 2  # Assume low = 2
                    i += 2
                    continue
                
                # Chest Pain Type indicators
                elif 'chest pain type 3' in current_line:
                    column_features['chest_pain_type_3'] = 1 if 'yes' in next_line.lower() else 0
                    i += 2
                    continue
                elif 'chest pain type 4' in current_line:
                    column_features['chest_pain_type_4'] = 1 if 'yes' in next_line.lower() else 0
                    i += 2
                    continue
                
                # Age Group indicators
                elif 'age group' in current_line:
                    if 'old' in current_line and 'senior' not in current_line:
                        column_features['age_group_old'] = 1 if 'yes' in next_line.lower() else 0
                    elif 'senior' in current_line:
                        column_features['age_group_senior'] = 1 if 'yes' in next_line.lower() else 0
                    elif 'young' in current_line:
                        column_features['age_group_young'] = 1 if 'yes' in next_line.lower() else 0
                    i += 2
                    continue
            
            i += 1
        
        return column_features

    # First try column-based extraction
    column_features = extract_from_columns(lines)
    features.update(column_features)

    # Helper for fuzzy key matching (fallback method)
    def find_value_by_key(keys, lines, default=None, num_type=int):
        for line in lines:
            for key in keys:
                if key in line.lower():
                    # Try to extract number after key
                    match = re.search(rf'{re.escape(key)}[\s:=-]*([\d.]+)', line, re.IGNORECASE)
                    if match:
                        try:
                            return num_type(match.group(1))
                        except:
                            continue
        # Try table/CSV style: key,value
        for line in lines:
            for key in keys:
                if line.lower().startswith(key):
                    parts = re.split(r'[:,\t\-]+', line, maxsplit=2)
                    if len(parts) > 1:
                        try:
                            return num_type(parts[1].strip())
                        except:
                            continue
        return default

    # Fallback extraction for features not found in column format
    if 'age' not in features:
        age_keys = ['age', 'patient age', 'years old', 'age(years)', 'age (years)']
        features['age'] = find_value_by_key(age_keys, lines)
        if features['age'] is None:
            # Try free text
            match = re.search(r'(\d+)[- ]year[s]?[- ]old', text_lower)
            if match:
                features['age'] = int(match.group(1))

    # Sex/Gender (fallback)
    if 'sex' not in features:
        sex_keys = ['sex', 'gender']
        sex_val = None
        for line in lines:
            for key in sex_keys:
                if key in line.lower():
                    if 'male' in line.lower() or 'm' in line.lower():
                        sex_val = 'Male'
                    elif 'female' in line.lower() or 'f' in line.lower():
                        sex_val = 'Female'
        if not sex_val:
            if re.search(r'\b(male|man|mr\.)\b', text_lower):
                sex_val = 'Male'
            elif re.search(r'\b(female|woman|mrs\.|ms\.)\b', text_lower):
                sex_val = 'Female'
        if sex_val:
            features['sex'] = sex_val

    # Blood Pressure (fallback)
    if 'resting_bp_s' not in features:
        bp_keys = ['bp', 'blood pressure', 'resting bp', 'systolic bp']
        bp_val = find_value_by_key(bp_keys, lines)
        if bp_val is None:
            # Try 120/80 style
            match = re.search(r'(\d{2,3})[ /\\-](\d{2,3})', text_lower)
            if match:
                bp_val = int(match.group(1))
        if bp_val:
            features['resting_bp_s'] = bp_val

    # Cholesterol (fallback)
    if 'cholesterol' not in features:
        chol_keys = ['cholesterol', 'chol', 'total cholesterol']
        features['cholesterol'] = find_value_by_key(chol_keys, lines)

    # Heart Rate (fallback)
    if 'max_heart_rate' not in features:
        hr_keys = ['heart rate', 'hr', 'pulse', 'max heart rate', 'maximum heart rate']
        features['max_heart_rate'] = find_value_by_key(hr_keys, lines)
        if features['max_heart_rate'] is None:
            match = re.search(r'(\d{2,3})\s*bpm', text_lower)
            if match:
                features['max_heart_rate'] = int(match.group(1))

    # Chest pain (fallback)
    if 'chest_pain_detected' not in features:
        chest_pain_keys = ['chest pain', 'angina', 'chest discomfort', 'chest tightness', 'chest pressure']
        for line in lines:
            for key in chest_pain_keys:
                if key in line.lower():
                    features['chest_pain_detected'] = True
                    break
        if 'chest_pain_detected' not in features:
            for key in chest_pain_keys:
                if re.search(key, text_lower):
                    features['chest_pain_detected'] = True
                    break

    # Clean up: remove None values
    features = {k: v for k, v in features.items() if v is not None}
    return features

# ---------------- DATABASE FUNCTIONS ---------------- #
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP
        )
    """)
    
    # Predictions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_data TEXT,
            prediction INTEGER,
            probability REAL,
            created_at TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    # Sessions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_token TEXT UNIQUE,
            created_at TIMESTAMP,
            expires_at TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()

def add_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username,email,password,created_at) VALUES (?,?,?,?)",
                  (username, email, generate_password_hash(password), datetime.datetime.now()))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def validate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, email, created_at FROM users WHERE username=?", (username,))
    user_row = c.fetchone()
    
    if user_row:
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        password_row = c.fetchone()
        
        if password_row and check_password_hash(password_row[0], password):
            conn.close()
            return {
                "id": user_row[0], 
                "username": user_row[1], 
                "email": user_row[2],
                "created_at": user_row[3]
            }
    
    conn.close()
    return None

def save_prediction(user_id, features, pred, proba):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (user_id,input_data,prediction,probability,created_at) VALUES (?,?,?,?,?)",
              (user_id, json.dumps(features), pred, proba, datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT input_data,prediction,probability,created_at FROM predictions WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_user_stats(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Total predictions
    c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (user_id,))
    total = c.fetchone()[0]
    
    # High risk predictions
    c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND probability > 0.7", (user_id,))
    high_risk = c.fetchone()[0]
    
    # Recent predictions (last 30 days)
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND created_at > ?", (user_id, thirty_days_ago))
    recent = c.fetchone()[0]
    
    # Average risk
    c.execute("SELECT AVG(probability) FROM predictions WHERE user_id=?", (user_id,))
    avg_risk = c.fetchone()[0] or 0
    
    conn.close()
    return {
        "total": total,
        "high_risk": high_risk,
        "recent": recent,
        "avg_risk": avg_risk
    }

def save_session_to_db(user_id, session_token):
    """Save session token to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    expires_at = datetime.datetime.now() + datetime.timedelta(days=30)
    
    try:
        c.execute("INSERT INTO user_sessions (user_id, session_token, created_at, expires_at) VALUES (?,?,?,?)",
                  (user_id, session_token, datetime.datetime.now(), expires_at))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def get_user_from_session(session_token):
    """Get user info from session token"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT u.id, u.username, u.email, u.created_at
        FROM users u 
        JOIN user_sessions s ON u.id = s.user_id 
        WHERE s.session_token = ? AND s.expires_at > ?
    """, (session_token, datetime.datetime.now()))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0], 
            "username": row[1], 
            "email": row[2], 
            "created_at": row[3]
        }
    return None

def delete_session(session_token):
    """Delete session token from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
    conn.commit()
    conn.close()

def generate_session_token():
    """Generate a unique session token"""
    return secrets.token_urlsafe(32)

# ---------------- HELPER FUNCTIONS ---------------- #
def classify_risk(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    return "High"

def get_risk_color(risk_level):
    colors = {
        "Low": "#10b981",
        "Moderate": "#f59e0b", 
        "High": "#ef4444"
    }
    return colors.get(risk_level, "#6b7280")

def render_top_nav():
    if st.session_state.user:
        user = st.session_state.user
        initials = ''.join([name[0].upper() for name in user['username'].split()[:2]])
        
        nav_html = f"""
        <div class="top-nav">
            <div class="nav-content">
                <div class="nav-brand">
                    <span class="nav-logo">❤</span>
                    <div>
                        <div class="nav-title">Heart Shield</div>
                        <div class="nav-subtitle">AI-Powered Heart Disease Prediction</div>
                    </div>
                </div>
                <div class="nav-user">
                    <div class="user-info">
                        <div class="user-name">Welcome, {user['username']}</div>
                        <div class="user-role">Healthcare Professional</div>
                    </div>
                    <div class="user-avatar">{initials}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(nav_html, unsafe_allow_html=True)
    else:
        nav_html = """
        <div class="top-nav">
            <div class="nav-content">
                <div class="nav-brand">
                    <span class="nav-logo">❤</span>
                    <div>
                        <div class="nav-title">Heart Shield</div>
                        <div class="nav-subtitle">AI-Powered Heart Disease Prediction</div>
                    </div>
                </div>
                <div class="nav-user">
                    <div class="user-info">
                        <div class="user-name">Please Sign In</div>
                        <div class="user-role">Guest User</div>
                    </div>
                    <div class="user-avatar">👤</div>
                </div>
            </div>
        </div>
        """
        st.markdown(nav_html, unsafe_allow_html=True)

def render_navigation_pills():
    if st.session_state.user:
        st.markdown('<div class="nav-pills-container">', unsafe_allow_html=True)
        current_page = st.session_state.current_page
        
        cols = st.columns([1,2,2,2,1,2,1])
        
        with cols[1]:
            active = current_page == "Prediction"
            if st.button("🔬 New Analysis", key="nav_prediction", 
                        type="primary" if active else "secondary",
                        use_container_width=True):
                st.session_state.current_page = "Prediction"
                if st.session_state.session_token:
                    st.query_params["session"] = st.session_state.session_token
                    st.query_params["page"] = "prediction"
                st.rerun()
        
        with cols[2]:
            active = current_page == "History"
            if st.button("📊 Past Results", key="nav_history",
                        type="primary" if active else "secondary",
                        use_container_width=True):
                st.session_state.current_page = "History"
                if st.session_state.session_token:
                    st.query_params["session"] = st.session_state.session_token
                    st.query_params["page"] = "history"
                st.rerun()
        
        with cols[3]:
            active = current_page == "Profile"
            if st.button("👤 Account", key="nav_profile",
                        type="primary" if active else "secondary",
                        use_container_width=True):
                st.session_state.current_page = "Profile"
                if st.session_state.session_token:
                    st.query_params["session"] = st.session_state.session_token
                    st.query_params["page"] = "profile"
                st.rerun()
        
        with cols[5]:
            if st.button("🚪 Sign Out", key="nav_logout", 
                        type="secondary", use_container_width=True):
                if st.session_state.session_token:
                    delete_session(st.session_state.session_token)
                st.session_state.user = None
                st.session_state.session_token = None
                st.session_state.current_page = "Login"
                st.query_params.clear()
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_welcome_board():
    """Render the welcome board for the authentication page"""
    return """
    <div class="auth-welcome-board">
        <div class="welcome-content">
            # <span class="welcome-logo"></span>
            <h1 class="welcome-title">Heart Shield</h1>
            <p class="welcome-subtitle">
                Advanced AI-powered heart disease prediction system designed to help you take control of your cardiovascular health with precision and confidence.
            </p>
            <ul class="welcome-features">
                <li class="welcome-feature">
                    <span class="welcome-feature-icon">🧠</span>
                    <span>AI-Powered Risk Assessment</span>
                </li>
                <li class="welcome-feature">
                    <span class="welcome-feature-icon">📊</span>
                    <span>Comprehensive Health Analytics</span>
                </li>
                <li class="welcome-feature">
                    <span class="welcome-feature-icon">🔒</span>
                    <span>Secure & Private Data Protection</span>
                </li>
                <li class="welcome-feature">
                    <span class="welcome-feature-icon">📱</span>
                    <span>User-Friendly Mobile Experience</span>
                </li>
                <li class="welcome-feature">
                    <span class="welcome-feature-icon">⚡</span>
                    <span>Instant Results & Recommendations</span>
                </li>
            </ul>
        </div>
    </div>
    """

def render_auth_page():
    """Render enhanced authentication page with split-screen design"""
    
    st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
    
    # Create two columns for split-screen layout
    col_left, col_right = st.columns([1, 1])
    
    # Left side - Welcome board
    with col_left:
        st.markdown(render_welcome_board(), unsafe_allow_html=True)
    
    # Right side - Auth forms
    with col_right:
        st.markdown('<div class="auth-forms-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-card-body">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Sign In", " Create Account"])
        
        with tab1:
            # st.markdown("""
            # <div class="auth-header">
            #      <span class="auth-logo"></span>
            #     <h1 class="auth-title">Welcome Back</h1>
            #     <p class="auth-subtitle">Sign in to access your personalized heart health dashboard and continue monitoring your cardiovascular wellness.</p>
            # </div>
            # """, unsafe_allow_html=True)
            
            with st.form("login_form", clear_on_submit=False):
                st.markdown('<div class="form-group">', unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="Enter your username", key="login_username")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="form-group">', unsafe_allow_html=True)
                password = st.text_input("Password", type="password", placeholder="Enter your secure password", key="login_password")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="form-row">', unsafe_allow_html=True)
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    remember_me = st.checkbox("Remember me", key="remember_login")
                st.markdown('</div>', unsafe_allow_html=True)
                
                submitted = st.form_submit_button("Sign In to Dashboard", use_container_width=True)
                
                if submitted:
                    if username and password:
                        with st.spinner("Authenticating your credentials..."):
                            user = validate_user(username, password)
                            if user:
                                st.session_state.user = user
                                session_token = generate_session_token()
                                if save_session_to_db(user["id"], session_token):
                                    st.session_state.session_token = session_token
                                    st.query_params["session"] = session_token
                                st.session_state.current_page = "Prediction"
                                st.success("Welcome back! Redirecting to your dashboard...")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Invalid credentials. Please check your username and password.")
                    else:
                        st.warning("Please fill in all required fields.")
            
            st.markdown("""
            <div class="auth-helper">
                <p>New to Heart Shield? Switch to the <strong>Create Account</strong> tab to get started with your heart health journey.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            # st.markdown("""
            # <div class="auth-header">
            #      <span class="auth-logo">❤️</span>
            #     <h1 class="auth-title">Join Heart Shield</h1>
            #     <p class="auth-subtitle">Create your secure account to access AI-powered heart disease prediction and personalized health insights.</p>
            # </div>
            # """, unsafe_allow_html=True)
            
            with st.form("signup_form", clear_on_submit=False):
                st.markdown('<div class="form-group">', unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="Choose a unique username", key="signup_username")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="form-group">', unsafe_allow_html=True)
                email = st.text_input("Email Address", placeholder="your.email@domain.com", key="signup_email")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="form-row">', unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    password = st.text_input("Password", type="password", placeholder="Create strong password", key="signup_password")
                with col_b:
                    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="signup_confirm")
                st.markdown('</div>', unsafe_allow_html=True)
                
                agree_terms = st.checkbox("I agree to the **Terms of Service** and **Privacy Policy**", key="agree_terms")
                
                submitted = st.form_submit_button("Create My Account", use_container_width=True)
                
                if submitted:
                    if username and email and password and confirm_password:
                        if password != confirm_password:
                            st.error("Passwords do not match. Please try again.")
                        elif len(password) < 6:
                            st.error("Password must be at least 6 characters long.")
                        elif not agree_terms:
                            st.error("Please accept the Terms of Service and Privacy Policy to continue.")
                        elif not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
                            st.error("Please enter a valid email address.")
                        else:
                            with st.spinner("Creating your secure account..."):
                                if add_user(username, email, password):
                                    st.success("Account created successfully! Please sign in to continue.")
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("Username or email already exists. Please choose different credentials.")
                    else:
                        st.warning("Please fill in all required fields.")
            
            st.markdown("""
            <div class="auth-helper">
                <p>Already have an account? Switch to the <strong>Sign In</strong> tab to access your dashboard.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="auth-footer">
            <p>Your data is protected with enterprise-grade security</p>
            <p style="margin-top: 0.5rem;">© 2024 Heart Shield - AI Healthcare Solutions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Close containers
        st.markdown('</div>', unsafe_allow_html=True)  # Close auth-card-body
        st.markdown('</div>', unsafe_allow_html=True)  # Close auth-container
        st.markdown('</div>', unsafe_allow_html=True)  # Close auth-forms-container
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close auth-wrapper

def render_prediction_page():
    """Render the prediction page with document upload functionality"""
    
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Heart Disease Risk Analysis</h1>
        <p class="page-subtitle">Upload medical documents or manually enter patient data for comprehensive heart disease risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card animate-fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Document Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Medical Document", 
        type=['pdf', 'docx', 'txt'],
        help="Upload patient medical records in PDF, DOCX, or TXT format"
    )
    
    extracted_features = {}
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        with st.spinner("Analyzing document and extracting medical information..."):
            # Extract text based on file type
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.lower().endswith(".docx"):
                text = extract_text_from_docx(uploaded_file)
            else:  # txt file
                try:
                    uploaded_file.seek(0)
                    text = uploaded_file.read().decode('utf-8')
                except Exception:
                    text = ""
            
            if text:
                extracted_features = extract_medical_features(text)
                if extracted_features:
                    st.info(f"Extracted {len(extracted_features)} medical parameters from the document")
                    
                    # Show extracted features
                    with st.expander("View Extracted Information"):
                        for key, value in extracted_features.items():
                            st.write(f"{key.replace('_', ' ').title()}: **{value}**")
                else:
                    st.warning("⚠ Could not extract medical information from the document. Please fill the form manually.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Manual Input Form
    st.markdown('<div class="glass-card animate-fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
    
    # Basic Information
    st.markdown("### 👤 Basic Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age (years)", 
            min_value=20, max_value=100, 
            value=extracted_features.get('age', 50),
            help="Patient's age in years"
        )
        sex = st.selectbox(
            "Sex", 
            ["Male", "Female"],
            index=0 if extracted_features.get('sex', 'Male') == 'Male' else 1
        )
    
    with col2:
        chest_pain_type = st.selectbox(
            "Chest Pain Type", 
            ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"],
            index=0 if extracted_features.get('chest_pain_detected') else 3
        )
    
    # Vital Signs
    st.markdown("### Vital Signs & Lab Results")
    col1, col2 = st.columns(2)
    
    with col1:
        resting_bp_s = st.number_input(
            "Resting Blood Pressure (mm Hg)", 
            min_value=80, max_value=200, 
            value=extracted_features.get('resting_bp_s', 120),
            help="Systolic blood pressure at rest"
        )
        cholesterol = st.number_input(
            "Cholesterol Level (mg/dl)", 
            min_value=100, max_value=600, 
            value=extracted_features.get('cholesterol', 200),
            help="Total cholesterol level"
        )
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar", 
            ["≤ 120 mg/dl", "> 120 mg/dl"],
            index=1 if extracted_features.get('diabetes_detected') else 0
        )
    
    with col2:
        max_heart_rate = st.number_input(
            "Maximum Heart Rate Achieved", 
            min_value=60, max_value=220, 
            value=extracted_features.get('max_heart_rate', 150),
            help="Maximum heart rate during stress test"
        )
        resting_ecg = st.selectbox(
            "Resting ECG Results", 
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        )
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", 
            ["No", "Yes"]
        )
    
    # Advanced Parameters
    st.markdown("### Advanced Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)", 
            min_value=0.0, max_value=10.0, 
            value=1.0, step=0.1,
            help="ST depression induced by exercise relative to rest"
        )
        st_slope = st.selectbox(
            "ST Slope", 
            ["Upsloping", "Flat", "Downsloping"]
        )
    
    with col2:
        risk_score_simple = st.number_input(
            "Simple Risk Score", 
            min_value=0, max_value=10, 
            value=2,
            help="Basic cardiovascular risk score"
        )
    
    # Additional Model Parameters
    with st.expander("Advanced Model Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chest_pain_type_3 = st.selectbox("Chest Pain Type 3", [0, 1], help="Binary indicator for specific chest pain type")
            chest_pain_type_4 = st.selectbox("Chest Pain Type 4", [0, 1], help="Binary indicator for specific chest pain type")
        
        with col2:
            age_group_old = st.selectbox("Old Age Group", [0, 1], help="Indicator for older age group")
            age_group_senior = st.selectbox("Senior Age Group", [0, 1], help="Indicator for senior age group")
        
        with col3:
            age_group_young = st.selectbox("Young Age Group", [0, 1], help="Indicator for younger age group")
    
    # Prediction Button
    if st.button("Generate Risk Assessment", use_container_width=True, type="primary"):
        if model is None:
            st.error("Prediction model not loaded. Ensure models/rf.joblib exists and is compatible.")
        elif st.session_state.user is None:
            st.error("Please sign in to save predictions.")
        else:
            with st.spinner("Analyzing patient data with AI model..."):
                # Convert inputs to model format
                sex_val = 1 if sex == "Male" else 0
                chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal": 2, "Asymptomatic": 3}
                chest_pain_val = chest_pain_map[chest_pain_type]
                fasting_blood_sugar_val = 1 if fasting_blood_sugar == "> 120 mg/dl" else 0
                resting_ecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
                resting_ecg_val = resting_ecg_map[resting_ecg]
                exercise_angina_val = 1 if exercise_angina == "Yes" else 0
                st_slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
                st_slope_val = st_slope_map[st_slope]
                
                # Prepare features
                features = {
                "age": age,
                "resting_bp_s": resting_bp_s,
                "cholesterol": cholesterol,
                "fasting_blood_sugar": fasting_blood_sugar_val,
                "resting_ecg": resting_ecg_val,
                "max_heart_rate": max_heart_rate,
                "exercise_angina": exercise_angina_val,
                "oldpeak": oldpeak,
                "st_slope": st_slope_val,
                "risk_score_simple": risk_score_simple,
                "chest_pain_type_3": chest_pain_type_3,
                "chest_pain_type_4": chest_pain_type_4,
                "age_group_old": age_group_old,
                "age_group_senior": age_group_senior,
                "age_group_young": age_group_young,
            }
                
                # Make prediction using RF model
                try:
                    df = pd.DataFrame([features])
                    proba = model.predict_proba(df)[0, 1]
                    pred = int(model.predict(df)[0])
                    
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    proba = 0.0
                    pred = 0

                risk_category = classify_risk(proba)
                
                # Save prediction
                try:
                    save_prediction(st.session_state.user["id"], features, pred, float(proba))
                except Exception:
                    # Don't break UI if DB save fails; show warning instead.
                    st.warning("Could not save prediction to database.")

                # Display enhanced results using Streamlit components
                result_class = "result-positive" if pred == 1 else "result-negative"
                result_icon = "🚨" if pred == 1 else "✅"
                result_title = "High Risk Detected" if pred == 1 else "Low Risk Assessment"
                risk_color = get_risk_color(risk_category)
                
                # Determine probability fill class
                if risk_category.lower() == "high":
                    prob_fill_class = "high-risk"
                elif risk_category.lower() == "moderate":
                    prob_fill_class = "moderate-risk"
                else:
                    prob_fill_class = "low-risk"
                
                # Generate recommendations based on risk level
                if pred == 1:
                    recommendations = [
                        "Schedule an appointment with a cardiologist immediately",
                        "Monitor blood pressure and cholesterol levels regularly",
                        "Adopt a heart-healthy diet with reduced sodium and saturated fats",
                        "Engage in regular, moderate exercise (consult your doctor first)",
                        "Consider lifestyle modifications including stress management"
                    ]
                    rec_icon = "🚨"
                    rec_title = "Immediate Action Required"
                else:
                    recommendations = [
                        "Continue regular health checkups and screenings",
                        "Maintain a balanced diet rich in fruits and vegetables",
                        "Stay physically active with at least 150 minutes of exercise weekly",
                        "Monitor key health metrics like blood pressure and cholesterol",
                        "Avoid smoking and limit alcohol consumption"
                    ]
                    rec_icon = "💡"
                    rec_title = "Preventive Recommendations"
                
                # Create result card container
                st.markdown(f'<div class="result-card {result_class}">', unsafe_allow_html=True)
                
                # Icon and title
                st.markdown(f'<div style="text-align: center;">', unsafe_allow_html=True)
                st.markdown(f'<span class="result-icon" style="font-size: 5rem; display: block; margin-bottom: 1.5rem;">{result_icon}</span>', unsafe_allow_html=True)
                st.markdown(f'<h2 class="result-title" style="font-size: 2rem; font-weight: 800; margin-bottom: 1rem; color: {"#dc2626" if pred == 1 else "#059669"};">{result_title}</h2>', unsafe_allow_html=True)
                
                # Probability display
                st.markdown(f'<div class="result-probability" style="font-size: 1.1rem; margin-bottom: 2rem;"><strong>Risk Probability: {proba:.1%}</strong></div>', unsafe_allow_html=True)
                
                # Progress bar
                progress_color = "#ef4444" if pred == 1 else "#10b981"
                st.markdown(f"""
                <div style="width: 100%; height: 12px; background: #edf2f7; border-radius: 6px; margin: 1rem 0 2rem 0; overflow: hidden;">
                    <div style="height: 100%; width: {proba * 100}%; background: linear-gradient(135deg, {progress_color} 0%, {progress_color}dd 100%); border-radius: 6px; transition: width 1.5s ease;"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk badge
                badge_color = "#dc2626" if risk_category.lower() == "high" else "#d97706" if risk_category.lower() == "moderate" else "#059669"
                badge_bg = "rgba(220, 38, 38, 0.15)" if risk_category.lower() == "high" else "rgba(217, 119, 6, 0.15)" if risk_category.lower() == "moderate" else "rgba(5, 150, 105, 0.15)"
                
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <span style="
                        display: inline-flex;
                        align-items: center;
                        gap: 0.5rem;
                        padding: 0.75rem 1.5rem;
                        border-radius: 25px;
                        font-weight: 700;
                        font-size: 0.95rem;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        background: {badge_bg};
                        color: {badge_color};
                        border: 2px solid {badge_color}30;
                    ">{risk_category} Risk Level</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown(f"""
                <div style="
                    background: #f7fafc; 
                    border-radius: 16px; 
                    padding: 1.5rem; 
                    margin-top: 1.5rem; 
                    text-align: left;
                ">
                    <h4 style="margin: 0 0 1rem 0; color: #1a202c; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                        {rec_icon} {rec_title}
                    </h4>
                    <ul style="margin: 0; padding-left: 1.5rem; color: #718096;">
                        {''.join([f'<li style="margin-bottom: 0.5rem; line-height: 1.5;">{rec}</li>' for rec in recommendations[:3]])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close text-align center
                st.markdown('</div>', unsafe_allow_html=True)  # Close result card
                
                # Add educational section about confidence scores
                with st.expander("🎓 Understanding Prediction Confidence", expanded=False):
                    st.markdown("""
                    ### What Does Prediction Confidence Mean?
                    
                    **Confidence Score** indicates how certain the AI models are about their prediction:
                    
                    - **90-100%**: Very reliable - Multiple models strongly agree
                    - **70-89%**: Reliable - Good consensus among models  
                    - **50-69%**: Moderate - Some uncertainty, additional consultation recommended
                    - **30-49%**: Low - High uncertainty, prediction may be inaccurate
                    - **Below 30%**: Very low - Models cannot reliably predict, seek medical advice
                    
                    ### Why Might Confidence Be Low?
                    
                    1. **Incomplete Data**: Missing important health information
                    2. **Unusual Values**: Health metrics outside typical ranges
                    3. **Borderline Case**: You fall between risk categories
                    4. **Model Disagreement**: Different AI models give different predictions
                    5. **Edge Case**: Your health profile is uncommon in training data
                    
                    ### What to Do with Low Confidence?
                    
                    ✅ **Verify your input data** for accuracy  
                    ✅ **Provide complete information** in all fields  
                    ✅ **Consult a healthcare professional** for clinical assessment  
                    ✅ **Consider additional tests** like ECG, stress test, or blood work  
                    ✅ **Re-assess periodically** with updated health data  
                    
                    ### Remember
                    This AI tool is designed to **supplement**, not **replace** professional medical advice. 
                    Always consult with qualified healthcare providers for important health decisions.
                    """)
                
                # Final recommendations based on prediction
                if pred == 1:
                    st.error("🚨 **High-risk prediction**: Strong recommendation for medical consultation.")
                else:
                    st.success("✅ **Low-risk assessment**: Continue healthy lifestyle and regular checkups.")
                
                st.balloons()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_history_page():
    """Render enhanced history page with statistics"""
    
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Prediction History</h1>
        <p class="page-subtitle">Comprehensive analysis of your previous heart disease risk assessments and trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    user = st.session_state.user
    if not user:
        st.info("Please sign in to view your history.")
        return
    
    # Get user stats and history
    stats = get_user_stats(user["id"])
    history = get_history(user["id"])
    
    # Statistics Dashboard
    st.markdown('<div class="stats-grid animate-fade-in">', unsafe_allow_html=True)
    
    stat_cards = [
        ("📊", stats["total"], "Total Assessments", "primary"),
        ("⚠", stats["high_risk"], "High Risk Cases", "error"),
        ("📅", stats["recent"], "Recent (30 days)", "warning"),
        ("📈", f"{stats['avg_risk']:.1%}", "Average Risk", "success")
    ]
    
    for icon, value, label, color in stat_cards:
        st.markdown(f"""
        <div class="stat-card">
            <span class="stat-icon">{icon}</span>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if history:
        # Prepare data for display
        history_data = []
        for i, (input_data, prediction, probability, created_at) in enumerate(history):
            try:
                input_dict = json.loads(input_data)
                age = input_dict.get('age', 'N/A')
                sex = 'Male' if input_dict.get('sex', 0) == 1 else 'Female'
            except:
                age = 'N/A'
                sex = 'N/A'
            
            history_data.append({
                "Date": created_at,
                "Age": age,
                "Sex": sex,
                "Result": "Heart Disease Risk" if prediction == 1 else "Low Risk",
                "Probability": f"{probability:.1%}",
                "Risk Level": classify_risk(probability),
                "Details": f"Assessment #{len(history) - i}"
            })
        
        df = pd.DataFrame(history_data)
        
        # History Table
        st.markdown('<div class="glass-card animate-fade-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Assessment History</div>', unsafe_allow_html=True)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.selectbox("Filter by Risk Level", ["All", "Low", "Moderate", "High"])
        with col2:
            result_filter = st.selectbox("Filter by Result", ["All", "Heart Disease Risk", "Low Risk"])
        with col3:
            limit = st.selectbox("Show Records", [10, 25, 50, "All"])
        
        # Apply filters
        filtered_df = df.copy()
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df["Risk Level"] == risk_filter]
        if result_filter != "All":
            filtered_df = filtered_df[filtered_df["Result"] == result_filter]
        if limit != "All":
            filtered_df = filtered_df.head(limit)
        
        if len(filtered_df) > 0:
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export functionality
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"heart_assessment_history_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("🔍 No records match the selected filters.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="glass-card">
            <div style="text-align: center; padding: 3rem;">
                <span style="font-size: 4rem;">📊</span>
                <h3>No Assessment History</h3>
                <p>You haven't made any predictions yet. Start by creating your first heart disease risk assessment!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_profile_page():
    """Render enhanced profile page"""
    
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Profile Dashboard</h1>
        <p class="page-subtitle">Manage your account settings and view your healthcare analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    user = st.session_state.user
    if not user:
        st.info("Please sign in to view your profile.")
        return

    # Fetch stats here so we can show them
    stats = get_user_stats(user["id"])
    
    # Use local layout columns properly
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown('<div class="profile-avatar">', unsafe_allow_html=True)
        initials = ''.join([n[0].upper() for n in user['username'].split()[:2]])
        st.markdown(f"<div style='text-align:center;font-size:2.5rem;font-weight:700;color:white'>{initials}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Account Information</div>', unsafe_allow_html=True)
        info_items = [
            ("👤", "Username", user['username']),
            ("📧", "Email Address", user.get('email', 'Not provided')),
            ("📅", "Member Since", user.get('created_at', 'Recently')[:10] if user.get('created_at') else 'Recently'),
            ("🔐", "Account Status", "Active & Secured"),
            ("📊", "Total Assessments", str(stats['total'])),
            ("⚠", "High Risk Cases", str(stats['high_risk'])),
            ("📈", "Average Risk Score", f"{stats['avg_risk']:.1%}")
        ]
        for icon, label, value in info_items:
            st.markdown(f"""
            <div class="info-item">
                <div class="info-label">
                    <span>{icon}</span>
                    {label}
                </div>
                <div class="info-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown('<div class="glass-card animate-fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Quick Actions</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # New Assessment button 
        if st.button("🔬 New Assessment", use_container_width=True, help="Start a new heart disease risk assessment"):
            st.session_state.current_page = "Prediction"
            if st.session_state.session_token:
                st.query_params["session"] = st.session_state.session_token
                st.query_params["page"] = "prediction"
            # Show temporary success message
            placeholder = st.empty()
            placeholder.success("🔬 Starting new health assessment...")
            time.sleep(0.5)  # Brief pause for user feedback
            placeholder.empty()
            st.rerun()
    with col2:
        # View History button
        if st.button("📊 View History", use_container_width=True, help="See all your past health assessments"):
            # Check if user has history first
            history_count = len(get_history(user["id"]))
            if history_count > 0:
                st.session_state.current_page = "History"
                if st.session_state.session_token:
                    st.query_params["session"] = st.session_state.session_token
                    st.query_params["page"] = "history"
                # Show temporary success message
                placeholder = st.empty()
                placeholder.success(f"📊 Loading your assessment history ({history_count} records)...")
                time.sleep(0.5)  # Brief pause for user feedback
                placeholder.empty()
                st.rerun()
            else:
                st.info("📝 No assessment history found. Complete your first health assessment to start tracking your health data!")
                # Offer to redirect to assessment
                if st.button("🔬 Start Your First Assessment", use_container_width=True, key="first_assessment"):
                    st.session_state.current_page = "Prediction"
                    if st.session_state.session_token:
                        st.query_params["session"] = st.session_state.session_token
                        st.query_params["page"] = "prediction"
                    st.rerun()
    with col3:
        # Export Data with improved functionality
        export_clicked = st.button("📥 Export Data", use_container_width=True, help="Download all your health assessment data")
        if export_clicked:
            with st.spinner("Preparing your data export..."):
                history = get_history(user["id"])
                if history:
                    history_data = []
                    for input_data, prediction, probability, created_at in history:
                        try:
                            input_dict = json.loads(input_data)
                            # Create comprehensive export data
                            row_data = {
                                "Assessment_Date": created_at,
                                "Username": user['username'],
                                "Result": "Heart Disease Risk Detected" if prediction == 1 else "Low Risk Assessment",
                                "Risk_Probability": f"{probability:.3f}",
                                "Risk_Percentage": f"{probability:.1%}",
                                "Risk_Category": classify_risk(probability),
                                # Include all input parameters
                                "Age": input_dict.get('age', ''),
                                "Sex": "Male" if input_dict.get('sex', 0) == 1 else "Female",
                                "Chest_Pain_Type": input_dict.get('cp', ''),
                                "Resting_Blood_Pressure": input_dict.get('trestbps', ''),
                                "Cholesterol": input_dict.get('chol', ''),
                                "Fasting_Blood_Sugar": "Yes" if input_dict.get('fbs', 0) == 1 else "No",
                                "Resting_ECG_Results": input_dict.get('restecg', ''),
                                "Max_Heart_Rate": input_dict.get('thalach', ''),
                                "Exercise_Induced_Angina": "Yes" if input_dict.get('exang', 0) == 1 else "No",
                                "ST_Depression": input_dict.get('oldpeak', ''),
                                "ST_Slope": input_dict.get('slope', ''),
                                "Major_Vessels": input_dict.get('ca', ''),
                                "Thalassemia": input_dict.get('thal', '')
                            }
                            history_data.append(row_data)
                        except json.JSONDecodeError:
                            # Fallback for corrupted data
                            row_data = {
                                "Assessment_Date": created_at,
                                "Username": user['username'],
                                "Result": "Heart Disease Risk Detected" if prediction == 1 else "Low Risk Assessment",
                                "Risk_Probability": f"{probability:.3f}",
                                "Risk_Percentage": f"{probability:.1%}",
                                "Risk_Category": classify_risk(probability),
                                "Note": "Detailed input data not available"
                            }
                            history_data.append(row_data)
                    
                    df = pd.DataFrame(history_data)
                    
                    # Create CSV with metadata header
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    csv_content = f"# Heart Shield - Health Assessment Export\n"
                    csv_content += f"# Generated on: {timestamp}\n"
                    csv_content += f"# User: {user['username']}\n"
                    csv_content += f"# Total Records: {len(history_data)}\n"
                    csv_content += f"# Export Note: This data is for personal health tracking only\n\n"
                    csv_content += df.to_csv(index=False)
                    
                    # Create filename with timestamp
                    filename = f"HeartShield_Export_{user['username']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    st.success(f"✅ Export ready! {len(history_data)} records prepared for download.")
                    st.download_button(
                        label="📥 Download Complete Health Data",
                        data=csv_content,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True,
                        help=f"Download {len(history_data)} assessment records with complete details"
                    )
                else:
                    st.info("📊 No assessment data available for export yet. Complete your first health assessment to start building your data history!")
    with col4:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN APP CONTROLLER ---------------- #
def main():
    st.set_page_config(
        page_title="Heart Shield - AI Health Assistant", 
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    load_css()
    init_db()

    # Initialize session state defaults
    if "user" not in st.session_state:
        st.session_state.user = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Login"
    if "session_token" not in st.session_state:
        st.session_state.session_token = None

    # ✅ Use st.query_params instead of deprecated experimental_get_query_params
    query_params = st.query_params

    if st.session_state.user is None and st.session_state.session_token is None:
        session_token = None
        if "session" in query_params:
            val = query_params.get("session")
            session_token = val[0] if isinstance(val, list) else val
        if session_token:
            user = get_user_from_session(session_token)
            if user:
                st.session_state.user = user
                st.session_state.session_token = session_token
                st.session_state.current_page = "Prediction"

    # Handle logout via URL
    if "logout" in query_params:
        if st.session_state.session_token:
            delete_session(st.session_state.session_token)
        st.session_state.user = None
        st.session_state.session_token = None
        st.session_state.current_page = "Login"
        st.query_params.clear()  # ✅ reset query params
        st.rerun()

    # Handle page routing via URL param
    if "page" in query_params and st.session_state.user:
        pg = query_params.get("page")
        pg_val = pg[0] if isinstance(pg, list) else pg
        if pg_val and pg_val.lower() in ["prediction", "history", "profile"]:
            st.session_state.current_page = pg_val.title()

    # Render navigation and appropriate page
    render_top_nav()
    
    if not st.session_state.user:
        # Don't render navigation pills when user is not logged in
        render_auth_page()
    else:
        # Only render navigation pills when user is logged in
        render_navigation_pills()
        page = st.session_state.current_page or "Prediction"
        if page == "Prediction":
            render_prediction_page()
        elif page == "History":
            render_history_page()
        elif page == "Profile":
            render_profile_page()
        else:
            render_prediction_page()
if __name__ == "__main__":
    main()
