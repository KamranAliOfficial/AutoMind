
"""
AutoMind ML - Ultimate Edition (Restored & Optimized)
Features: Gradient UI, SHAP, Hyperparameters, PDF Report.
Optimized for: 8GB RAM Laptops.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, confusion_matrix

# Advanced ML (Restored)
try: 
    import shap
    HAS_SHAP = True
except: 
    HAS_SHAP = False

# PDF Report
try: from fpdf import FPDF
except: FPDF = None

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoMind ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL PLOT THEME ---
plt.rcParams.update({
    "figure.facecolor": "#0F1420",
    "axes.facecolor": "#0F1420",
    "axes.edgecolor": "#2196F3",
    "axes.labelcolor": "#CBD5E1",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#2A3B55",
    "text.color": "#FFFFFF",
    "figure.dpi": 300,
    "grid.alpha": 0.3
})
sns.set_palette("viridis")

# -----------------------------------------------------------------------------
# 2. PREMIUM UI/UX SYSTEM (Ultra-Optimized with Better Readability)
# -----------------------------------------------------------------------------
def inject_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Outfit:wght@300;500;700;900&family=Inter:wght@400;600;800&display=swap');
        
        /* DESIGN TOKENS */
        :root {
            --primary: #2196F3;
            --secondary: #6C5CE7;
            --accent: #00D4FF;
            --bg-dark: #0a0e17;
            --bg-card: rgba(20, 30, 48, 0.6);
            --glass-border: rgba(255,255,255,0.08);
            --text-primary: #FFFFFF;
            --text-secondary: #CBD5E1;
            --neon-blue: #00D4FF;
            --neon-purple: #BC13FE;
        }

        /* 3D PERSPECTIVE CONTAINER */
        .stApp {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
            perspective: 1000px;
            font-family: 'Inter', sans-serif;
        }
        
        /* 3D GLASS CARDS WITH TILT EFFECT */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 24px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
            transform-style: preserve-3d;
        }
        
        .glass-card:hover {
            transform: translateY(-5px) scale(1.01) rotateX(2deg);
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.4),
                0 0 20px rgba(33, 150, 243, 0.2);
            border-color: rgba(33, 150, 243, 0.3);
        }

        /* NEON GLOW TEXT */
        h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 3.5rem !important;
            font-weight: 900 !important;
            background: linear-gradient(to right, #2196F3, #00D4FF, #BC13FE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 15px rgba(33, 150, 243, 0.5));
            animation: glow-text 3s infinite alternate;
        }
        
        @keyframes glow-text {
            from { filter: drop-shadow(0 0 10px rgba(33, 150, 243, 0.4)); }
            to { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.8)); }
        }

        /* 3D BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, rgba(33,150,243,0.8), rgba(108,92,231,0.8)) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: white !important;
            border-radius: 12px !important;
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: 1.5px;
            padding: 0.8rem 2.5rem !important;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 
                0 4px 6px rgba(0,0,0,0.1),
                inset 0 1px 0 rgba(255,255,255,0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 10px 20px rgba(33, 150, 243, 0.4),
                0 0 15px var(--neon-blue);
            border-color: var(--neon-blue) !important;
        }
        
        .stButton > button:active {
            transform: translateY(1px);
        }

        /* INPUT FIELDS - FUTURISTIC HUD STYLE */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background: rgba(0, 0, 0, 0.4) !important;
            border: 1px solid rgba(33, 150, 243, 0.2) !important;
            color: var(--neon-blue) !important;
            font-family: 'Courier New', monospace !important;
            border-radius: 8px !important;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            box-shadow: 0 0 15px rgba(33, 150, 243, 0.3) !important;
            border-color: var(--neon-blue) !important;
            transform: scale(1.01);
        }

        /* METRIC CARDS WITH FLOATING ANIMATION */
        .stMetric {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 16px !important;
            padding: 20px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }
        
        .stMetric label { color: var(--text-muted) !important; font-family: 'Orbitron'; }
        .stMetric div[data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
            text-shadow: 0 0 10px rgba(33, 150, 243, 0.5);
            font-size: 2.2rem !important;
        }

        /* SIDEBAR FROSTED GLASS */
        section[data-testid="stSidebar"] {
            background: rgba(10, 14, 23, 0.85);
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* CUSTOM SCROLLBAR */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0a0e17; }
        ::-webkit-scrollbar-thumb { 
            background: linear-gradient(180deg, #2196F3, #6C5CE7); 
            border-radius: 4px;
        }
        
        /* DATAFRAME TECH STYLE */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(33, 150, 243, 0.3);
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(33, 150, 243, 0.1);
        }
        
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. ADVANCED LOGIC (OPTIMIZED SHAP & PIPELINE)
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df.loc[:, ~df.columns.str.contains('^Unnamed')]
    except: return None

@st.cache_data
def run_shap_explainer(_model, X, sample_size=50):
    """Safely runs SHAP on a tiny sample to prevent crashing."""
    if not HAS_SHAP: return None
    try:
        # Sample background data
        bg = X.sample(min(sample_size, len(X)), random_state=42)
        
        # Check model type for Explainer
        model_step = _model.named_steps['clf']
        pre_step = _model.named_steps['pre']
        
        # Transform background
        bg_trans = pre_step.transform(bg)
        feature_names = None
        
        # Recover feature names if possible (complex with ColumnTransformer)
        try:
            feature_names = pre_step.get_feature_names_out()
        except:
            pass

        if hasattr(model_step, 'feature_importances_'):
            explainer = shap.TreeExplainer(model_step)
            shap_values = explainer.shap_values(bg_trans)
            return shap_values, bg_trans, feature_names
        else:
            # Linear models
            explainer = shap.LinearExplainer(model_step, bg_trans)
            shap_values = explainer.shap_values(bg_trans)
            return shap_values, bg_trans, feature_names
    except Exception as e:
        return None, str(e), None

# -----------------------------------------------------------------------------
# 4. INIT & LOADING
# -----------------------------------------------------------------------------

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

inject_ui()

# Session State
keys = ['df', 'df_clean', 'model', 'target', 'task', 'metrics', 'feats', 'intro']
for k in keys: 
    if k not in st.session_state: st.session_state[k] = None

# 🎬 CINEMATIC PRO LOADING (Sci-Fi Style)
if not st.session_state.get('intro'):
    import time
    
    # Custom CSS for the loader
    st.markdown("""
    <style>
        .loader-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding-top: 50px;
            animation: fadeIn 1s ease-in;
        }
        
        .logo-text {
            font-family: 'Outfit', sans-serif;
            font-size: 4.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #FFFFFF 0%, #2196F3 50%, #6C5CE7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
            text-shadow: 0 0 30px rgba(33, 150, 243, 0.5);
            animation: pulse-text 2s infinite alternate;
        }
        
        .ring-loader {
            position: relative;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #00D4FF;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
            animation: spin 1s linear infinite;
        }
        
        .ring-loader::before {
            content: '';
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            bottom: 10px;
            border-radius: 50%;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #6C5CE7;
            animation: spin-reverse 2s linear infinite;
        }
        
        .ring-loader::after {
            content: '';
            position: absolute;
            top: 25px;
            left: 25px;
            right: 25px;
            bottom: 25px;
            border-radius: 50%;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #2196F3;
            animation: spin 3s linear infinite;
        }
        
        .status-text {
            margin-top: 40px;
            font-family: 'Courier New', monospace;
            font-size: 1.2rem;
            color: #00D4FF;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes spin-reverse {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(-360deg); }
        }
        
        @keyframes pulse-text {
            from { text-shadow: 0 0 20px rgba(33, 150, 243, 0.4); }
            to { text-shadow: 0 0 40px rgba(33, 150, 243, 0.8), 0 0 80px rgba(108, 92, 231, 0.6); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

    # Full screen container
    placeholder = st.empty()
    
    with placeholder.container():
        st.markdown('<div class="loader-container">', unsafe_allow_html=True)
        st.markdown('<div class="logo-text">AUTOMIND ML</div>', unsafe_allow_html=True)
        st.markdown('<div class="ring-loader"></div>', unsafe_allow_html=True)
        
        status = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Simulated Boot Sequence
        steps = [
            ("⚡ INITIALIZING NEURAL CORE", 0.7),
            ("📡 ESTABLISHING UPLINK", 0.5),
            ("💎 LOADING PREMIUM ASSETS", 0.6),
            ("🧬 OPTIMIZING ALGORITHMS", 0.6),
            ("🚀 SYSTEM READY", 0.4)
        ]
        
        for text, duration in steps:
            status.markdown(f'<div style="text-align: center;"><p class="status-text">{text}...</p></div>', unsafe_allow_html=True)
            time.sleep(duration)
            
    time.sleep(0.3)
    placeholder.empty()
    st.session_state.intro = True
    st.rerun()

# -----------------------------------------------------------------------------
# 5. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ AutoMind")
    
    # Navigation
    menu = st.radio("Navigation", 
        ["Dashboard", "Data Analysis", "PCA Analysis", "Model Training", "Prediction", "Report"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Status Indicator (From Screenshot)
    st.markdown("""
    <div style="background: rgba(33, 150, 243, 0.1); border: 1px solid #2196F3; border-radius: 4px; padding: 10px; margin-bottom: 20px;">
        <span style="color: #2196F3; font-weight: 600;">Status: Ready</span>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. MAIN CONTENT
# -----------------------------------------------------------------------------

# --- DASHBOARD ---
if menu == "Dashboard":
    # Load Profile Image safely
    profile_img_b64 = ""
    if os.path.exists("assets/profile.jpg"):
        try:
            profile_img_b64 = f"data:image/jpg;base64,{get_img_as_base64('assets/profile.jpg')}"
        except: pass
    
    # HERO SECTION (Matching Image 1)
    st.markdown(f"""
    <style>
        .hero-card {{
            background: linear-gradient(135deg, #1e2a4a 0%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .hero-text {{
            color: white;
            flex: 1;
        }}
        .hero-title {{
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 15px;
            background: linear-gradient(90deg, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero-sub {{
            font-size: 1.1rem;
            color: #94A3B8;
            margin-bottom: 25px;
            line-height: 1.6;
            max-width: 600px;
        }}
        .pill {{
            background: rgba(255,255,255,0.1);
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 0.8rem;
            letter-spacing: 1px;
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
            display: inline-block;
        }}
        .hero-img-container {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            padding: 4px;
            background: linear-gradient(135deg, #2196F3, #6C5CE7);
            box-shadow: 0 0 20px rgba(33, 150, 243, 0.4);
            margin-left: 40px;
            flex-shrink: 0;
            overflow: hidden;
            display: flex; /* Center image */
            justify-content: center;
            align-items: center;
        }}
        .hero-img {{
            width: 100%;
            height: 100%;
            border-radius: 50%;
            object-fit: cover;
            border: 4px solid #0f172a;
        }}
    </style>
    
    <div class="hero-card">
        <div class="hero-text">
            <div class="hero-title">AutoMind ML</div>
            <div class="hero-sub">
                An intelligent, end-to-end Data Science platform utilizing advanced AutoML 
                algorithms to clean, analyze, and model your data automatically.
            </div>
            <div class="pill">MADE BY KAMRAN</div>
        </div>
        <div class="hero-img-container">
            <img src="{profile_img_b64}" class="hero-img"  
            onerror="this.style.display='none'">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Data Ingestion")
    f = st.file_uploader("Upload Dataset (CSV)", type="csv")
    if f:
        # Prevent reloading/resetting df_clean on every interaction unless file changes
        file_id = f"{f.name}-{f.size}"
        if st.session_state.get("file_id") != file_id:
            df = load_data(f)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_clean = df.copy()
                st.session_state.file_id = file_id
                st.session_state.metrics = None # Reset model metrics
                st.session_state.model = None
                st.success("New Dataset Ingested.")
        
        # Always show current df state
        df = st.session_state.df
        if df is not None:
            st.dataframe(df.head(), use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Features", df.shape[1])
            try:
                comp = 100 - (df.isnull().sum().sum()/df.size)*100
                c3.metric("Completeness", f"{comp:.1f}%")
            except: c3.metric("Completeness", "N/A")
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- DATA ANALYSIS (CLEANING + EDA) ---
elif menu == "Data Analysis":
    st.title("Data Intelligence Hub")
    df = st.session_state.get('df')
    
    if df is not None:
        tabs = st.tabs(["Cleaning & Repairs", "Deep EDA"])
        
        with tabs[0]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Use df_clean if available, else fallback to df
            current_df = st.session_state.get('df_clean', df)
            
            c1, c2 = st.columns(2)
            c1.metric("Missing Values", current_df.isnull().sum().sum())
            c2.metric("Duplicate Rows", current_df.duplicated().sum())
            
            if st.button("Auto-Repair Dataset", type="primary"):
                try:
                    clean = df.drop_duplicates()
                    num = clean.select_dtypes(include=np.number).columns
                    cat = clean.select_dtypes(exclude=np.number).columns
                    
                    if len(num) > 0:
                        clean[num] = clean[num].fillna(clean[num].median())
                    
                    for c in cat: 
                        mode_val = clean[c].mode()
                        fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
                        clean[c] = clean[c].fillna(fill_val)
                        
                    st.session_state.df_clean = clean
                    st.success("Dataset Successfully Repaired.")
                    st.rerun() # Force refresh to update metrics
                except Exception as e:
                    st.error(f"Repair Failed: {e}")
            
            st.dataframe(current_df.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[1]:
            st.subheader("Distribution Analysis")
            df_c = st.session_state.get('df_clean', df)
            num = df_c.select_dtypes(include=np.number).columns
            
            if len(num)>0:
                sel = st.selectbox("Select Feature for Distribution", num)
                try:
                    fig, ax = plt.subplots(figsize=(10,4))
                    # Sample only if large
                    plot_data = df_c[sel].dropna()
                    if len(plot_data) > 2000: plot_data = plot_data.sample(2000)
                    
                    sns.histplot(plot_data, kde=True, color="#2196F3", ax=ax)
                    fig.patch.set_alpha(0)
                    ax.set_facecolor("none")
                    ax.tick_params(colors="white")
                    for s in ax.spines.values(): s.set_color("#444")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot distribution: {e}")
                
                st.subheader("Correlation Matrix")
                if len(num) > 1:
                    try:
                        fig2, ax2 = plt.subplots(figsize=(8,6))
                        corr = df_c[num].corr()
                        sns.heatmap(corr, annot=True, cmap="mako", ax=ax2, fmt=".2f")
                        fig2.patch.set_alpha(0)
                        ax2.tick_params(colors="white", which='both')
                        st.pyplot(fig2)
                    except Exception as e:
                        st.warning(f"Correlation plot failed: {e}")
                else:
                    st.info("Need at least 2 numeric features for correlation.")
            else:
                st.warning("No numeric features found for analysis.")
# --- PCA ---
elif menu == "PCA Analysis":
    st.title("Dimensionality Reduction (PCA)")
    df = st.session_state.df_clean
    if df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        target = st.selectbox("Select Target Variable", df.columns)
        st.session_state.target = target
        
        # Task Detect
        if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10: 
            st.session_state.task = "Regression"
        else: 
            st.session_state.task = "Classification"
        st.info(f"Task Detected: **{st.session_state.task}**")
        
        if st.button("Generate 2D Projection"):
            with st.spinner("Projecting..."):
                try:
                    # Optimized PCA
                    # 1. Force convert all potential numeric columns (some might be read as objects)
                    X_raw = df.drop(columns=[target])
                    for col in X_raw.columns:
                        try:
                            X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
                        except: pass
                    
                    # 2. Select only valid numeric columns
                    X = X_raw.select_dtypes(include=np.number).dropna(axis=1, how='all').fillna(0) # Logic: drop cols that became ALL NaN
                    
                    if X.shape[1] > 1:
                        # Standardize
                        scaler = StandardScaler()
                        X_sc = scaler.fit_transform(X)
                        pca = PCA(n_components=2)
                        comps = pca.fit_transform(X_sc)
                        
                        res = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                        res['Target'] = df[target].values
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(8,6))
                        sns.scatterplot(data=res, x='PC1', y='PC2', hue='Target', palette='viridis', ax=ax)
                        fig.patch.set_alpha(0)
                        ax.set_facecolor("none")
                        ax.tick_params(colors="white")
                        st.pyplot(fig)
                    else:
                        st.warning("Insufficient numeric features for PCA.")
                except Exception as e:
                    st.error(f"PCA Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- MODEL TRAINING ---
elif menu == "Model Training":
    st.title("Neural Model Lab")
    df = st.session_state.df_clean
    if df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Ensure target is selected (should be from PCA, but allow re-select)
        target = st.selectbox("Target Variable", df.columns, index=df.columns.get_loc(st.session_state.target) if st.session_state.target in df.columns else 0)
        st.session_state.target = target
        
        # 0. LOGIC: Auto-Detect Task Type (Regression vs Classification)
        # Critical to update this if user changes target here!
        if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10: 
            st.session_state.task = "Regression"
        else: 
            st.session_state.task = "Classification"
        
        task = st.session_state.task
        st.write(f"Configuring **{task}** Pipeline")
        
        c1, c2 = st.columns(2)
        algo = c1.selectbox("Algorithm", ["Random Forest", "Linear/Logistic Regression"])
        n_est = c2.slider("Trees (RF Only)", 10, 100, 50) if "Forest" in algo else 0
        
        if st.button("Train Model", type="primary"):
            with st.spinner("🚀 Training on Neural Engine..."):
                try:
                    # 1. OPTIMIZATION: High Cardinality Check
                    high_card_cols = []
                    cat_cols = df.select_dtypes(exclude=np.number).columns
                    for c in cat_cols:
                        if c == target: continue # Skip target
                        if df[c].nunique() > 50:
                            high_card_cols.append(c)
                    
                    if high_card_cols:
                        df = df.drop(columns=high_card_cols)
                        st.warning(f"⚠️ Dropped high-cardinality columns to speed up: {high_card_cols}")

                    # 2. OPTIMIZATION: Data Sampling ...
                    if len(df) > 10000:
                        df = df.sample(10000, random_state=42)
                        
                    # 3. SAFETY: Drop Unstructured Text (Long Descriptions)
                    # Solves 'could not convert string to float' if OHE tries to process job descriptions
                    text_drop_cols = []
                    cat_candidates = df.select_dtypes(include=['object', 'string']).columns
                    for c in cat_candidates:
                        if c == target: continue
                        # Check average AND max string length
                        try:
                            s = df[c].astype(str).str.len()
                            if s.mean() > 50 or s.max() > 200: # Aggressive drop for job descriptions
                                text_drop_cols.append(c)
                        except: pass
                    
                    if text_drop_cols:
                        df = df.drop(columns=text_drop_cols)
                        st.warning(f"⚠️ Dropped unstructured text columns: {text_drop_cols}")

                    X = df.drop(columns=[target])
                    y = df[target]
                    
                    # Preprocessing
                    num = X.select_dtypes(include=np.number).columns
                    cat = X.select_dtypes(exclude=np.number).columns
                    
                    # FORCE CONVERSION for Safety
                    # Sklearn sometimes chokes if mixed types exist
                    for c in cat: X[c] = X[c].astype(str)
                    for c in num: X[c] = pd.to_numeric(X[c], errors='coerce')
                    
                    # 3. OPTIMIZATION: Efficient Encoders
                    pre = ColumnTransformer([
                        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num),
                        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))]), cat)
                    ])
                    
                    # Model & Encode
                    le = None
                    if task == "Classification":
                         le = LabelEncoder()
                         y = le.fit_transform(y)
                         st.session_state.le = le
                         # n_jobs=-1 uses ALL CPU cores
                         if "Forest" in algo: m = RandomForestClassifier(n_estimators=n_est, max_depth=10, n_jobs=-1)
                         else: m = LogisticRegression(max_iter=500, n_jobs=-1)
                    else:
                         if "Forest" in algo: m = RandomForestRegressor(n_estimators=n_est, max_depth=10, n_jobs=-1)
                         else: m = LinearRegression(n_jobs=-1)
                    
                    pipe = Pipeline([('pre', pre), ('clf', m)])
                    
                    # Train/Test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                    
                    # Metrics
                    res = {}
                    if task == "Classification":
                        res['Accuracy'] = accuracy_score(y_test, y_pred)
                        res['F1 Score'] = f1_score(y_test, y_pred, average='weighted')
                    else:
                        res['R2 Score'] = r2_score(y_test, y_pred)
                        res['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                    st.session_state.model = pipe
                    st.session_state.metrics = res
                    st.session_state.feats = X.columns.tolist()
                    st.session_state.score = list(res.values())[0] # For report
                    
                    st.success(f"✅ Model Trained Successfully in {algo}")
                    st.json(res)
                    
                except Exception as e:
                    st.error(f"Training Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION ---
elif menu == "Prediction":
    st.title("Real-time Inference")
    if st.session_state.model:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Ensure form is properly structured
        with st.form("pred_form"):
            st.subheader("Input Values")
            params = {}
            feats = st.session_state.feats
            
            # Create columns for inputs
            cols = st.columns(3)
            
            # Limit to 15 inputs to prevent UI explosion
            display_feats = feats[:15]
            
            for i, f in enumerate(display_feats): 
                with cols[i % 3]:
                     params[f] = st.text_input(f, "0")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Submit Button MUST be inside the form
            submitted = st.form_submit_button("🔮 Predict Result", type="primary")
            
            if submitted:
                try:
                    idf = pd.DataFrame([params])
                    for c in idf.columns:
                        try: idf[c] = idf[c].astype(float)
                        except: pass
                    
                    # Add missing columns (0) for safety
                    for f in feats: 
                        if f not in idf.columns: idf[f] = 0
                    
                    # Order columns exactly like training
                    idf = idf[feats]
                    
                    pred = st.session_state.model.predict(idf)
                    val = pred[0]
                    
                    if st.session_state.task == "Classification" and st.session_state.le:
                        try:
                            val = st.session_state.le.inverse_transform([int(val)])[0]
                        except: pass
                        
                    st.success(f"### Predicted Outcome: {val}")
                    
                    # INSIGHTS: Explanation for Model Bias
                    with st.expander("ℹ️ Why is the result not 0?", expanded=False):
                        st.markdown("""
                        **Understanding Model Bias:**
                        1. **Intercept (Bias):** Most models have a starting "Baseline Value" even when inputs are zero.
                        2. **Scaling:** Since we used `StandardScaler` (Z-Score), an input of **0** is converted to the **Average Value** of the training data, not mathematical zero. 
                        
                        *Essentially, the model is predicting based on the average/baseline scenario.*
                        """)
                    
                except Exception as e:
                    st.error(f"Inference Error: {e}")
                    
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ Please Train a Model first in the 'Model Training' tab.")

# --- REPORT ---
elif menu == "Report":
    st.title("Academic Report Generation")
    
    if st.session_state.metrics and FPDF:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("Generate a standardized academic report summarizing the entire pipeline.")
        
        if st.button("Download PDF Report"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "AutoMind ML - Project Report", ln=True)
                pdf.ln(5)
                
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, f"Task Type: {st.session_state.task}", ln=True)
                pdf.cell(0, 10, f"Target Variable: {st.session_state.target}", ln=True)
                pdf.ln(5)
                
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Model Performance", ln=True)
                pdf.set_font("Arial", '', 12)
                for k,v in st.session_state.metrics.items():
                    pdf.cell(0, 10, f"{k}: {v:.4f}", ln=True)
                
                pdf.ln(10)
                pdf.cell(0, 10, f"Generated by AutoMind ML (Kamran Ali)", ln=True)
                
                out = pdf.output(dest='S').encode('latin-1')
                st.download_button("Click to Download", out, "AutoMind_Report.pdf", "application/pdf")
            except Exception as e:
                st.error(f"PDF Gen Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    elif not FPDF:
        st.warning("FPDF library missing.")
    else:
        st.warning("No Model Metrics found. Please train a model.")
