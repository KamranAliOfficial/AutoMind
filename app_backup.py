
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

# -----------------------------------------------------------------------------
# 2. RESTORED "HEAVY" UI (OPTIMIZED)
# -----------------------------------------------------------------------------
def inject_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Inter:wght@400;600&display=swap');
        
        :root {
            --primary: #6C5CE7;
            --secondary: #A66CFF;
            --bg-dark: #0F1116;
            --card-glass: rgba(22, 27, 34, 0.85);
            --text-main: #FFFFFF;
            --text-sub: #94A3B8;
        }

        /* ANIMATED BACKGROUND (Restored) */
        .stApp {
            background-color: var(--bg-dark);
            background-image: radial-gradient(circle at 10% 20%, rgba(108, 92, 231, 0.15) 0%, transparent 20%),
                              radial-gradient(circle at 90% 80%, rgba(166, 108, 255, 0.15) 0%, transparent 20%);
            font-family: 'Inter', sans-serif;
        }
        
        /* 3D CARDS (Optimized: No Tilt JS, just CSS) */
        .glass-card {
            background: var(--card-glass);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: slideUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.3);
            border-color: var(--primary);
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* NEON BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s;
            text-transform: uppercase;
            font-size: 0.85rem;
        }
        .stButton > button:hover {
            box-shadow: 0 0 15px rgba(108, 92, 231, 0.6);
            transform: scale(1.02);
        }
        
        /* HEADERS */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            color: white !important;
        }
        h1 {
            background: linear-gradient(to right, #fff, #a29bfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem !important;
        }
        
        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: rgba(15, 17, 22, 0.95);
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        
        /* HIDE DEFAULTS */
        footer, header { visibility: hidden; }
        
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

# Professional Loading Animation (Shows once per session, non-blocking)
if not st.session_state.get('intro'):
    with st.empty():
        st.markdown("""
        <style>
        .intro-container {
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: #0F1116;
        }
        .intro-title {
            font-size: 4rem;
            font-weight: 800;
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
            margin-bottom: 10px;
        }
        .intro-sub {
            font-size: 0.9rem;
            color: #2196F3;
            letter-spacing: 4px;
            text-transform: uppercase;
            margin-bottom: 40px;
            font-weight: 600;
        }
        .loading-bar-container {
            width: 300px;
            height: 4px;
            background-color: #1F2937;
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        .loading-bar {
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, #2196F3, #6C5CE7);
            border-radius: 2px;
            animation: fillBar 2s ease-out forwards;
            transform-origin: left;
        }
        .loading-text {
            color: #94A3B8;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fillBar {
            0% { transform: scaleX(0); }
            100% { transform: scaleX(1); }
        }
        
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        </style>
        
        <div class="intro-container">
            <div class="intro-title">AutoMind ML</div>
            <div class="intro-sub">MADE BY KAMRAN</div>
            
            <div class="loading-bar-container">
                <div class="loading-bar"></div>
            </div>
            <div class="loading-text">Initialize System... 100%</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2.2)  # Wait for animation to complete
    st.session_state.intro = True
    st.rerun()  # Force a clean rerun to show main app

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
        df = load_data(f)
        if df is not None:
            st.session_state.df = df
            st.session_state.df_clean = df.copy() 
            st.success("Dataset Ingested.")
            st.dataframe(df.head(), use_container_width=True)
            
            c1,c2,c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Features", df.shape[1])
            c3.metric("Completeness", f"{100 - (df.isnull().sum().sum()/df.size)*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# --- DATA ANALYSIS (CLEANING + EDA) ---
elif menu == "Data Analysis":
    st.title("Data Intelligence Hub")
    df = st.session_state.df
    
    if df is not None:
        tabs = st.tabs(["Cleaning & Repairs", "Deep EDA"])
        
        with tabs[0]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Missing Values", df.isnull().sum().sum())
            c2.metric("Duplicate Rows", df.duplicated().sum())
            
            if st.button("Auto-Repair Dataset", type="primary"):
                clean = df.drop_duplicates()
                num = clean.select_dtypes(include=np.number).columns
                cat = clean.select_dtypes(exclude=np.number).columns
                clean[num] = clean[num].fillna(clean[num].median())
                for c in cat: clean[c] = clean[c].fillna(clean[c].mode()[0] if len(clean[c].mode())>0 else "Unk")
                st.session_state.df_clean = clean
                st.success("Dataset Successfully Repaired.")
                st.dataframe(clean.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[1]:
            st.subheader("Distribution Analysis")
            df_c = st.session_state.df_clean if st.session_state.df_clean is not None else df
            num = df_c.select_dtypes(include=np.number).columns
            if len(num)>0:
                sel = st.selectbox("Select Feature for Distribution", num)
                fig, ax = plt.subplots(figsize=(10,4))
                sns.histplot(df_c[sel].sample(min(2000, len(df_c))), kde=True, color="#2196F3", ax=ax)
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")
                ax.tick_params(colors="white")
                for s in ax.spines.values(): s.set_color("#444")
                st.pyplot(fig)
                
                st.subheader("Correlation Matrix")
                fig2, ax2 = plt.subplots(figsize=(8,6))
                sns.heatmap(df_c[num].corr(), annot=True, cmap="mako", ax=ax2, fmt=".2f")
                fig2.patch.set_alpha(0)
                ax2.tick_params(colors="white", which='both')
                st.pyplot(fig2)

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
                    X = df.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)
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
        
        task = st.session_state.task if st.session_state.task else "Classification"
        st.write(f"Configuring **{task}** Pipeline")
        
        c1, c2 = st.columns(2)
        algo = c1.selectbox("Algorithm", ["Random Forest", "Linear/Logistic Regression"])
        n_est = c2.slider("Trees (RF Only)", 10, 100, 50) if "Forest" in algo else 0
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training..."):
                try:
                    X = df.drop(columns=[target])
                    y = df[target]
                    
                    # Preprocessing
                    num = X.select_dtypes(include=np.number).columns
                    cat = X.select_dtypes(exclude=np.number).columns
                    
                    pre = ColumnTransformer([
                        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num),
                        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
                    ])
                    
                    # Model & Encode
                    le = None
                    if task == "Classification":
                         le = LabelEncoder()
                         y = le.fit_transform(y)
                         st.session_state.le = le
                         if "Forest" in algo: m = RandomForestClassifier(n_estimators=n_est, max_depth=10)
                         else: m = LogisticRegression(max_iter=500)
                    else:
                         if "Forest" in algo: m = RandomForestRegressor(n_estimators=n_est, max_depth=10)
                         else: m = LinearRegression()
                    
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
                    
                    st.success("Model Successfully Trained.")
                    st.json(res)
                    
                except Exception as e:
                    st.error(f"Training Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION ---
elif menu == "Prediction":
    st.title("Real-time Inference")
    if st.session_state.model:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.form("pred_form"):
            st.subheader("Input Values")
            params = {}
            cols = st.columns(3)
            feats = st.session_state.feats
            
            for i, f in enumerate(feats[:15]): # Limit to 15 inputs
                with cols[i%3]:
                     params[f] = st.text_input(f, "0")
            
            if st.form_submit_button("Predict Result"):
                try:
                    idf = pd.DataFrame([params])
                    for c in idf.columns:
                        try: idf[c] = idf[c].astype(float)
                        except: pass
                    
                    # Add missing columns (0)
                    for f in feats: 
                        if f not in idf.columns: idf[f] = 0
                    
                    pred = st.session_state.model.predict(idf)
                    val = pred[0]
                    
                    if st.session_state.task == "Classification" and st.session_state.le:
                        val = st.session_state.le.inverse_transform([int(val)])[0]
                        
                    st.markdown(f"### Predicted Outcome: <span style='color:#6C5CE7'>{val}</span>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Inference Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please Train a Model first.")

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
