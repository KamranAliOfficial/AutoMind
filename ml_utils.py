import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error

class MLUtils:
    @staticmethod
    def load_data(uploaded_file):
        """Loads CSV data safely."""
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            return None

    @staticmethod
    def clean_data(df):
        """Removes duplicates and basic cleaning."""
        try:
            # Drop duplicates
            df = df.drop_duplicates()
            # We don't drop NA here, we impute in pipeline, 
            # but for EDA we might want a clean version? 
            # The prompt asks for cleaning: missing values (mean/median/mode).
            # We will rely on the pipeline for imputation to avoid data leakage,
            # but we can do a pass for display purposes or simple cleaning if requested.
            # Let's keep it simple: Pipeline handles imputation during training.
            return df
        except Exception as e:
            return df

    @staticmethod
    def generate_insights(df, target_column=None):
        """Generates simple text insights."""
        insights = []
        try:
            insights.append(f"• Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            
            missing = df.isnull().sum().sum()
            if missing > 0:
                insights.append(f"• Total missing values found: {missing}. These will be handled automatically.")
            else:
                insights.append(f"• No missing values detected. Data is clean.")

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            insights.append(f"• Features: {len(num_cols)} numeric, {len(cat_cols)} categorical.")
            
            if target_column:
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    skew = df[target_column].skew()
                    if abs(skew) > 1:
                        insights.append(f"• Target variable '{target_column}' is skewed (Skewness: {round(skew, 2)}).")
        except:
            pass
        return insights

    @staticmethod
    def preprocess_data(df, target_column, task_type):
        """
         Pipeline:
         - Imputes missing values (Median for num, Mode for cat)
         - Encodes categorical features (OneHot)
         - Scales numeric features
        """
        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                verbose_feature_names_out=False
            )

            target_encoder = None
            if task_type == 'Classification':
                if y.dtype == 'object' or y.dtype == 'category':
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test, preprocessor, target_encoder

        except Exception as e:
            raise Exception(f"Preprocessing failed: {str(e)}")

    @staticmethod
    def train_models(X_train, y_train, X_test, y_test, preprocessor, task_type):
        """Sequential training with specific models."""
        results = []
        trained_models = {}

        if task_type == 'Regression':
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10),
                "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
                "XGBoost": XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            }

        for name, model in models.items():
            try:
                clf = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', model)])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                metrics = {}
                metrics["Model"] = name
                if task_type == 'Regression':
                    metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
                    metrics["MAE"] = mean_absolute_error(y_test, y_pred)
                    metrics["R2 Score"] = r2_score(y_test, y_pred)
                else:
                    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
                    metrics["Precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics["Recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics["F1 Score"] = f1_score(y_test, y_pred, average='weighted')
                
                results.append(metrics)
                trained_models[name] = clf
            except Exception as e:
                print(f"Failed to train {name}: {e}")

        return pd.DataFrame(results), trained_models

    @staticmethod
    def calculate_shap(model_pipeline, X_train, X_test, task_type):
        """Calculates SHAP values with hard limit 100 samples."""
        try:
            model = model_pipeline.named_steps['classifier']
            preprocessor = model_pipeline.named_steps['preprocessor']
            
            # Preprocess data
            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"Feature {i}" for i in range(X_train_transformed.shape[1])]

            # DataFrame for SHAP
            X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

            # LIMIT TO 100 SAMPLES
            if len(X_train_df) > 100:
                X_summary = shap.kmeans(X_train_df, 10) 
            else:
                X_summary = X_train_df

            X_explain = X_test_df.iloc[:100]

            model_name = model.__class__.__name__
            
            if 'Linear' in model_name or 'Logistic' in model_name:
                 explainer = shap.LinearExplainer(model, X_summary)
            else:
                 explainer = shap.TreeExplainer(model)
                 
            shap_values = explainer.shap_values(X_explain)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            return explainer, shap_values, X_explain, feature_names

        except Exception as e:
            print(f"SHAP Error: {e}")
            return None, None, None, None

    @staticmethod
    def generate_pdf_report(session_state):
        """Generates a professional PDF report."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Header
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(79, 70, 229) # Brand Color
        pdf.cell(0, 15, txt="AutoMind ML - Project Report", ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, txt="Made by Kamran", ln=True, align='C')
        pdf.ln(10)
        
        # Dataset Overview
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, txt="1. Dataset Overview", ln=True)
        pdf.set_font("Arial", size=10)
        
        df = session_state['df']
        pdf.cell(0, 5, txt=f"Filename: {session_state.get('filename', 'dataset.csv')}", ln=True)
        pdf.cell(0, 5, txt=f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns", ln=True)
        pdf.cell(0, 5, txt=f"Target Variable: {session_state.get('target_column', 'N/A')}", ln=True)
        pdf.cell(0, 5, txt=f"Task Type: {session_state.get('task_type', 'N/A')}", ln=True)
        pdf.ln(5)

        # Insights
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="2. Automatic Insights", ln=True)
        pdf.set_font("Arial", size=10)
        insights = MLUtils.generate_insights(df, session_state.get('target_column'))
        for insight in insights:
             pdf.cell(0, 5, txt=insight, ln=True)
        pdf.ln(5)
        
        # Model Performance
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="3. Model Leaderboard", ln=True)
        pdf.set_font("Arial", size=10)
        
        if 'leaderboard' in session_state:
            lb = session_state['leaderboard']
            # Header
            col_width = 35
            for col in lb.columns:
                pdf.cell(col_width, 8, str(col), border=1, align='C')
            pdf.ln()
            # Rows
            for index, row in lb.iterrows():
                for col in lb.columns:
                    val = str(round(row[col], 3)) if isinstance(row[col], (int, float)) else str(row[col])
                    pdf.cell(col_width, 8, val, border=1, align='C')
                pdf.ln()
        
        pdf.ln(5)
        
        # Best Model
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="4. Best Performing Model", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, txt=f"Selected Model: {session_state.get('best_model_name', 'None')}", ln=True)
        
        # Footer
        pdf.set_y(-15)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"Generated by AutoMind ML | Developer: Kamran", 0, 0, 'C')
        
        return pdf.output(dest='S').encode('latin-1')
