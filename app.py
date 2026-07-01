import os
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from backend.city_analysis import analyze_city

# Configure page metadata and layout
st.set_page_config(
    page_title="Urban Growth Intelligence Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling injection for rich UI aesthetics
st.markdown("""
<style>
    /* Main body background color styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Metrics panel card styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #ff4b4b;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Navigation Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161a24;
    }
    
    /* Header card wrapper styling */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-card h1 {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .header-card p {
        color: #94a3b8;
        font-size: 1.2rem;
    }
    
    /* Technical specification pill badges */
    .tech-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background-color: #334155;
        color: #f8fafc;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 0.2rem;
        border: 1px solid #475569;
    }
    
    /* Workflow step item box styling */
    .workflow-step {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define project path directories
PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

# Sidebar navigation structure
st.sidebar.markdown("## 🏙️ Navigation")
page = st.sidebar.radio(
    "Select Screen:",
    ["Home", "Analyze City", "Compare Cities", "About"]
)

# Display a unified header
st.markdown("""
<div class="header-card">
    <h1>Urban Growth Intelligence Platform</h1>
    <p>AI-powered Urban Growth Prediction using Satellite Imagery and OpenStreetMap Features</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SCREEN 1: HOME -----------------
if page == "Home":
    st.markdown("## 🏠 Platform Overview")
    st.markdown(
        "Welcome to the **Urban Growth Intelligence Platform**. This systems-level geospatial AI application "
        "orchestrates remote sensing satellite data, OpenStreetMap features, and classical machine learning "
        "classifiers to predict and analyze urban sprawl at grid-level granularity."
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚙️ Multi-Stage Processing Pipeline")
        st.markdown(
            """
            The backend engine automatically executes a modular data-engineering workflow to ingest and transform features:
            
            1. **Administrative Boundary Geocoding** - Geocodes city polygons via Nominatim coordinate systems.
            2. **OSM Spatial Data Downloader** - Pulls infrastructure networks (buildings, road lines, highways).
            3. **OSM Cleanup & Grid Generation** - Discretizes administrative regions into regular spatial cells.
            4. **Sentinel-2 Multi-Spectral Query** - Ingests surface reflectance imagery from Google Earth Engine.
            5. **Index Computations (NDVI/NDBI/NDWI)** - Calculates vegetation, built-up, and moisture indexes.
            6. **Zonal Statistical Extraction** - Computes spatial features and temporal change deltas.
            7. **Predictive Analytics Classifier** - Feeds features to the optimized baseline XGBoost classifier.
            8. **Explainability Driver** - Generates local explanation drivers using SHAP.
            """
        )
        
    with col2:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("This platform integrates industry-standard scientific libraries:")
        
        techs = [
            "Python 3.10", "XGBoost", "Google Earth Engine", "OSMnx", 
            "GeoPandas", "Shapely", "Rasterio", "SHAP", "PyTorch", "Pytest", "Streamlit"
        ]
        
        badge_html = "".join([f'<span class="tech-badge">{t}</span>' for t in techs])
        st.markdown(badge_html, unsafe_allow_html=True)
        
        st.markdown("### 📊 Platform Metrics & Generalization")
        st.markdown(
            "- **Final ML Classifier:** XGBoost Baseline\n"
            "- **Accuracy Score:** `95.76%`\n"
            "- **Precision (macro):** `95.74%`\n"
            "- **Recall (macro):** `95.75%`\n"
            "- **F1 Score (macro):** `95.74%`\n"
            "- **Generalization CV Macro F1:** `0.9637` (std = `0.0095`)"
        )

# ----------------- SCREEN 2: ANALYZE CITY -----------------
elif page == "Analyze City":
    st.markdown("## 🔍 Perform City Ingestion & Analysis")
    st.markdown(
        "Choose an existing pre-cached city or input any custom municipal boundary. "
        "The system will execute the processing pipeline dynamically, update progress, and generate predictions."
    )
    
    analysis_type = st.radio(
        "Select target city type:",
        ["Option 1: Choose pre-processed existing city", "Option 2: Analyze a new city name"]
    )
    
    city_name = ""
    if analysis_type == "Option 1: Choose pre-processed existing city":
        city_name = st.selectbox("Existing Cities:", ["Bengaluru", "Hyderabad", "Pune"])
    else:
        city_name = st.text_input("Enter city name (e.g. Mysuru, Chennai, Mumbai, Delhi):", placeholder="Mysuru")
        
    if st.button("🚀 Analyze City"):
        if not city_name:
            st.error("Please enter a valid city name.")
        else:
            # Setup interactive streamlit status callbacks
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def st_progress_callback(step, progress, status):
                progress_bar.progress(progress / 100.0)
                status_text.markdown(f"**Current Stage:** `{step}` ... **{progress}%**")
                
            try:
                # Call analysis pipeline
                summary = analyze_city(city_name, status_callback=st_progress_callback)
                
                st.success(f"Analysis completed successfully for {city_name}!")
                
                # Render Metrics panel Cards
                st.markdown("### 📈 Prediction Summary Statistics")
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric("Total Spatial Cells", f"{summary['number_of_grids']:,}")
                m_col2.metric("Avg Growth Score", f"{summary['average_growth_score']:.4f}")
                m_col3.metric("High Growth Cells", f"{summary['high_growth_count']:,}")
                m_col4.metric("Medium Growth Cells", f"{summary['medium_growth_count']:,}")
                m_col5.metric("Low Growth Cells", f"{summary['low_growth_count']:,}")
                
                # Load predictions table
                pred_csv_path = Path(summary["prediction_file"])
                if pred_csv_path.exists():
                    st.markdown("### 📋 Prediction Table Results")
                    df_preds = pd.read_csv(pred_csv_path)
                    
                    # Display table
                    st.dataframe(
                        df_preds.style.format({
                            "growth_score": "{:.4f}",
                            "prediction_probability": "{:.2%}"
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Add Download Link
                    with open(pred_csv_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Full Predictions CSV",
                            data=f,
                            file_name=f"{city_name.lower()}_predictions.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Failed to execute analysis pipeline for {city_name}. Error details: {e}")

# ----------------- SCREEN 3: COMPARE CITIES -----------------
elif page == "Compare Cities":
    st.markdown("## 📊 Cross-City Analytics Comparison")
    st.markdown(
        "Load and compare summary expansion distributions across previously processed cities."
    )
    
    # Identify processed cities directories
    processed_cities = []
    if PREDICTIONS_DIR.exists():
        processed_cities = [d.name for d in PREDICTIONS_DIR.iterdir() if d.is_dir()]
        
    if not processed_cities:
        st.warning("No cities have been processed yet. Go to 'Analyze City' to generate predictions first.")
    else:
        selected_cities = st.multiselect(
            "Select cities to compare:",
            options=processed_cities,
            default=processed_cities[:min(len(processed_cities), 3)]
        )
        
        if not selected_cities:
            st.info("Please select at least one city to begin comparison.")
        else:
            comparison_records = []
            distribution_records = {}
            
            for city in selected_cities:
                summary_json_path = PREDICTIONS_DIR / city / "prediction_summary.json"
                if summary_json_path.exists():
                    with open(summary_json_path, "r") as f:
                        summary = json.load(f)
                    
                    total = summary["number_of_grids"]
                    high_pct = (summary["high_growth_count"] / total) * 100 if total > 0 else 0
                    med_pct = (summary["medium_growth_count"] / total) * 100 if total > 0 else 0
                    low_pct = (summary["low_growth_count"] / total) * 100 if total > 0 else 0
                    
                    comparison_records.append({
                        "City": city,
                        "Total Grids": total,
                        "Avg Growth Score": summary["average_growth_score"],
                        "High Growth Cells": summary["high_growth_count"],
                        "Medium Growth Cells": summary["medium_growth_count"],
                        "Low Growth Cells": summary["low_growth_count"]
                    })
                    
                    distribution_records[city] = {
                        "Low": low_pct,
                        "Medium": med_pct,
                        "High": high_pct
                    }
                    
            if comparison_records:
                df_compare = pd.DataFrame(comparison_records)
                st.markdown("### 📋 General Comparison Metrics")
                st.dataframe(
                    df_compare.style.format({
                        "Avg Growth Score": "{:.4f}",
                        "Total Grids": "{:,}",
                        "High Growth Cells": "{:,}",
                        "Medium Growth Cells": "{:,}",
                        "Low Growth Cells": "{:,}"
                    }),
                    use_container_width=True
                )
                
                # Render comparative bar charts
                st.markdown("### 📊 Urban Growth Category Distributions (%)")
                df_dist = pd.DataFrame(distribution_records).T
                st.bar_chart(df_dist, use_container_width=True)
                
# ----------------- SCREEN 4: ABOUT -----------------
elif page == "About":
    st.markdown("## ℹ️ Technical & Methodology Details")
    st.markdown(
        """
        The **Urban Growth Intelligence Platform** is built to predict and evaluate urban development boundaries. 
        
        ### 📡 Remote Sensing & Spectral Modeling
        By utilizing **Sentinel-2 Harmonized Surface Reflectance imagery**, the platform calculates the following multi-spectral indicators:
        - **Normalized Difference Vegetation Index (NDVI)**: Filters green spaces to evaluate vegetation clearing.
        - **Normalized Difference Built-Up Index (NDBI)**: Measures asphalt and structural density.
        - **Normalized Difference Water Index (NDWI)**: Monitors lakes and wetlands to calculate hydrological change.
        
        These indexes are computed for both a baseline year (2019) and the analysis year (2026), generating absolute and normalized difference deltas.
        
        ### 🏢 Infrastructure Spatial Networks
        Using OpenStreetMap datasets, localized infrastructure density parameters are compiled inside a 1000m × 1000m grid cell framework:
        - Building Count & Footprint Area Ratio.
        - Roadway Networks Length & Density.
        - Highway distances and municipal city center coordinates indices.
        
        ### 🤖 XGBoost Baseline Model
        Predictions are made using an optimized gradient-boosted tree framework (`XGBoost Classifier`) containing 300 base estimator trees and max depth constraints. This configuration provides maximum generalization accuracy on complex, high-dimensional tabular datasets.
        """
    )
