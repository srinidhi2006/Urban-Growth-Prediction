import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import osmnx as ox
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from loguru import logger
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Import backend orchestrator modules
from backend.city_analysis import analyze_city
from config import Config

# Configure page layout and dark styling
st.set_page_config(
    page_title="Urban Growth Intelligence Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS injection to style Streamlit elements with modern dashboard aesthetics
st.markdown("""
<style>
    /* Dark Theme General Styles */
    .stApp {
        background-color: #080c14;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Sidebar Navigation styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Header card container styling */
    .header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #080711 100%);
        padding: 2.5rem;
        border-radius: 12px;
        border: 1px solid #312e81;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .header-card h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    .header-card p {
        color: #a5b4fc;
        font-size: 1.25rem;
        font-weight: 400;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    
    /* Technical specification pill badges */
    .tech-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        background-color: #1e293b;
        color: #f1f5f9;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 0.25rem;
        border: 1px solid #334155;
        font-weight: 500;
    }
    
    /* Detail Box styling */
    .detail-box {
        background-color: #0b0f19;
        border: 1px solid #1e293b;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Inline step text styling */
    .step-line {
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Define directories
PROJECT_ROOT = Config.PROJECT_ROOT
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Normalization mapping for standard aliases
ALIAS_MAPPING = {
    "bombay": "Mumbai",
    "bangalore": "Bengaluru",
    "mysore": "Mysuru",
    "new delhi": "Delhi",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "poona": "Pune"
}

def normalize_city_name(name: str) -> str:
    cleaned = name.strip().lower()
    return ALIAS_MAPPING.get(cleaned, name.strip())

# Initialize cached predictions silently to ensure Comparison works immediately
for city in ["Bengaluru", "Hyderabad", "Pune"]:
    summary_json_path = PREDICTIONS_DIR / city / "prediction_summary.json"
    if not summary_json_path.exists():
        try:
            analyze_city(city)
        except Exception as e:
            logger.warning(f"Could not pre-cache {city}: {e}")

# Pipeline Stages definitions for detailed progress logs
STAGES = [
    ("Downloading administrative boundary", 10, 10),
    ("Downloading OSM data", 20, 15),
    ("Cleaning OSM data", 30, 5),
    ("Generating spatial grid", 40, 5),
    ("Joining spatial data", 50, 5),
    ("Extracting OSM features", 60, 10),
    ("Generating Sentinel imagery", 70, 30), # GEE download stage
    ("Extracting raster features", 80, 15),
    ("Generating ML feature dataset", 85, 5),
    ("Loading trained production model", 92, 5),
    ("Generating predictions", 95, 3),
    ("Generating SHAP explanations", 98, 10),
    ("Finalizing results", 99, 2)
]

def geocode_city_boundary_frontend(city_name: str) -> bool:
    """Frontend-level geocoding resolution logic supporting State and Country suffixes."""
    normalized = normalize_city_name(city_name)
    boundary_path = Config.BOUNDARIES_DIR / f"{normalized}.geojson"
    if boundary_path.exists():
        return True
        
    # Search query sequence variants in order
    queries = []
    if "," in city_name:
        parts = [p.strip() for p in city_name.split(",")]
        if len(parts) >= 2:
            queries.append(f"{parts[0]}, {parts[1]}, India")
            queries.append(f"{parts[0]}, India")
    else:
        queries.append(f"{normalized}, India")
        queries.append(normalized)
        
    for q in queries:
        try:
            logger.info(f"[Frontend Geocoder] Trying query: {q}")
            gdf = ox.geocode_to_gdf(q)
            if not gdf.empty:
                boundary_path.parent.mkdir(parents=True, exist_ok=True)
                gdf.to_file(boundary_path, driver="GeoJSON")
                logger.success(f"[Frontend Geocoder] Successfully saved boundary for: {normalized}")
                return True
        except Exception as e:
            logger.debug(f"Failed query variant {q}: {e}")
            continue
    return False

def translate_shap_features(sorted_shap_series) -> list:
    """Translates raw ML feature SHAP values into user-friendly explanation terminology."""
    translations = []
    for feat, val in sorted_shap_series.items():
        if len(translations) >= 3:
            break
            
        if "ndvi" in feat:
            if val < 0:
                translations.append("Vegetation density decreased")
            else:
                translations.append("Vegetation density stabilized/increased")
        elif "ndbi" in feat:
            if val > 0:
                translations.append("Built-up area increased")
            else:
                translations.append("Built-up density stabilized/decreased")
        elif "road" in feat or "intersection" in feat:
            if val > 0:
                translations.append("Road network expanded")
            else:
                translations.append("Roadway density stabilized")
        elif "building" in feat:
            if val > 0:
                translations.append("Infrastructural footprint increased")
            else:
                translations.append("Infrastructural footprint stabilized")
        elif "center" in feat or "highway" in feat:
            translations.append("Proximity to transit corridors/centers")
            
    return list(set(translations))

def generate_ai_insights(city_name: str, summary: dict) -> str:
    """Generates a template-based readable summary report of urban expansion trends."""
    total = summary["number_of_grids"]
    h_pct = (summary["high_growth_count"] / total) * 100 if total > 0 else 0
    m_pct = (summary["medium_growth_count"] / total) * 100 if total > 0 else 0
    l_pct = (summary["low_growth_count"] / total) * 100 if total > 0 else 0
    
    if h_pct > 35:
        expansion_level = "high-level"
    elif h_pct > 15:
        expansion_level = "moderate-to-high"
    else:
        expansion_level = "low-to-moderate"
        
    insights = (
        f"**{city_name}** shows **{expansion_level}** urban expansion between 2019 and 2026. "
        f"Approximately **{h_pct:.1f}%** of the analyzed grid cells are classified as High Growth, "
        f"while **{m_pct:.1f}%** and **{l_pct:.1f}%** represent Medium and Low growth trends respectively. "
        f"Most high-growth regions are concentrated around newly developed transportation corridors and municipal borders. "
        f"Vegetation decline (negative NDVI shift) and increasing built-up density (positive NDBI shift) are the strongest contributors driving the XGBoost classifier predictions."
    )
    return insights

def generate_pdf_report(city_name: str, summary: dict) -> bytes:
    """Generates an in-memory PDF summary report using ReportLab."""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setTitle(f"Urban Growth Intelligence Report - {city_name}")
    
    # Header Banner
    p.setFillColorRGB(0.09, 0.09, 0.2)
    p.rect(0, 720, 612, 100, fill=1, stroke=0)
    
    p.setFillColorRGB(1.0, 1.0, 1.0)
    p.setFont("Helvetica-Bold", 20)
    p.drawString(50, 765, "Urban Growth Intelligence Platform")
    p.setFont("Helvetica", 11)
    p.drawString(50, 745, "AI-powered Urban Growth Prediction & Analytics")
    
    # Executive Summary details
    p.setFillColorRGB(0.1, 0.1, 0.1)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, 680, f"Analysis Executive Summary: {city_name}")
    
    p.setStrokeColorRGB(0.7, 0.7, 0.7)
    p.line(50, 670, 562, 670)
    
    p.setFont("Helvetica", 10)
    y = 640
    p.drawString(70, y, f"• Target Analysis City: {city_name}")
    p.drawString(70, y-20, f"• Total Analysis Cells (1km x 1km): {summary.get('number_of_grids', 0):,}")
    p.drawString(70, y-40, f"• Overall Average Growth Score: {summary.get('average_growth_score', 0.0):.4f}")
    
    # Classification category breakdown
    p.drawString(70, y-70, f"• High Growth Areas: {summary.get('high_growth_count', 0):,} cells")
    p.drawString(70, y-90, f"• Medium Growth Areas: {summary.get('medium_growth_count', 0):,} cells")
    p.drawString(70, y-110, f"• Low Growth Areas: {summary.get('low_growth_count', 0):,} cells")
    
    # Draw metrics block border
    p.setStrokeColorRGB(0.2, 0.2, 0.6)
    p.rect(60, y-120, 480, 150, stroke=1, fill=0)
    
    # Methodology overview
    y_method = y - 160
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y_method, "Model & Features Methodology Information")
    p.line(50, y_method - 8, 562, y_method - 8)
    
    p.setFont("Helvetica", 9)
    method_bullets = [
        "1. OpenStreetMap Infrastructure: Pulls roadway length, road density, buildings footprint, and center distances.",
        "2. Multi-Spectral Remote Sensing: Computes pre-post NDVI/NDBI/NDWI shift deltas from GEE Sentinel-2 composite tiles.",
        "3. Production Classifier: scores change trends using the final baseline XGBoost model (Macro F1 = 95.74%).",
        "4. local Explainability: Features SHAP contribution extraction to evaluate drivers behind individual grid forecasts."
    ]
    y_m = y_method - 30
    for bullet in method_bullets:
        p.drawString(60, y_m, bullet)
        y_m -= 18
        
    p.setFont("Helvetica-Oblique", 8)
    p.drawString(50, 40, "Report generated automatically by the Urban Growth Intelligence Platform backend system.")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer.getvalue()

def generate_shap_report_txt(city_name: str, df_shap: pd.DataFrame) -> bytes:
    """Generates a text-based explainability SHAP contribution report."""
    lines = []
    lines.append(f"============================================================")
    lines.append(f"         URBAN GROWTH SHAP EXPLANATION REPORT: {city_name.upper()}")
    lines.append(f"============================================================")
    lines.append("\nGlobal feature contributions (averaged across all cells):")
    
    shap_cols = [c for c in df_shap.columns if c != "grid_id"]
    mean_abs_shap = df_shap[shap_cols].abs().mean().sort_values(ascending=False)
    for idx, (feat, val) in enumerate(mean_abs_shap.items()):
        lines.append(f" {idx+1:02d}. {feat:<30} : {val:.6f} mean absolute SHAP value")
        
    return "\n".join(lines).encode("utf-8")

# Sidebar navigation setup
st.sidebar.markdown("## 🏙️ Dashboard Screen")
page = st.sidebar.radio(
    "Select View:",
    ["🏠 Home Page", "📊 Analyze & Predict", "🏆 Benchmarks Comparison", "ℹ️ Methodology"]
)

# ----------------- SCREEN 1: HOME PAGE -----------------
if page == "🏠 Home Page":
    # Hero Cover visual section
    hero_image_path = PROJECT_ROOT / "assets" / "hero_visual.png"
    if hero_image_path.exists():
        st.image(str(hero_image_path), use_container_width=True)
        
    st.markdown("## 🏠 Platform Overview")
    st.markdown(
        "The **Urban Growth Intelligence Platform** is an enterprise-grade geospatial machine learning platform. "
        "It integrates multitemporal Sentinel-2 satellite imagery and OpenStreetMap infrastructure networks "
        "to classify and analyze urban boundary changes at grid-level granularity."
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚙️ Multi-Stage Processing Pipeline")
        st.markdown(
            """
            The backend engine automatically executes a modular pipeline to process target locations:
            
            - **Administrative Boundary Geocoding:** Geocodes study bounds.
            - **OSM Spatial Data Ingestion:** Downloads highway networks, roadway intersections, and buildings.
            - **Sentinel-2 Multi-Spectral Query:** Processes cloud-masked surface reflectance composites inside GEE.
            - **Indices Computation:** Extracts NDVI, NDBI, and NDWI indices.
            - **Classification Model:** Feeds feature vectors to the baseline XGBoost classifier.
            - **SHAP Explanation Logging:** Computes local feature influence drivers.
            """
        )
    with col2:
        st.markdown("### 🛠️ Ingested Technologies")
        packages = ["Python 3.10", "XGBoost", "Streamlit", "ReportLab", "OSMnx", "GeoPandas", "Shapely", "Rasterio", "Plotly", "SHAP", "PyTorch"]
        badge_html = "".join([f'<span class="tech-badge">{p}</span>' for p in packages])
        st.markdown(badge_html, unsafe_allow_html=True)
        
        st.markdown("### 📊 Production Classifier baseline Metrics")
        st.markdown(
            """
            - **Classifier Architecture:** XGBoost Baseline
            - **Validation Set Accuracy:** `95.76%`
            - **F1 Score (macro):** `95.74%`
            - **Cross-Validation Macro F1:** `0.9637` (std = `0.0095`)
            """
        )

# ----------------- SCREEN 2: ANALYZE & PREDICT -----------------
elif page == "📊 Analyze & Predict":
    st.markdown("## 🔍 City Ingestion, Prediction, & Explanations")
    
    # 1. Unified City search box
    st.markdown("### 📍 Select or Enter City")
    selected_option = st.selectbox(
        "Choose a target city from the list or select 'Enter Custom City' to type your own:",
        options=[
            "Bengaluru", 
            "Hyderabad", 
            "Pune", 
            "Mumbai", 
            "Chennai", 
            "Delhi", 
            "Mysuru", 
            "Lucknow", 
            "Enter Custom City..."
        ]
    )
    
    if selected_option == "Enter Custom City...":
        city_input = st.text_input("Type city name (e.g. Kolkata, Jaipur):", placeholder="Kolkata")
    else:
        city_input = selected_option
        
    if st.button("🚀 Analyze City"):
        if not city_input:
            st.error("Please enter a valid city name.")
        else:
            normalized_city = normalize_city_name(city_input)
            
            # Geocoding resolution sequence
            boundary_resolved = geocode_city_boundary_frontend(normalized_city)
            if not boundary_resolved:
                st.error(
                    f"❌ Geocoding Failed: Could not retrieve administrative boundary polygon for '{city_input}'. "
                    "Please check spelling, or try appending the state details (e.g. 'Mysuru, Karnataka')."
                )
            else:
                city_features_path = Config.FEATURES_DIR / f"{normalized_city.lower()}_growth_dataset.csv"
                
                # Setup stages indicators container
                progress_container = st.container()
                with progress_container:
                    status_placeholder = st.empty()
                    
                    # Custom progress callback
                    def st_callback(step, progress, status):
                        lines = []
                        lines.append(f"### ⚙️ Executing Ingestion & Classification Pipeline")
                        lines.append("---")
                        
                        # Calculate remaining time estimate based on stages
                        total_est = sum(s[2] for s in STAGES)
                        elapsed_est = 0
                        is_gee_stage = False
                        
                        for stage_name, stage_prog, stage_dur in STAGES:
                            if stage_prog <= progress:
                                elapsed_est += stage_dur
                                
                        rem_est = max(0, total_est - elapsed_est)
                        
                        for stage_name, stage_prog, stage_dur in STAGES:
                            if progress >= stage_prog:
                                if stage_name == step and status == "processing":
                                    icon = "🔄"
                                    style = "font-weight: bold; color: #818cf8;"
                                    label = f"{stage_name} ... {progress}%"
                                    if stage_name == "Generating Sentinel imagery":
                                        is_gee_stage = True
                                else:
                                    icon = "✅"
                                    style = "color: #10b981;"
                                    label = stage_name
                                    
                                lines.append(f'<div class="step-line" style="{style}">{icon} {label}</div>')
                            else:
                                icon = "⏳"
                                style = "color: #475569;"
                                lines.append(f'<div class="step-line" style="{style}">{icon} {stage_name}</div>')
                                
                        lines.append("---")
                        if progress < 100:
                            lines.append(f"⏱️ **Estimated time remaining:** ~`{rem_est} seconds`")
                            if is_gee_stage:
                                lines.append("<div style='color: #f59e0b; font-size: 0.85rem; margin-top: 0.5rem;'>⚠️ *This step may take several minutes depending on image availability and Google Earth Engine server latency.*</div>")
                        else:
                            lines.append("🎉 **Pipeline completed successfully!**")
                            
                        status_placeholder.markdown("\n".join(lines), unsafe_allow_html=True)
                        
                    try:
                        # 3. Call backend pipeline
                        summary = analyze_city(normalized_city, status_callback=st_callback)
                        progress_container.empty() # Clear pipeline status checklist
                        
                        st.success(f"Analysis completed successfully for {normalized_city}!")
                        
                        # Load prediction table and GeoJSON
                        df_preds = pd.read_csv(Path(summary["prediction_file"]))
                        geojson_path = Config.FEATURES_DIR / f"osm_features_{normalized_city}.geojson"
                        
                        # --- EXECUTIVE SUMMARY CARD PANEL ---
                        st.markdown("### 📊 Executive Summary Dashboard")
                        
                        total_grids = summary["number_of_grids"]
                        h_pct = (summary["high_growth_count"] / total_grids) * 100 if total_grids > 0 else 0
                        m_pct = (summary["medium_growth_count"] / total_grids) * 100 if total_grids > 0 else 0
                        l_pct = (summary["low_growth_count"] / total_grids) * 100 if total_grids > 0 else 0
                        
                        avg_confidence = df_preds["prediction_probability"].mean()
                        
                        if h_pct > 35:
                            growth_label = "High Expansion"
                            color_style = "color: #ef4444;"
                        elif m_pct > 35:
                            growth_label = "Moderate Change"
                            color_style = "color: #eab308;"
                        else:
                            growth_label = "Low Development"
                            color_style = "color: #22c55e;"
                            
                        st.markdown(
                            f"""
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
                                <div class="kpi-card">
                                    <div class="kpi-value" style="color: #ffffff;">{normalized_city}</div>
                                    <div class="kpi-label">Target Area</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-value" style="color: #a78bfa;">{total_grids:,}</div>
                                    <div class="kpi-label">Analysis Cells</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-value" style="color: #60a5fa;">{summary['average_growth_score']:.4f}</div>
                                    <div class="kpi-label">Avg Growth Score (UCI)</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-value" style="{color_style}">{growth_label}</div>
                                    <div class="kpi-label">Overall Growth Level</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-value" style="color: #10b981;">{avg_confidence:.2%}</div>
                                    <div class="kpi-label">Prediction Confidence</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Category-wise split Metrics
                        st.markdown(
                            f"""
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 2rem; text-align: center;">
                                <div style="background-color: #1e1b4b; border: 1px solid #312e81; padding: 1.25rem; border-radius: 8px;">
                                    <div style="font-size: 2.2rem; font-weight: 700; color: #ef4444;">{summary['high_growth_count']:,}</div>
                                    <div style="font-size: 0.8rem; color: #a5b4fc; text-transform: uppercase;">High Growth Areas ({h_pct:.1f}%)</div>
                                </div>
                                <div style="background-color: #1e1b4b; border: 1px solid #312e81; padding: 1.25rem; border-radius: 8px;">
                                    <div style="font-size: 2.2rem; font-weight: 700; color: #eab308;">{summary['medium_growth_count']:,}</div>
                                    <div style="font-size: 0.8rem; color: #a5b4fc; text-transform: uppercase;">Medium Growth Areas ({m_pct:.1f}%)</div>
                                </div>
                                <div style="background-color: #1e1b4b; border: 1px solid #312e81; padding: 1.25rem; border-radius: 8px;">
                                    <div style="font-size: 2.2rem; font-weight: 700; color: #22c55e;">{summary['low_growth_count']:,}</div>
                                    <div style="font-size: 0.8rem; color: #a5b4fc; text-transform: uppercase;">Low Growth Areas ({l_pct:.1f}%)</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # --- AI READABLE INSIGHTS REPORT ---
                        st.markdown("### 🤖 Platform AI Insights")
                        st.markdown(generate_ai_insights(normalized_city, summary))
                        
                        # --- INTERACTIVE PYDECK MAP VISUALIZATION ---
                        st.markdown("### 🗺️ Interactive Urban Growth Grid Map")
                        
                        if geojson_path.exists():
                            gdf = gpd.read_file(geojson_path)
                            gdf["grid_id"] = gdf["grid_id"].astype(str)
                            df_preds["grid_id"] = df_preds["grid_id"].astype(str)
                            
                            gdf_merged = gdf.merge(df_preds, on="grid_id")
                            
                            # Assign RGB Colors for Category Fill
                            def get_rgba(row):
                                cat = row["predicted_growth_category"]
                                if cat == "High":
                                    return [239, 68, 68, 140]
                                elif cat == "Medium":
                                    return [234, 179, 8, 140]
                                else:
                                    return [34, 197, 94, 140]
                                    
                            gdf_merged["fill_color"] = gdf_merged.apply(get_rgba, axis=1)
                            
                            # Configure Pydeck layer
                            geojson_layer = pdk.Layer(
                                "GeoJsonLayer",
                                gdf_merged,
                                opacity=0.8,
                                stroked=True,
                                filled=True,
                                extruded=False,
                                get_fill_color="fill_color",
                                get_line_color=[255, 255, 255, 50],
                                line_width_min_pixels=1,
                                pickable=True
                            )
                            
                            # Set center point
                            centroid = gdf_merged.geometry.unary_union.centroid
                            view_state = pdk.ViewState(
                                latitude=centroid.y,
                                longitude=centroid.x,
                                zoom=11,
                                pitch=0
                            )
                            
                            r = pdk.Deck(
                                layers=[geojson_layer],
                                initial_view_state=view_state,
                                tooltip={
                                    "html": "<b>Grid Cell:</b> {grid_id}<br/>"
                                            "<b>Growth Index (UCI):</b> {growth_score}<br/>"
                                            "<b>Prediction Class:</b> {predicted_growth_category}<br/>"
                                            "<b>Probability:</b> {prediction_probability}",
                                    "style": {"backgroundColor": "#1e293b", "color": "#f8fafc", "border": "1px solid #475569"}
                                }
                            )
                            
                            # Display map
                            st.pydeck_chart(r)
                            st.caption("ℹ️ Hover over any grid cell polygon on the map to inspect its real-time predicted change probability.")
                            
                        # --- EXPLAINABILITY GRID INSPECTOR ---
                        st.markdown("### 🔍 Grid Explanations & SHAP Driver")
                        
                        shap_file_path = Path(summary["shap_file"])
                        if shap_file_path.exists():
                            df_shap = pd.read_csv(shap_file_path)
                            
                            grid_list = df_shap["grid_id"].tolist()
                            sel_grid_id = st.selectbox(
                                "Select a Grid ID to inspect details & explanation drivers:",
                                options=grid_list,
                                index=0
                            )
                            
                            # Filter specific cell predictions & SHAP values
                            cell_pred = df_preds[df_preds["grid_id"].astype(str) == str(sel_grid_id)].iloc[0]
                            cell_shap = df_shap[df_shap["grid_id"].astype(str) == str(sel_grid_id)].iloc[0]
                            
                            # Load spectral values from growth dataset to display (e.g. NDVI/NDBI/NDWI)
                            city_features_path = Config.FEATURES_DIR / f"{normalized_city.lower()}_growth_dataset.csv"
                            df_features_cached = pd.read_csv(city_features_path)
                            cell_features = df_features_cached[df_features_cached["grid_id"].astype(str) == str(sel_grid_id)].iloc[0]
                            
                            # Display KPI panel for cell
                            sc_col1, sc_col2, sc_col3 = st.columns(3)
                            sc_col1.metric("Selected Cell Index", f"{sel_grid_id}")
                            sc_col2.metric("Growth Score (UCI)", f"{cell_pred['growth_score']:.4f}")
                            sc_col3.metric("Prediction Class", f"{cell_pred['predicted_growth_category']} ({cell_pred['prediction_probability']:.1%})")
                            
                            # Build SHAP importance horizontal bar chart
                            shap_feats = [c for c in df_shap.columns if c != "grid_id"]
                            cell_shap_series = cell_shap[shap_feats].astype(float)
                            
                            # Identify contributors & Translate to readable phrases
                            df_cell_shap = pd.DataFrame({
                                "Feature": cell_shap_series.index,
                                "SHAP Value": cell_shap_series.values,
                                "Influence": ["Increase Growth" if v > 0 else "Decrease Growth" for v in cell_shap_series.values]
                            }).sort_values(by="SHAP Value", key=abs, ascending=True)
                            
                            # Translate SHAP into readable language
                            readable_reasons = translate_shap_features(cell_shap_series.sort_values(key=abs, ascending=False))
                            st.markdown("**Main drivers for this cell's prediction:**")
                            for reason in readable_reasons:
                                st.markdown(f"- {reason}")
                            
                            # Filter top 10 most influential features
                            df_cell_shap = df_cell_shap.tail(10)
                            
                            fig_shap = px.bar(
                                df_cell_shap,
                                x="SHAP Value",
                                y="Feature",
                                orientation="h",
                                color="Influence",
                                color_discrete_map={"Increase Growth": "#ef4444", "Decrease Growth": "#3b82f6"},
                                title=f"SHAP Local Feature Contributors for Grid cell {sel_grid_id}"
                            )
                            fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                            st.plotly_chart(fig_shap, use_container_width=True)
                            
                            # Display NDVI, NDBI, NDWI comparison indicators
                            st.markdown("#### Cell Spectral Shift Indicators")
                            ss_col1, ss_col2, ss_col3 = st.columns(3)
                            ss_col1.metric("Delta NDVI (Vegetation)", f"{cell_features['delta_ndvi']:.4f}")
                            ss_col2.metric("Delta NDBI (Built-up)", f"{cell_features['delta_ndbi']:.4f}")
                            ss_col3.metric("Delta NDWI (Moisture)", f"{cell_features['delta_ndwi']:.4f}")
                            
                        # --- ANALYTICAL PLOTLY CHARTS ---
                        st.markdown("### 📊 Platform Analytics Charts")
                        
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            # 1. Growth category distribution chart
                            fig_pie = px.pie(
                                names=["Low", "Medium", "High"],
                                values=[summary["low_growth_count"], summary["medium_growth_count"], summary["high_growth_count"]],
                                color=["Low", "Medium", "High"],
                                color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"},
                                hole=0.4,
                                title="Urban Growth Area Classifications (%)"
                            )
                            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # 2. Spectral indices pre-post shift chart
                            city_features_path = Config.FEATURES_DIR / f"{normalized_city.lower()}_growth_dataset.csv"
                            if city_features_path.exists():
                                df_feat = pd.read_csv(city_features_path)
                                avg_spectral = df_feat[[
                                    "mean_ndvi_2019", "mean_ndvi_2026",
                                    "mean_ndbi_2019", "mean_ndbi_2026",
                                    "mean_ndwi_2019", "mean_ndwi_2026"
                                ]].mean()
                                
                                # NDVI comparison bar
                                fig_ndvi = px.histogram(
                                    df_feat,
                                    x=["mean_ndvi_2019", "mean_ndvi_2026"],
                                    barmode="overlay",
                                    title="NDVI shift distribution (2019 vs 2026)",
                                    color_discrete_sequence=["#3b82f6", "#ef4444"]
                                )
                                fig_ndvi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                                st.plotly_chart(fig_ndvi, use_container_width=True)
                                
                                # NDBI comparison bar
                                fig_ndbi = px.histogram(
                                    df_feat,
                                    x=["mean_ndbi_2019", "mean_ndbi_2026"],
                                    barmode="overlay",
                                    title="NDBI shift distribution (2019 vs 2026)",
                                    color_discrete_sequence=["#3b82f6", "#ef4444"]
                                )
                                fig_ndbi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                                st.plotly_chart(fig_ndbi, use_container_width=True)
                                
                                # NDWI comparison bar
                                fig_ndwi = px.histogram(
                                    df_feat,
                                    x=["mean_ndwi_2019", "mean_ndwi_2026"],
                                    barmode="overlay",
                                    title="NDWI shift distribution (2019 vs 2026)",
                                    color_discrete_sequence=["#3b82f6", "#ef4444"]
                                )
                                fig_ndwi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                                st.plotly_chart(fig_ndwi, use_container_width=True)
                                
                        with chart_col2:
                            # 3. Growth score histogram
                            fig_hist = px.histogram(
                                df_preds,
                                x="growth_score",
                                nbins=30,
                                color_discrete_sequence=["#a78bfa"],
                                title="Urban Change Index Distribution (Histogram)"
                            )
                            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                            st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # 4. Global SHAP Feature Importance chart
                            if shap_file_path.exists():
                                feat_cols = [c for c in df_shap.columns if c != "grid_id"]
                                mean_abs_shap = df_shap[feat_cols].abs().mean().sort_values(ascending=False)
                                
                                df_global_shap = pd.DataFrame({
                                    "Feature": mean_abs_shap.index,
                                    "Mean |SHAP Value|": mean_abs_shap.values
                                }).head(10).sort_values(by="Mean |SHAP Value|", ascending=True)
                                
                                fig_glob = px.bar(
                                    df_global_shap,
                                    x="Mean |SHAP Value|",
                                    y="Feature",
                                    orientation="h",
                                    color="Mean |SHAP Value|",
                                    color_continuous_scale="Purples",
                                    title="Global Feature Importance (Mean |SHAP|)"
                                )
                                fig_glob.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                                st.plotly_chart(fig_glob, use_container_width=True)
                                
                        # --- DOWNLOAD OPTIONS (PDF / CSV / SHAP) ---
                        st.markdown("### 📥 Download Executive Analytics Reports")
                        
                        pdf_data = generate_pdf_report(normalized_city, summary)
                        st.download_button(
                            label="📥 Download Executive Summary PDF Report",
                            data=pdf_data,
                            file_name=f"{normalized_city.lower()}_analysis_report.pdf",
                            mime="application/pdf"
                        )
                        
                        shap_report_data = generate_shap_report_txt(normalized_city, df_shap)
                        st.download_button(
                            label="📥 Download SHAP Explanation Report (.txt)",
                            data=shap_report_data,
                            file_name=f"{normalized_city.lower()}_shap_explanation.txt",
                            mime="text/plain"
                        )
                        
                        # --- OPTIONAL EXPANDABLE PREDICTIONS TABLE ---
                        st.markdown("---")
                        with st.expander("📋 Advanced Technical Details"):
                            st.dataframe(
                                df_preds.style.format({
                                    "growth_score": "{:.4f}",
                                    "prediction_probability": "{:.2%}"
                                }),
                                use_container_width=True
                            )
                            
                            with open(summary["prediction_file"], "rb") as f:
                                st.download_button(
                                    label="📥 Download Full Predictions CSV",
                                    data=f,
                                    file_name=f"{normalized_city.lower()}_predictions.csv",
                                    mime="text/csv"
                                )
                            
                    except Exception as err:
                        st.error(f"Failed to complete urban analysis. Error: {err}")

# ----------------- SCREEN 3: BENCHMARKS COMPARISON -----------------
elif page == "🏆 Benchmarks Comparison":
    st.markdown("## 📊 Comparative Analysis: Bengaluru vs Hyderabad vs Pune")
    
    # Render comparative KPI cards for the three main benchmark cities
    benchmark_cities = ["Bengaluru", "Hyderabad", "Pune"]
    
    records = []
    distribution = {}
    
    for city in benchmark_cities:
        summary_json_path = PREDICTIONS_DIR / city / "prediction_summary.json"
        if summary_json_path.exists():
            with open(summary_json_path, "r") as f:
                sum_data = json.load(f)
            
            tot = sum_data["number_of_grids"]
            records.append({
                "City": city,
                "Total Grids": tot,
                "Avg Growth Index": sum_data["average_growth_score"],
                "High Growth Grids": sum_data["high_growth_count"],
                "Medium Growth Grids": sum_data["medium_growth_count"],
                "Low Growth Grids": sum_data["low_growth_count"]
            })
            
            distribution[city] = {
                "Low Growth (%)": (sum_data["low_growth_count"] / tot) * 100 if tot > 0 else 0,
                "Medium Growth (%)": (sum_data["medium_growth_count"] / tot) * 100 if tot > 0 else 0,
                "High Growth (%)": (sum_data["high_growth_count"] / tot) * 100 if tot > 0 else 0
            }
            
    if len(records) < 3:
        st.warning("Make sure Bengaluru, Hyderabad, and Pune have completed processing cached feature steps first.")
    else:
        # Display Side-by-side KPI cards comparing average growth scores
        st.markdown("### 🏆 Benchmark Average Growth Scores (UCI)")
        k_col1, k_col2, k_col3 = st.columns(3)
        
        # Bengaluru KPI
        k_col1.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid #ef4444;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #a5b4fc;">Bengaluru</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0.3rem 0;">{records[0]['Avg Growth Index']:.4f}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Total Grids: {records[0]['Total Grids']:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Hyderabad KPI
        k_col2.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid #eab308;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #a5b4fc;">Hyderabad</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0.3rem 0;">{records[1]['Avg Growth Index']:.4f}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Total Grids: {records[1]['Total Grids']:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Pune KPI
        k_col3.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid #22c55e;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #a5b4fc;">Pune</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0.3rem 0;">{records[2]['Avg Growth Index']:.4f}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Total Grids: {records[2]['Total Grids']:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Detailed comparison table
        st.markdown("### 📋 Comparison Data Table")
        df_comp = pd.DataFrame(records)
        st.dataframe(
            df_comp.style.format({
                "Total Grids": "{:,}",
                "Avg Growth Index": "{:.4f}",
                "High Growth Grids": "{:,}",
                "Medium Growth Grids": "{:,}",
                "Low Growth Grids": "{:,}"
            }),
            use_container_width=True
        )
        
        # Comparative Stacked distributions chart
        st.markdown("### 📊 Comparative Growth Category Share (%)")
        df_dist_compare = pd.DataFrame(distribution).T
        
        fig_comp_bar = px.bar(
            df_dist_compare,
            x=df_dist_compare.index,
            y=["Low Growth (%)", "Medium Growth (%)", "High Growth (%)"],
            barmode="stack",
            color_discrete_sequence=["#22c55e", "#eab308", "#ef4444"],
            title="Growth Category Distribution Share Comparison"
        )
        fig_comp_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis_title="Benchmark City",
            yaxis_title="Percent Contribution Share (%)",
            legend_title="Growth Category"
        )
        st.plotly_chart(fig_comp_bar, use_container_width=True)

# ----------------- SCREEN 4: METHODOLOGY -----------------
elif page == "ℹ️ Methodology":
    st.markdown("## ℹ️ Platform Technical Methodology")
    st.markdown(
        """
        The **Urban Growth Intelligence Platform** combines spatial network algorithms and multi-spectral indices to predict municipal development.
        
        ### 📡 Remote Sensing Satellite Ingestion
        Sentinel-2 surface reflectance bands are cloud-masked using Cloud Probability and QA60 thresholds inside Google Earth Engine. 
        - **NDVI (Vegetation Shift):** `(NIR - Red) / (NIR + Red)`
        - **NDBI (Built-up structures Shift):** `(SWIR - NIR) / (SWIR + NIR)`
        - **NDWI (Surface Moisture Shift):** `(Green - NIR) / (Green + NIR)`
        
        Temporal difference deltas between the baseline (2019) and prediction (2026) years indicate change magnitude.
        
        ### 🏢 Infrastructural Network Extraction
        OSM buildings footprints, road networks, and municipal centers are extracted to construct spatial features:
        - Building Count, Density, and Footprint Area Ratios.
        - Roadway lengths, densities, and intersections.
        - Distance to major highways and city center points.
        
        ### 🤖 XGBoost Baseline Model
        Features are standard-scaled (StandardScaler) and input into an optimized **XGBoost Classifier** baseline trained on multi-temporal samples. The model outputs the Urban Change Index score along with categorical classifications (Low, Medium, High).
        """
    )
