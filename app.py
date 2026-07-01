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
    page_title="Urban Growth Intelligence Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS injection to create a high-quality dark theme dashboard interface
st.markdown("""
<style>
    /* Dark Theme General Styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Sidebar Navigation styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Header card container styling */
    .header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #312e81;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-card h1 {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .header-card p {
        color: #818cf8;
        font-size: 1.15rem;
        font-weight: 400;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Detail Box styling */
    .detail-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define directories
PROJECT_ROOT = Config.PROJECT_ROOT
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize cached predictions silently to ensure Comparison works immediately
for city in ["Bengaluru", "Hyderabad", "Pune"]:
    summary_json_path = PREDICTIONS_DIR / city / "prediction_summary.json"
    if not summary_json_path.exists():
        try:
            analyze_city(city)
        except Exception as e:
            logger.warning(f"Could not pre-cache {city}: {e}")

# Pipeline Stages definitions for checking list boxes
STAGES = [
    ("Downloading administrative boundary", 10),
    ("Downloading OSM data", 20),
    ("Cleaning OSM data", 30),
    ("Generating spatial grid", 40),
    ("Joining spatial data", 50),
    ("Extracting OSM features", 60),
    ("Generating Sentinel imagery", 70),
    ("Extracting raster features", 80),
    ("Generating ML feature dataset", 85),
    ("Loading trained production model", 92),
    ("Generating predictions", 95),
    ("Generating SHAP explanations", 98),
    ("Finalizing results", 99)
]

def geocode_city_boundary_frontend(city_name: str) -> bool:
    """Frontend-level geocoding resolution logic supporting State and Country suffixes."""
    boundary_path = Config.BOUNDARIES_DIR / f"{city_name}.geojson"
    if boundary_path.exists():
        return True
        
    # Search query sequence variants
    queries = []
    # If comma specified, assume user entered state (e.g. Mysuru, Karnataka)
    if "," in city_name:
        parts = [p.strip() for p in city_name.split(",")]
        if len(parts) >= 2:
            queries.append(f"{parts[0]}, {parts[1]}, India")
            queries.append(f"{parts[0]}, India")
    else:
        queries.append(f"{city_name}, India")
        queries.append(city_name)
        
    for q in queries:
        try:
            logger.info(f"[Frontend Geocoder] Trying query: {q}")
            gdf = ox.geocode_to_gdf(q)
            if not gdf.empty:
                boundary_path.parent.mkdir(parents=True, exist_ok=True)
                gdf.to_file(boundary_path, driver="GeoJSON")
                logger.success(f"[Frontend Geocoder] Successfully saved boundary for: {city_name}")
                return True
        except Exception as e:
            logger.debug(f"Failed query variant {q}: {e}")
            continue
    return False

def generate_pdf_report(city_name: str, summary: dict) -> bytes:
    """Generates an in-memory PDF summary report using ReportLab."""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setTitle(f"Urban Growth Intelligence Report - {city_name}")
    
    # Header Banner
    p.setFillColorRGB(0.09, 0.09, 0.2) # Dark background
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

# Sidebar controls layout
st.sidebar.markdown("## 🏙️ Platform Navigation")
page = st.sidebar.radio(
    "Select Interface Screen:",
    ["Home", "Analyze City Dashboard", "Compare Cities", "Methodology"]
)

# ----------------- SCREEN 1: HOME -----------------
if page == "Home":
    st.markdown("## 🏠 Platform Overview")
    st.markdown(
        "The **Urban Growth Intelligence Platform** is an enterprise-grade spatial machine learning platform. "
        "It enables developers, urban planners, and researchers to model, predict, and explain boundary expansions "
        "using high-resolution Sentinel-2 Earth observation imagery and localized OpenStreetMap (OSM) infrastructure networks."
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚙️ End-to-End Processing Workflow")
        st.markdown(
            """
            The backend engine automatically executes a modular pipeline to query, clean, and classify spatial cells:
            
            - **Administrative Boundary Geocoding:** Geocodes and parses municipal borders.
            - **OSM Spatial Data Ingestion:** Downloads infrastructural features (buildings, road lines, intersections).
            - **Sentinel-2 Multi-Spectral Query:** Queries masked surface reflectance composites via GEE.
            - **Spectral Indices Calculations:** Extracts grid cell NDVI, NDBI, and NDWI indices.
            - **Zonal Statistics Integration:** Integrates spatial densities and spectral shift deltas.
            - **Production Classification:** Scores growth score probability via final optimized XGBoost trees.
            - **SHAP Explanation Logging:** Evaluates local feature contributors behind individual cell predictions.
            """
        )
    with col2:
        st.markdown("### 🛠️ Production Baseline Metrics")
        st.markdown(
            "The platform leverages an optimized **XGBoost Classifier** baseline trained on multi-temporal datasets:"
        )
        
        # Display Metrics list
        st.markdown(
            """
            - **Classification Model:** XGBoost Baseline
            - **Test Set Accuracy:** `95.76%`
            - **Macro F1 Score:** `95.74%`
            - **Out-of-sample Cross-Validation F1:** `96.37%`
            - **Generalization Standard Deviation:** `0.95%`
            """
        )
        
        st.markdown("### 📦 Key Ingested Packages")
        packages = ["Python 3.10", "XGBoost", "Streamlit", "ReportLab", "OSMnx", "GeoPandas", "Shapely", "Rasterio", "Plotly", "SHAP", "PyTorch"]
        badge_html = "".join([f'<span class="tech-badge">{p}</span>' for p in packages])
        st.markdown(badge_html, unsafe_allow_html=True)

# ----------------- SCREEN 2: ANALYZE CITY DASHBOARD -----------------
elif page == "Analyze City Dashboard":
    st.markdown("## 🔍 City Growth Prediction & Analytics")
    
    # Input Selection UI
    analysis_type = st.radio(
        "Select target analysis type:",
        ["Option 1: Pre-processed City Selection", "Option 2: Analyze a Custom City"]
    )
    
    city_name = ""
    if analysis_type == "Option 1: Pre-processed City Selection":
        city_name = st.selectbox("Select Target City:", ["Bengaluru", "Hyderabad", "Pune"])
    else:
        city_name = st.text_input("Enter city name (e.g. Mysuru, Chennai, Delhi):", placeholder="Mysuru")
        
    if st.button("🚀 Ingest & Analyze"):
        if not city_name:
            st.error("Please enter a valid city name.")
        else:
            # 1. Resolve boundary polygon geocoding variants
            boundary_resolved = geocode_city_boundary_frontend(city_name)
            if not boundary_resolved:
                st.error(
                    f"❌ Geocoding Failed: Could not retrieve administrative boundary polygon for '{city_name}'. "
                    "Please check spelling, or try appending the state details (e.g. 'Mysuru, Karnataka')."
                )
            else:
                # 2. Setup stage-by-stage status check indicators
                progress_container = st.container()
                with progress_container:
                    status_placeholder = st.empty()
                    
                    def st_callback(step, progress, status):
                        lines = []
                        lines.append(f"### ⚙️ Executing Ingestion & Classification Pipeline")
                        lines.append("---")
                        
                        for stage_name, stage_prog in STAGES:
                            if progress >= stage_prog:
                                if stage_name == step and status == "processing":
                                    icon = "🔄"
                                    style = "font-weight: bold; color: #818cf8;"
                                    label = f"{stage_name} ... {progress}%"
                                else:
                                    icon = "✅"
                                    style = "color: #10b981;"
                                    label = stage_name
                                    
                                lines.append(f'<div style="margin-bottom: 0.4rem; {style}">{icon} {label}</div>')
                            else:
                                icon = "⏳"
                                style = "color: #475569;"
                                lines.append(f'<div style="margin-bottom: 0.4rem; {style}">{icon} {stage_name}</div>')
                                
                        status_placeholder.markdown("\n".join(lines), unsafe_allow_html=True)
                        
                    try:
                        # 3. Call backend pipeline
                        summary = analyze_city(city_name, status_callback=st_callback)
                        progress_container.empty() # Clear pipeline logs
                        
                        st.success(f"Analysis completed successfully for {city_name}!")
                        
                        # --- EXECUTIVE SUMMARY METRICS ---
                        st.markdown("### 📊 Executive Summary Dashboard")
                        
                        # Compute overall category classification percentages
                        total_grids = summary["number_of_grids"]
                        h_pct = (summary["high_growth_count"] / total_grids) * 100
                        m_pct = (summary["medium_growth_count"] / total_grids) * 100
                        l_pct = (summary["low_growth_count"] / total_grids) * 100
                        
                        # Overall Category label determination
                        if h_pct > 35:
                            growth_label = "High Growth Expansion"
                            color_style = "color: #ef4444;"
                        elif m_pct > 35:
                            growth_label = "Moderate Change"
                            color_style = "color: #eab308;"
                        else:
                            growth_label = "Low Development Shift"
                            color_style = "color: #22c55e;"
                            
                        # Layout Metric Cards
                        st.markdown(
                            f"""
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
                                <div class="kpi-card">
                                    <div class="kpi-value" style="color: #ffffff;">{city_name}</div>
                                    <div class="kpi-label">Target Area</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-value" style="color: #a78bfa;">{total_grids:,}</div>
                                    <div class="kpi-label">Analysis Cells</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-value" style="color: #ff4b4b;">{summary['average_growth_score']:.4f}</div>
                                    <div class="kpi-label">Avg Growth Score (UCI)</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-value" style="{color_style}">{growth_label}</div>
                                    <div class="kpi-label">Overall Category</div>
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
                        
                        # --- INTERACTIVE PYDECK MAP VISUALIZATION ---
                        st.markdown("### 🗺️ Interactive Urban Growth Grid Map")
                        
                        # Load prediction table and GeoJSON
                        df_preds = pd.read_csv(Path(summary["prediction_file"]))
                        geojson_path = Config.FEATURES_DIR / f"osm_features_{city_name}.geojson"
                        
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
                            
                            # Display KPI panel for cell
                            sc_col1, sc_col2, sc_col3 = st.columns(3)
                            sc_col1.metric("Selected Cell Index", f"{sel_grid_id}")
                            sc_col2.metric("Growth Score (UCI)", f"{cell_pred['growth_score']:.4f}")
                            sc_col3.metric("Prediction Class", f"{cell_pred['predicted_growth_category']} ({cell_pred['prediction_probability']:.1%})")
                            
                            # Build SHAP importance horizontal bar chart
                            shap_feats = [c for c in df_shap.columns if c != "grid_id"]
                            cell_shap_series = cell_shap[shap_feats].astype(float)
                            
                            # Identify contributors
                            df_cell_shap = pd.DataFrame({
                                "Feature": cell_shap_series.index,
                                "SHAP Value": cell_shap_series.values,
                                "Influence": ["Increase Growth" if v > 0 else "Decrease Growth" for v in cell_shap_series.values]
                            }).sort_values(by="SHAP Value", key=abs, ascending=True)
                            
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
                            city_features_path = Config.FEATURES_DIR / f"{city_name.lower()}_growth_dataset.csv"
                            if city_features_path.exists():
                                df_feat = pd.read_csv(city_features_path)
                                avg_spectral = df_feat[[
                                    "mean_ndvi_2019", "mean_ndvi_2026",
                                    "mean_ndbi_2019", "mean_ndbi_2026",
                                    "mean_ndwi_2019", "mean_ndwi_2026"
                                ]].mean()
                                
                                df_spectral_compare = pd.DataFrame([
                                    {"Index": "NDVI", "Year": "2019", "Mean Value": avg_spectral["mean_ndvi_2019"]},
                                    {"Index": "NDVI", "Year": "2026", "Mean Value": avg_spectral["mean_ndvi_2026"]},
                                    {"Index": "NDBI", "Year": "2019", "Mean Value": avg_spectral["mean_ndbi_2019"]},
                                    {"Index": "NDBI", "Year": "2026", "Mean Value": avg_spectral["mean_ndbi_2026"]},
                                    {"Index": "NDWI", "Year": "2019", "Mean Value": avg_spectral["mean_ndwi_2019"]},
                                    {"Index": "NDWI", "Year": "2026", "Mean Value": avg_spectral["mean_ndwi_2026"]}
                                ])
                                
                                fig_spec = px.bar(
                                    df_spectral_compare,
                                    x="Index",
                                    y="Mean Value",
                                    color="Year",
                                    barmode="group",
                                    color_discrete_map={"2019": "#3b82f6", "2026": "#f43f5e"},
                                    title="Pre-Post Spectral Indices Shifts Comparison"
                                )
                                fig_spec.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                                st.plotly_chart(fig_spec, use_container_width=True)
                                
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
                                
                        # --- DOWNLOAD OPTIONS (PDF / CSV) ---
                        st.markdown("### 📥 Download Executive Analytics Reports")
                        
                        pdf_data = generate_pdf_report(city_name, summary)
                        st.download_button(
                            label="📥 Download Executive Summary PDF Report",
                            data=pdf_data,
                            file_name=f"{city_name.lower()}_analysis_report.pdf",
                            mime="application/pdf"
                        )
                        
                        # --- OPTIONAL EXPANDABLE PREDICTIONS TABLE ---
                        st.markdown("---")
                        with st.expander("📋 View Complete Predictions Data Table"):
                            st.dataframe(
                                df_preds.style.format({
                                    "growth_score": "{:.4f}",
                                    "prediction_probability": "{:.2%}"
                                }),
                                use_container_width=True
                            )
                            
                    except Exception as err:
                        st.error(f"Failed to complete urban analysis. Error: {err}")

# ----------------- SCREEN 3: COMPARE CITIES -----------------
elif page == "Compare Cities":
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
elif page == "Methodology":
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
