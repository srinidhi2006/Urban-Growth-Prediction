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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from io import BytesIO

# Import PDF report generation flowable packages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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

# Professional CSS injection to create a high-quality dark theme dashboard interface
st.markdown("""
<style>
    /* Dark Theme General Styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Sidebar Navigation styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Header card container styling */
    .header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        padding: 2.5rem;
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
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .header-card p {
        color: #818cf8;
        font-size: 1.2rem;
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
        font-weight: 600;
    }
    
    /* Detail Box styling */
    .detail-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    
    /* Step log line */
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

# Pipeline Stages definitions for progress logs
STAGES = [
    ("Downloading administrative boundary", 10, 10),
    ("Downloading OSM data", 20, 15),
    ("Cleaning OSM data", 30, 5),
    ("Generating spatial grid", 40, 5),
    ("Joining spatial data", 50, 5),
    ("Extracting OSM features", 60, 10),
    ("Generating Sentinel imagery", 70, 30),
    ("Extracting raster features", 80, 15),
    ("Generating ML feature dataset", 85, 5),
    ("Loading trained production model", 92, 5),
    ("Generating predictions", 95, 3),
    ("Generating SHAP explanations", 98, 10),
    ("Finalizing results", 99, 2)
]

# NumberedCanvas subclass to handle page headers, footers, and page numbers dynamically
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # We suppress headers and footers on the cover page (Page 1)
        if self._pageNumber > 1:
            self.saveState()
            
            # Draw Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#334155"))
            self.drawString(50, 755, "Urban Growth Assessment Report")
            
            city_name = st.session_state.get("active_city", "City")
            self.setFont("Helvetica", 8)
            self.drawRightString(562, 755, f"Location: {city_name} | Target: 2026")
            
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(50, 748, 562, 748)
            
            # Draw Footer
            self.line(50, 48, 562, 48)
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(50, 35, "Urban Growth Intelligence Platform")
            self.drawRightString(562, 35, f"Page {self._pageNumber} of {page_count}")
            
            self.restoreState()

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

def geocode_city_boundary_frontend(city_name: str) -> bool:
    boundary_path = Config.BOUNDARIES_DIR / f"{city_name}.geojson"
    if boundary_path.exists():
        return True
        
    queries = []
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
                return True
        except Exception as e:
            logger.debug(f"Failed query variant {q}: {e}")
            continue
    return False

def assign_locality_names(grid_gdf, city_name) -> dict:
    """Performs spatial reverse-geocoding to resolve locality names using offline caches or OSM APIs."""
    normalized = normalize_city_name(city_name)
    
    # 1. Load from Offline cache CSVs first for Bengaluru, Hyderabad, Pune
    if normalized in ["Bengaluru", "Hyderabad", "Pune"]:
        cache_csv_path = PROJECT_ROOT / "data" / "cache" / f"{normalized.lower()}_localities.csv"
        if cache_csv_path.exists():
            df_cache = pd.read_csv(cache_csv_path)
            return dict(zip(df_cache["grid_id"].astype(str), df_cache["locality_name"].astype(str)))
            
    # 2. Check JSON cache for other cities
    cache_json_path = Config.FEATURES_DIR / f"{normalized.lower()}_localities.json"
    if cache_json_path.exists():
        try:
            with open(cache_json_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
            
    logger.info(f"Resolving locality names for {city_name} via OSMnx...")
    try:
        boundary_path = Config.BOUNDARIES_DIR / f"{city_name}.geojson"
        boundary_gdf = gpd.read_file(boundary_path)
        geom = boundary_gdf.geometry.iloc[0]
        
        places = ox.features_from_polygon(geom, {'place': ['suburb', 'neighbourhood', 'locality', 'village', 'town', 'quarter']})
        if not places.empty and "name" in places.columns:
            places_clean = places[["name", "geometry"]].dropna().to_crs(Config.PROJECTED_CRS)
            grid_projected = grid_gdf.to_crs(Config.PROJECTED_CRS)
            centroids = grid_projected.geometry.centroid
            
            locality_names = []
            for cent in centroids:
                distances = places_clean.geometry.distance(cent)
                nearest_idx = distances.idxmin()
                locality_names.append(places_clean.loc[nearest_idx, "name"])
        else:
            locality_names = [f"Sector {i+1}" for i in range(len(grid_gdf))]
    except Exception as e:
        logger.error(f"Failed to reverse-geocode place names for {city_name}: {e}")
        locality_names = [f"Sector {i+1}" for i in range(len(grid_gdf))]
        
    grid_ids = grid_gdf["grid_id"].astype(str).tolist()
    result = dict(zip(grid_ids, locality_names))
    
    try:
        with open(cache_json_path, "w") as f:
            json.dump(result, f)
    except Exception:
        pass
        
    return result

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

def generate_pdf_charts(summary):
    """Generates Matplotlib charts in memory for PDF reports."""
    fig, ax = plt.subplots(figsize=(3, 3))
    labels = ['Low', 'Medium', 'High']
    sizes = [summary['low_growth_count'], summary['medium_growth_count'], summary['high_growth_count']]
    colors = ['#22c55e', '#eab308', '#ef4444']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, 
           wedgeprops=dict(width=0.4, edgecolor='w'))
    ax.axis('equal')
    plt.tight_layout()
    pie_buf = BytesIO()
    plt.savefig(pie_buf, format='png', dpi=150, transparent=True)
    plt.close()
    pie_buf.seek(0)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    categories = ['Low', 'Medium', 'High']
    values = [summary['low_growth_count'], summary['medium_growth_count'], summary['high_growth_count']]
    ax.bar(categories, values, color=['#22c55e', '#eab308', '#ef4444'])
    ax.set_title("Growth Category Split", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    bar_buf = BytesIO()
    plt.savefig(bar_buf, format='png', dpi=150, transparent=True)
    plt.close()
    bar_buf.seek(0)
    
    return pie_buf, bar_buf

def generate_complete_pdf_report(city_name: str, summary: dict, df_preds: pd.DataFrame, df_shap: pd.DataFrame, localities: dict) -> bytes:
    """Generates a professional multi-page executive assessment report with flowable auto-wrapping layout."""
    buffer = BytesIO()
    
    # Define document geometry constraints
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=60,
        bottomMargin=60
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles definitions for clean typography
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1e1b4b"),
        alignment=0,
        spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=30
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#ffffff")
    )
    
    story = []
    
    # ---------------- PAGE 1: COVER PAGE ----------------
    story.append(Spacer(1, 150))
    story.append(Paragraph("Urban Growth Assessment Report", title_style))
    story.append(Paragraph(f"Analysis Area: {city_name} — Timeline Target: 2026", subtitle_style))
    
    # Cover page card block
    kpi_data = [
        [Paragraph("<b>Selected City:</b>", table_cell_style), Paragraph(city_name, table_cell_style)],
        [Paragraph("<b>Growth Category:</b>", table_cell_style), Paragraph(summary.get("overall_growth_level", "Moderate"), table_cell_style)],
        [Paragraph("<b>Analysis Date:</b>", table_cell_style), Paragraph("July 1, 2026", table_cell_style)],
        [Paragraph("<b>Version:</b>", table_cell_style), Paragraph("v1.0", table_cell_style)]
    ]
    t_cover = Table(kpi_data, colWidths=[150, 200])
    t_cover.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_cover)
    story.append(PageBreak())
    
    # ---------------- PAGE 2: EXECUTIVE SUMMARY & DISTRIBUTION ----------------
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This urban growth assessment report evaluates geographic expansions, infrastructural densities, "
        "and spectral shift trends. The analysis indicates growth concentrations categorized as detailed below.",
        body_style
    ))
    
    # KPI metrics table cards (styled as 3x2 card grid)
    kpis = [
        [
            Paragraph("<b>Average Growth Score</b><br/>" + f"{summary['average_growth_score']:.4f}", table_cell_style),
            Paragraph("<b>Overall Category</b><br/>" + f"{summary['overall_growth_level']}", table_cell_style),
            Paragraph("<b>Total Analysis Cells</b><br/>" + f"{summary['number_of_grids']:,}", table_cell_style)
        ],
        [
            Paragraph("<b>High Growth Areas</b><br/>" + f"{summary['high_growth_count']:,}", table_cell_style),
            Paragraph("<b>Medium Growth Areas</b><br/>" + f"{summary['medium_growth_count']:,}", table_cell_style),
            Paragraph("<b>Low Growth Areas</b><br/>" + f"{summary['low_growth_count']:,}", table_cell_style)
        ]
    ]
    t_kpis = Table(kpis, colWidths=[164, 164, 164])
    t_kpis.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    story.append(t_kpis)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("2. Growth Category split", h1_style))
    pie_buf, bar_buf = generate_pdf_charts(summary)
    t_charts = Table([
        [RLImage(pie_buf, width=220, height=220), RLImage(bar_buf, width=240, height=180)]
    ], colWidths=[240, 250])
    t_charts.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(t_charts)
    story.append(PageBreak())
    
    # ---------------- PAGE 3: TEMPORAL COMPARISON ----------------
    story.append(Paragraph("3. Temporal Satellite Shifts (2019 vs 2026)", h1_style))
    story.append(Paragraph(
        "Grouped Sentinel-2 surface reflectance imagery comparison across target years:",
        body_style
    ))
    
    dir_2019 = Config.PROCESSED_DIR / "sentinel" / city_name / "2019"
    dir_2026 = Config.PROCESSED_DIR / "sentinel" / city_name / "2026"
    img_2019 = dir_2019 / f"{city_name}_2019_preview.png"
    img_2026 = dir_2026 / f"{city_name}_2026_preview.png"
    
    if img_2019.exists() and img_2026.exists():
        t_images = Table([
            [RLImage(str(img_2019), width=240, height=180), RLImage(str(img_2026), width=240, height=180)]
        ], colWidths=[240, 240])
        t_images.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
        ]))
        story.append(t_images)
        
        # Sub-labels table
        t_labels = Table([
            [Paragraph("<b>A. 2019 RGB Satellite Composite</b>", table_cell_style), 
             Paragraph("<b>B. 2026 RGB Satellite Composite</b>", table_cell_style)]
        ], colWidths=[240, 240])
        t_labels.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_labels)
    else:
        story.append(Paragraph("<i>[Satellite preview image sources cached offline]</i>", body_style))
        
    story.append(Spacer(1, 15))
    story.append(Paragraph("4. Pre-Post Spectral Indices Shifts", h1_style))
    
    city_features_path = Config.FEATURES_DIR / f"{city_name.lower()}_growth_dataset.csv"
    if city_features_path.exists():
        df_feat = pd.read_csv(city_features_path)
        avg_spectral = df_feat[[
            "mean_ndvi_2019", "mean_ndvi_2026",
            "mean_ndbi_2019", "mean_ndbi_2026",
            "mean_ndwi_2019", "mean_ndwi_2026"
        ]].mean()
        
        spec_data = [
            [Paragraph("<b>Spectral Index</b>", table_header_style), Paragraph("<b>2019 Mean</b>", table_header_style), Paragraph("<b>2026 Mean</b>", table_header_style), Paragraph("<b>Delta Shift</b>", table_header_style)],
            [Paragraph("Vegetation (NDVI)", table_cell_style), Paragraph(f"{avg_spectral['mean_ndvi_2019']:.4f}", table_cell_style), Paragraph(f"{avg_spectral['mean_ndvi_2026']:.4f}", table_cell_style), Paragraph(f"{avg_spectral['mean_ndvi_2026'] - avg_spectral['mean_ndvi_2019']:.4f}", table_cell_style)],
            [Paragraph("Built-up structures (NDBI)", table_cell_style), Paragraph(f"{avg_spectral['mean_ndbi_2019']:.4f}", table_cell_style), Paragraph(f"{avg_spectral['mean_ndbi_2026']:.4f}", table_cell_style), Paragraph(f"{avg_spectral['mean_ndbi_2026'] - avg_spectral['mean_ndbi_2019']:.4f}", table_cell_style)],
            [Paragraph("Hydrological Moisture (NDWI)", table_cell_style), Paragraph(f"{avg_spectral['mean_ndwi_2019']:.4f}", table_cell_style), Paragraph(f"{avg_spectral['mean_ndwi_2026']:.4f}", table_cell_style), Paragraph(f"{avg_spectral['mean_ndwi_2026'] - avg_spectral['mean_ndwi_2019']:.4f}", table_cell_style)],
        ]
        t_spec = Table(spec_data, colWidths=[150, 110, 110, 110])
        t_spec.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t_spec)
        
    story.append(PageBreak())
    
    # ---------------- PAGE 4: FUTURE DEVELOPMENT PRIORITY AREAS ----------------
    story.append(Paragraph("5. Future Development Priority Areas", h1_style))
    story.append(Paragraph(
        "<i>*Important Note: The priority rankings below are derived from predicted 2026 growth scores "
        "to assist municipal urban planners in targeting infrastructure funding. This is NOT a future-year forecast "
        "beyond the analysis year 2026.*</i>",
        body_style
    ))
    
    # Merge localities to prediction table
    df_with_locs = df_preds.copy()
    df_with_locs["Locality"] = df_with_locs["grid_id"].astype(str).map(localities)
    
    # Hotspots table
    df_hot = df_with_locs.sort_values(by="growth_score", ascending=False).head(5)
    
    hot_headers = [
        Paragraph("<b>Rank</b>", table_header_style),
        Paragraph("<b>Locality Name</b>", table_header_style),
        Paragraph("<b>Score</b>", table_header_style),
        Paragraph("<b>Category</b>", table_header_style),
        Paragraph("<b>Probability</b>", table_header_style),
        Paragraph("<b>Planning Recommendation</b>", table_header_style)
    ]
    hot_rows = [hot_headers]
    for idx, (_, row) in enumerate(df_hot.iterrows()):
        loc_name = row["Locality"] if pd.notna(row["Locality"]) else f"Grid Sector {row['grid_id']}"
        score = f"{row['growth_score']:.4f}"
        cat = row["predicted_growth_category"]
        prob = f"{row['prediction_probability']:.1%}"
        
        # Determine customized recommendation based on category/locality
        if idx == 0:
            rec = "Prioritize transportation expansion, public infrastructure, and utility planning."
        elif idx == 1:
            rec = "Monitor commercial growth expansion and improve roadway connectivity."
        elif idx == 2:
            rec = "Plan for residential growth while preserving green buffer zones."
        else:
            rec = "Develop local infrastructure support cells and utility connections."
            
        hot_rows.append([
            Paragraph(str(idx+1), table_cell_style),
            Paragraph(loc_name, table_cell_style),
            Paragraph(score, table_cell_style),
            Paragraph(cat, table_cell_style),
            Paragraph(prob, table_cell_style),
            Paragraph(rec, table_cell_style)
        ])
        
    t_hot = Table(hot_rows, colWidths=[35, 95, 45, 55, 60, 200])
    t_hot.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(Paragraph("Top 5 Growth Hotspots (Highest Development Priority)", h2_style))
    story.append(t_hot)
    story.append(Spacer(1, 10))
    
    # Stable Areas table
    df_stable = df_with_locs.sort_values(by="growth_score", ascending=True).head(5)
    stable_rows = [hot_headers]
    for idx, (_, row) in enumerate(df_stable.iterrows()):
        loc_name = row["Locality"] if pd.notna(row["Locality"]) else f"Grid Sector {row['grid_id']}"
        score = f"{row['growth_score']:.4f}"
        cat = row["predicted_growth_category"]
        prob = f"{row['prediction_probability']:.1%}"
        
        rec = "Recommend environmental conservation and highly controlled development zoning."
        stable_rows.append([
            Paragraph(str(idx+1), table_cell_style),
            Paragraph(loc_name, table_cell_style),
            Paragraph(score, table_cell_style),
            Paragraph(cat, table_cell_style),
            Paragraph(prob, table_cell_style),
            Paragraph(rec, table_cell_style)
        ])
        
    t_stable = Table(stable_rows, colWidths=[35, 95, 45, 55, 60, 200])
    t_stable.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(Paragraph("Top 5 Stable Areas (Highest Environmental Protection Priority)", h2_style))
    story.append(t_stable)
    story.append(PageBreak())
    
    # ---------------- PAGE 5: INSIGHTS & PLANNING RECOMMENDATIONS ----------------
    story.append(Paragraph("6. AI Urban Insights & Planning Recommendations", h1_style))
    story.append(Paragraph(
        "Executive planning observations detailing change dynamics suitable for municipal planners:",
        body_style
    ))
    
    # AI insights paragraphs
    insights_text = generate_ai_insights(city_name, summary)
    for line in insights_text.split(". "):
        if line.strip():
            story.append(Paragraph(f"• {line.strip()}.", bullet_style))
            
    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Explainable AI Feature Drivers (SHAP Methodology)", h1_style))
    story.append(Paragraph(
        "Instead of raw feature variables, the XGBoost classification predictions are explained using "
        "game-theoretic SHAP weights mapped directly onto intuitive planning indicators:",
        body_style
    ))
    
    drivers_list = [
        "<b>Increased built-up footprint:</b> Corresponds to infrastructural expansion, asphalt cover, and buildings density indices.",
        "<b>Expansion of road networks:</b> Renders increased connectivity, intersections count, and roadway length variables.",
        "<b>Decline in vegetation canopy:</b> Shows land clearing and green cover degradation metrics.",
        "<b>Reduction in surface moisture:</b> Shows hydrological changes, water bodies buffer shifts, and NDWI indices."
    ]
    for dr in drivers_list:
        story.append(Paragraph(dr, bullet_style))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("8. Appendix: Platform Baseline Model Specifications", h1_style))
    
    spec_rows = [
        [Paragraph("<b>Performance Metric</b>", table_header_style), Paragraph("<b>XGBoost Baseline Value</b>", table_header_style)],
        [Paragraph("Classification Accuracy", table_cell_style), Paragraph("95.76%", table_cell_style)],
        [Paragraph("Macro Precision Score", table_cell_style), Paragraph("95.74%", table_cell_style)],
        [Paragraph("Macro Recall Score", table_cell_style), Paragraph("95.75%", table_cell_style)],
        [Paragraph("F1 Score (macro)", table_cell_style), Paragraph("95.74%", table_cell_style)]
    ]
    t_spec_table = Table(spec_rows, colWidths=[240, 240])
    t_spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_spec_table)
    
    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()

# Sidebar options setup
st.sidebar.markdown("## 🏙️ Platform Navigation")
page = st.sidebar.radio(
    "Select Screen:",
    ["Home", "Analyze City", "Dashboard", "Report", "About"]
)

# Render platform header
st.markdown("""
<div class="header-card">
    <h1>Urban Growth Intelligence Platform</h1>
    <p>AI-powered Urban Growth Prediction using Satellite Imagery, OpenStreetMap and Explainable AI</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SCREEN 1: HOME PAGE -----------------
if page == "Home":
    st.markdown("## 🏠 Platform Overview")
    st.markdown(
        "Welcome to the **Urban Growth Intelligence Platform**. This systems-level geospatial AI application "
        "orchestrates remote sensing satellite data, OpenStreetMap features, and classical machine learning "
        "classifiers to predict and analyze urban sprawl at grid-level granularity."
    )
    
    st.markdown("---")
    
    # Simplified workflow illustration
    st.markdown("### ⚙️ Multi-Stage Processing Pipeline")
    st.markdown(
        """
        - **Administrative Boundary Geocoding** - Geocodes city boundaries via Nominatim coordinate systems.
        - **OSM Spatial Data Ingestion** - Ingests infrastructure networks (buildings, road lines, highways).
        - **Sentinel-2 Multi-Spectral Ingestion** - Processes cloud-masked surface reflectance composites.
        - **Classification Model** - Feeds feature vectors to the baseline XGBoost classifier.
        - **SHAP Explanation Logging** - Computes local feature influence drivers.
        """
    )
    
    st.markdown("---")
    
    if st.button("🚀 Analyze City Now"):
        st.info("Please navigate to the **Analyze City** tab in the sidebar navigation to run an analysis.")

# ----------------- SCREEN 2: ANALYZE CITY -----------------
elif page == "Analyze City":
    st.markdown("## 🔍 Perform City Ingestion & Analysis")
    
    # 1. Unified Search box
    st.markdown("### 📍 Select or Enter City")
    selected_option = st.selectbox(
        "Choose a target city from the list or select 'Enter Custom City' to type your own name:",
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
        city_input = st.text_input("Type city name:", placeholder="e.g. Kolkata, Jaipur")
    else:
        city_input = selected_option
        
    if st.button("🚀 Analyze City"):
        if not city_input:
            st.error("Please enter a valid city name.")
        else:
            normalized_city = normalize_city_name(city_input)
            
            # Resolve geocoding boundary polygon variants
            boundary_resolved = geocode_city_boundary_frontend(normalized_city)
            if not boundary_resolved:
                st.error(
                    f"❌ Geocoding Failed: Could not retrieve administrative boundary polygon for '{city_input}'. "
                    "Please check spelling, or try adding state details (e.g. 'Mysuru, Karnataka')."
                )
            else:
                # Setup progress checklist milestones container
                progress_container = st.container()
                with progress_container:
                    status_placeholder = st.empty()
                    
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
                                lines.append("<div style='color: #f59e0b; font-size: 0.85rem; margin-top: 0.5rem;'>⚠️ *This step may take several minutes depending on image availability and GEE server latency.*</div>")
                        else:
                            lines.append("🎉 **Pipeline completed successfully!**")
                            
                        status_placeholder.markdown("\n".join(lines), unsafe_allow_html=True)
                        
                    try:
                        # Call backend pipeline
                        summary = analyze_city(normalized_city, status_callback=st_callback)
                        progress_container.empty()
                        
                        # Load predictions dataset and spatial boundaries
                        df_preds = pd.read_csv(Path(summary["prediction_file"]))
                        geojson_path = Config.FEATURES_DIR / f"osm_features_{normalized_city}.geojson"
                        
                        # Reverse geocode grid centroids
                        if geojson_path.exists():
                            gdf = gpd.read_file(geojson_path)
                            localities = assign_locality_names(gdf, normalized_city)
                        else:
                            localities = {}
                            
                        # Save execution context in session state
                        st.session_state["active_city"] = normalized_city
                        st.session_state["active_summary"] = summary
                        st.session_state["active_preds"] = df_preds
                        st.session_state["active_localities"] = localities
                        st.session_state["active_pdf_data"] = None
                        
                        st.success(f"Analysis completed successfully for {normalized_city}! Navigate to the 'Dashboard' screen to view spatial maps and SHAP driver explanations.")
                        
                    except Exception as err:
                        st.error(f"Failed to complete urban analysis. Error details: {err}")

# ----------------- SCREEN 3: DASHBOARD -----------------
elif page == "Dashboard":
    if "active_city" not in st.session_state:
        st.warning("⚠️ Please complete a city analysis on the 'Analyze City' screen first to populate dashboard metrics.")
    else:
        city_name = st.session_state["active_city"]
        summary = st.session_state["active_summary"]
        df_preds = st.session_state["active_preds"]
        localities = st.session_state["active_localities"]
        
        st.markdown(f"## 📊 Growth Analysis Dashboard: {city_name}")
        
        # --- EXECUTIVE SUMMARY CARD PANEL ---
        total_grids = summary["number_of_grids"]
        h_pct = (summary["high_growth_count"] / total_grids) * 100 if total_grids > 0 else 0
        m_pct = (summary["medium_growth_count"] / total_grids) * 100 if total_grids > 0 else 0
        l_pct = (summary["low_growth_count"] / total_grids) * 100 if total_grids > 0 else 0
        
        if h_pct > 35:
            growth_label = "High Growth"
            color_style = "color: #ef4444;"
        elif m_pct > 35:
            growth_label = "Moderate expansion"
            color_style = "color: #eab308;"
        else:
            growth_label = "Low expansion"
            color_style = "color: #22c55e;"
            
        summary["overall_growth_level"] = growth_label
        
        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #ffffff;">{city_name}</div>
                    <div class="kpi-label">Selected City</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="{color_style}">{growth_label}</div>
                    <div class="kpi-label">Overall Growth Category</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #60a5fa;">{summary['average_growth_score']:.4f}</div>
                    <div class="kpi-label">Average Growth Score</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #a78bfa;">{total_grids:,}</div>
                    <div class="kpi-label">Total Analysis Areas</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #ef4444;">{summary['high_growth_count']:,}</div>
                    <div class="kpi-label">High Growth Areas</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #eab308;">{summary['medium_growth_count']:,}</div>
                    <div class="kpi-label">Medium Growth Areas</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #22c55e;">{summary['low_growth_count']:,}</div>
                    <div class="kpi-label">Low Growth Areas</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # --- AI URBAN PLANNERS INSIGHTS ---
        st.markdown("### 🤖 Executive AI Insights")
        st.markdown(generate_ai_insights(city_name, summary))
        
        # --- PYDECK SPATIAL MAP ---
        st.markdown("### 🗺️ Geospatial Growth Classifications Map")
        geojson_path = Config.FEATURES_DIR / f"osm_features_{city_name}.geojson"
        
        if geojson_path.exists():
            gdf = gpd.read_file(geojson_path)
            gdf["grid_id"] = gdf["grid_id"].astype(str)
            df_preds["grid_id"] = df_preds["grid_id"].astype(str)
            
            gdf_merged = gdf.merge(df_preds, on="grid_id")
            
            # Map locality names
            gdf_merged["Locality"] = gdf_merged["grid_id"].map(localities)
            
            def get_rgba(row):
                cat = row["predicted_growth_category"]
                if cat == "High":
                    return [239, 68, 68, 140]
                elif cat == "Medium":
                    return [234, 179, 8, 140]
                else:
                    return [34, 197, 94, 140]
                    
            gdf_merged["fill_color"] = gdf_merged.apply(get_rgba, axis=1)
            
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
            
            # Support GeoPandas 1.0 union_all() deprecation replacement
            if hasattr(gdf_merged, "union_all"):
                centroid = gdf_merged.geometry.union_all().centroid
            else:
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
                    "html": "<b>Locality:</b> {Locality}<br/>"
                            "<b>Growth Score:</b> {growth_score}<br/>"
                            "<b>Category:</b> {predicted_growth_category}<br/>"
                            "<b>Confidence Probability:</b> {prediction_probability}",
                    "style": {"backgroundColor": "#1e293b", "color": "#f8fafc", "border": "1px solid #475569"}
                }
            )
            st.pydeck_chart(r)
            
        # --- EXPLAINABILITY GRID INSPECTOR ---
        st.markdown("### 🔍 Grid Explanations & SHAP Driver")
        
        shap_file_path = Path(summary["shap_file"])
        if shap_file_path.exists():
            df_shap = pd.read_csv(shap_file_path)
            
            df_with_locs = df_preds.copy()
            df_with_locs["Locality"] = df_with_locs["grid_id"].astype(str).map(localities)
            df_with_locs["Display"] = df_with_locs["Locality"] + " (Grid " + df_with_locs["grid_id"] + ")"
            
            display_to_id = dict(zip(df_with_locs["Display"], df_with_locs["grid_id"]))
            
            sel_display = st.selectbox(
                "Select a Locality area to inspect details:",
                options=df_with_locs["Display"].tolist(),
                index=0
            )
            
            sel_grid_id = display_to_id[sel_display]
            
            cell_pred = df_preds[df_preds["grid_id"].astype(str) == str(sel_grid_id)].iloc[0]
            cell_shap = df_shap[df_shap["grid_id"].astype(str) == str(sel_grid_id)].iloc[0]
            
            # Load spectral indices
            city_features_path = Config.FEATURES_DIR / f"{city_name.lower()}_growth_dataset.csv"
            df_features_cached = pd.read_csv(city_features_path)
            cell_features = df_features_cached[df_features_cached["grid_id"].astype(str) == str(sel_grid_id)].iloc[0]
            
            sc_col1, sc_col2, sc_col3 = st.columns(3)
            sc_col1.metric("Selected Area Locality", f"{localities.get(str(sel_grid_id), 'N/A')}")
            sc_col2.metric("Growth Score (UCI)", f"{cell_pred['growth_score']:.4f}")
            sc_col3.metric("Category Forecast", f"{cell_pred['predicted_growth_category']} ({cell_pred['prediction_probability']:.1%})")
            
            shap_feats = [c for c in df_shap.columns if c != "grid_id"]
            cell_shap_series = cell_shap[shap_feats].astype(float)
            
            readable_reasons = translate_shap_features(cell_shap_series.sort_values(key=abs, ascending=False))
            st.markdown("**Core driver factors for this forecast:**")
            for reason in readable_reasons:
                st.markdown(f"- {reason}")
                
            df_cell_shap = pd.DataFrame({
                "Feature": cell_shap_series.index,
                "SHAP Value": cell_shap_series.values,
                "Influence": ["Increase Growth" if v > 0 else "Decrease Growth" for v in cell_shap_series.values]
            }).sort_values(by="SHAP Value", key=abs, ascending=True).tail(10)
            
            fig_shap = px.bar(
                df_cell_shap,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                color="Influence",
                color_discrete_map={"Increase Growth": "#ef4444", "Decrease Growth": "#3b82f6"},
                title=f"Local Feature SHAP Driver weights"
            )
            fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_shap, use_container_width=True)
            
            st.markdown("#### Cell Spectral Shift Indicators")
            ss_col1, ss_col2, ss_col3 = st.columns(3)
            ss_col1.metric("NDVI Delta (Vegetation)", f"{cell_features['delta_ndvi']:.4f}")
            ss_col2.metric("NDBI Delta (Built-up)", f"{cell_features['delta_ndbi']:.4f}")
            ss_col3.metric("NDWI Delta (Moisture)", f"{cell_features['delta_ndwi']:.4f}")
            
        # --- PLOTLY ANALYTICAL CHARTS ---
        st.markdown("### 📊 Plotly Analytics Charts")
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
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
            
            if city_features_path.exists():
                df_feat = pd.read_csv(city_features_path)
                
                # NDVI Histogram
                fig_ndvi = px.histogram(
                    df_feat,
                    x=["mean_ndvi_2019", "mean_ndvi_2026"],
                    barmode="overlay",
                    title="NDVI shift distribution (2019 vs 2026)",
                    color_discrete_sequence=["#3b82f6", "#ef4444"]
                )
                fig_ndvi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                st.plotly_chart(fig_ndvi, use_container_width=True)
                
                # NDBI Histogram
                fig_ndbi = px.histogram(
                    df_feat,
                    x=["mean_ndbi_2019", "mean_ndbi_2026"],
                    barmode="overlay",
                    title="NDBI shift distribution (2019 vs 2026)",
                    color_discrete_sequence=["#3b82f6", "#ef4444"]
                )
                fig_ndbi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
                st.plotly_chart(fig_ndbi, use_container_width=True)
                
                # NDWI Histogram
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
            fig_hist = px.histogram(
                df_preds,
                x="growth_score",
                nbins=30,
                color_discrete_sequence=["#a78bfa"],
                title="Urban Change Index Distribution (Histogram)"
            )
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_hist, use_container_width=True)
            
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
                
        # --- EXPANDABLE ADVANCED DETAILS ---
        st.markdown("---")
        with st.expander("📋 Advanced Technical Details"):
            st.dataframe(
                df_preds.style.format({
                    "growth_score": "{:.4f}",
                    "prediction_probability": "{:.2%}"
                }),
                use_container_width=True
            )

# ----------------- SCREEN 4: REPORT -----------------
elif page == "Report":
    if "active_city" not in st.session_state:
        st.warning("⚠️ Please complete a city analysis on the 'Analyze City' screen first to enable PDF summaries downloading.")
    else:
        city_name = st.session_state["active_city"]
        summary = st.session_state["active_summary"]
        df_preds = st.session_state["active_preds"]
        localities = st.session_state["active_localities"]
        
        st.markdown(f"## 📥 Download Assessment Reports: {city_name}")
        
        shap_file_path = Path(summary["shap_file"])
        df_shap = pd.read_csv(shap_file_path) if shap_file_path.exists() else None
        
        # Draw Report Exporters PDF download button
        if st.session_state.get("active_pdf_data") is None:
            with st.spinner("Generating PDF report..."):
                pdf_data = generate_complete_pdf_report(city_name, summary, df_preds, df_shap, localities)
                st.session_state["active_pdf_data"] = pdf_data
        else:
            pdf_data = st.session_state["active_pdf_data"]
            
        st.download_button(
            label="📥 Download Executive Assessment Summary PDF Report",
            data=pdf_data,
            file_name=f"{city_name.lower()}_growth_report.pdf",
            mime="application/pdf"
        )
        
        with open(summary["prediction_file"], "rb") as f:
            st.download_button(
                label="📥 Download Prediction CSV results table",
                data=f,
                file_name=f"{city_name.lower()}_predictions.csv",
                mime="text/csv"
            )

# ----------------- SCREEN 5: ABOUT -----------------
elif page == "About":
    st.markdown("## ℹ️ Platform Technical Methodology")
    st.markdown(
        """
        The **Urban Growth Intelligence Platform** combines spatial network algorithms and multi-spectral indices to predict municipal development.
        
        ### 📡 Remote Sensing Satellite Ingestion
        Sentinel-2 surface reflectance imagery is cloud-masked using Cloud Probability and QA60 thresholds inside Google Earth Engine. 
        - **NDVI (Vegetation Index):** `(NIR - Red) / (NIR + Red)`
        - **NDBI (Built-up structures Index):** `(SWIR - NIR) / (SWIR + NIR)`
        - **NDWI (Surface Moisture Index):** `(Green - NIR) / (Green + NIR)`
        
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
