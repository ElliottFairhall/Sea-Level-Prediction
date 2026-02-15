from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy.stats import linregress

# Configuration
PAGE_TITLE = "Sea Level Prediction"
PAGE_ICON = ":chart_with_upwards_trend:"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

# Project paths
CURRENT_DIR = Path(__file__).parent
CSV_FILE = CURRENT_DIR / "assets" / "data" / "epa-sea-level.csv"
CSS_FILE = CURRENT_DIR / "styles" / "main.css"
IMAGE_FILE = CURRENT_DIR / "assets" / "images" / "Sea Level.jpg"


def load_css(file_path: Path) -> None:
    if file_path.exists():
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class='header-container'>
            <h1 class='main-title'>Sea Level Prediction</h1>
            <p class='sub-title'>ANALYZING GLOBAL OCEAN TRENDS</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(file_path: Path):
    return pd.read_csv(file_path)


def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            "<h2 style='font-family:Outfit; margin-bottom:0;'>Data Engine</h2>",
            unsafe_allow_html=True,
        )
        st.caption("Configuring high-fidelity signals.")
        st.markdown("---")

        years = sorted(df["Year"].unique())
        start_year = st.slider("Start Year", int(min(years)), 2020, 1880)
        end_year = st.slider("Forecast Year", 2020, 2050, 2050)

        st.markdown("---")
        st.caption("Environment: Sea Level Forecast V1.0")
        st.caption("Aesthetic: Soft Rose / Sky")
    return start_year, end_year


def render_metrics(df: pd.DataFrame):
    total_records = len(df)
    latest_level = df["CSIRO Adjusted Sea Level"].iloc[-1]
    avg_rise = (
        df["CSIRO Adjusted Sea Level"].iloc[-1] - df["CSIRO Adjusted Sea Level"].iloc[0]
    ) / (df["Year"].iloc[-1] - df["Year"].iloc[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{total_records}")
    col2.metric("Latest Sea Level", f"{latest_level:.2f} in")
    col3.metric("Avg. Annual Rise", f"{avg_rise:.3f} in/yr")


def render_visualizations(df: pd.DataFrame, start_year: int, end_year: int):
    # Filter data for scatter
    mask = (df["Year"] >= start_year) & (df["Year"] <= 2020)
    filtered_df = df[mask]

    # Regression
    res = linregress(df["Year"], df["CSIRO Adjusted Sea Level"])
    x_pred = pd.Series([float(i) for i in range(start_year, end_year + 1)])
    y_pred = res.slope * x_pred + res.intercept

    # Create Plotly figure
    fig = go.Figure()

    # Scatter plot of raw data
    fig.add_trace(
        go.Scatter(
            x=filtered_df["Year"],
            y=filtered_df["CSIRO Adjusted Sea Level"],
            mode="markers",
            name="Historical Data",
            marker=dict(color="#fda4af", size=8, opacity=0.6),
        )
    )

    # Prediction line
    fig.add_trace(
        go.Scatter(
            x=x_pred,
            y=y_pred,
            mode="lines",
            name="Linear Trend",
            line=dict(color="#7dd3fc", width=3),
        )
    )

    fig.update_layout(
        title="Historical & Predicted Sea Level Rise",
        xaxis_title="Year",
        yaxis_title="Sea Level (inches)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=14, color="#f8fafc"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, width="stretch")


def main():
    load_css(CSS_FILE)
    render_header()

    df = load_data(CSV_FILE)
    start_year, end_year = render_sidebar(df)

    tab1, tab2 = st.tabs(["Project Overview", "Analysis"])

    with tab2:
        render_metrics(df)
        st.markdown("<br>", unsafe_allow_html=True)
        render_visualizations(df, start_year, end_year)

        st.markdown(
            """
        ### Impact Summary
        Over the past century, global sea levels have risen significantly. The current trend suggests a continuous increase,
        impacting coastal infrastructure and ecosystems worldwide. Linear regression highlights the persistent upward trajectory
        if current conditions continue.
        """
        )

    with tab1:
        if IMAGE_FILE.exists():
            image = Image.open(IMAGE_FILE)
            st.image(image, width="stretch")

        st.markdown(
            """
        ### Project Outline
        In this project, we provide an analysis of sea level changes looking at data since 1880.
        Rising sea levels have become a well-documented topic, with significant impacts on coastal communities,
        ecosystems, and infrastructure.

        ### Methodology
        - **Data Source:** EPA Global Average Absolute Sea Level Change.
        - **Visualizations:** High-fidelity interactive charts using Plotly.
        - **Analysis:** Linear regression using Scipy for forecasting future trends.
        """
        )


if __name__ == "__main__":
    main()
