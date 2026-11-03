"""Cached loaders for the page.

Every loader returns ``(data, error)``. On success ``error`` is None; on failure
``data`` is None and ``error`` says why, so one unavailable source leaves the
rest of the page working instead of stopping it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src import analysis, sources

logger = logging.getLogger(__name__)

EPA_FILE = (
    Path(__file__).resolve().parent.parent / "assets" / "data" / "epa-sea-level.csv"
)
LONG_TTL = 6 * 3600
LIVE_TTL = 300
LIVE_HOURS = 48

Result = tuple[pd.DataFrame | None, str | None]


def _safe(call, label: str) -> Result:
    """Run a fetch and turn any expected failure into an error message."""
    try:
        return call(), None
    except (sources.SourceError, ValueError, KeyError) as error:
        logger.warning("%s unavailable: %s", label, error)
        return None, f"{label} could not be loaded ({error})"


@st.cache_data(show_spinner=False)
def load_epa() -> pd.DataFrame:
    """Load the bundled EPA / CSIRO file in millimetres."""
    return analysis.load_epa(EPA_FILE)


@st.cache_data(ttl=LONG_TTL, show_spinner="Fetching the satellite record...")
def load_satellite() -> Result:
    """Fetch the University of Colorado global satellite series."""
    return _safe(sources.fetch_colorado, "The satellite record")


@st.cache_data(ttl=LONG_TTL, show_spinner="Fetching the tide gauge...")
def load_station(key: str) -> Result:
    """Fetch one gauge's monthly series.

    Args:
        key: A key from ``sources.STATION_BY_KEY``.
    """
    station = sources.STATION_BY_KEY[key]
    return _safe(lambda: sources.fetch_station(station), station.name)


@st.cache_data(ttl=LONG_TTL, show_spinner="Fetching a century of Newlyn readings...")
def load_newlyn_daily() -> Result:
    """Fetch Newlyn's daily research-quality series."""
    return _safe(sources.fetch_newlyn_daily, "The Newlyn daily record")


@st.cache_data(ttl=LONG_TTL, show_spinner=False)
def load_ea_stations() -> tuple[list[dict[str, str]] | None, str | None]:
    """List Environment Agency tide gauges."""
    try:
        return sources.fetch_ea_tide_stations(), None
    except (sources.SourceError, ValueError) as error:
        return None, f"The live tide gauge list could not be loaded ({error})"


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def load_ea_readings(measure: str) -> Result:
    """Fetch the last two days of 15-minute readings for one gauge.

    Args:
        measure: Measure id from ``load_ea_stations``.
    """
    since = datetime.now(sources.UTC) - timedelta(hours=LIVE_HOURS)
    return _safe(
        lambda: sources.fetch_ea_readings(measure, since), "The live tide readings"
    )
