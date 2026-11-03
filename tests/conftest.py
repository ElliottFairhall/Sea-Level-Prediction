"""Shared fixtures: offline copies of every data source, in the real formats.

Nothing in the test suite touches the network. The fakes mimic the files and
responses checked against the live services, and a test can ask for an outage
to see how the page copes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta

import numpy as np
import pytest
import streamlit as st

from src import sources

COLORADO_FILE_URL = (
    "https://sealevel.colorado.edu/files/2026_rel2/gmsl_2026rel2_seasons_rmvd.txt"
)


def colorado_text(end: float = 2026.45) -> str:
    """A 10-day global series from late 1992 to ``end`` in the Colorado layout."""
    years = np.arange(1992.96, end, 10 / 365.25)
    rng = np.random.default_rng(1)
    level = (
        -20
        + 3.4 * (years - 1993)
        + 0.04 * (years - 1993) ** 2
        + rng.normal(0, 2, len(years))
    )
    lines = ["# Date      2026_rel2 w/ seasonal signals and GIA removed (mm)"]
    lines += [f"{y:.10f}  {v:.3f}" for y, v in zip(years, level, strict=True)]
    return "\n".join(lines)


def psmsl_text(station_id: str) -> str:
    """A monthly RLR file from 1915 to late 2025, with a few missing months."""
    rng = np.random.default_rng(int(station_id))
    rows = []
    for i, year in enumerate(np.arange(1915.0, 2025.95, 1 / 12)):
        month = i % 12
        if i % 97 == 0:
            rows.append(f"{year + 0.04:9.4f};  -99999;  0;  000")
            continue
        level = (
            7000
            + 2.0 * (year - 1915)
            + 60 * np.sin(2 * np.pi * month / 12)
            + rng.normal(0, 25)
        )
        rows.append(f"{year + 0.04:9.4f};  {level:6.0f};  0;  000")
    return "\n".join(rows)


def noaa_text() -> str:
    """Monthly means in metres from 1900 to September 2026, in NOAA's JSON layout."""
    rng = np.random.default_rng(7)
    data = []
    for year in range(1900, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 9:
                break
            metres = (
                0.003 * (year - 1900)
                + 0.05 * np.sin(2 * np.pi * month / 12)
                + rng.normal(0, 0.02)
            )
            data.append(
                {"year": str(year), "month": str(month), "MSL": f"{metres:.3f}"}
            )
    return json.dumps({"metadata": {"id": "x"}, "data": data})


def uhslc_text() -> str:
    """Newlyn-like daily levels from 1915 to 2016 in ERDDAP's two-header CSV layout."""
    rng = np.random.default_rng(3)
    days = np.arange(np.datetime64("1915-04-23"), np.datetime64("2017-01-01"))
    years = (
        days.astype("datetime64[Y]").astype(int)
        + 1970
        + (days.astype(int) % 365.25) / 365.25
    )
    level = (
        3000
        + 1.9 * (years - 1915)
        + 120 * np.sin(2 * np.pi * years)
        + rng.normal(0, 60, len(days))
    )
    lines = ["time,sea_level,uhslc_id", "UTC,millimeters,"]
    for day, value in zip(days, level, strict=True):
        shown = "NaN" if rng.random() < 0.01 else f"{value:.0f}"
        lines.append(f"{day}T12:00:00Z,{shown},294")
    return "\n".join(lines)


def ea_stations_text() -> str:
    """Environment Agency station list with two ordnance-datum tide gauges."""

    def item(label: str, ref: str) -> dict:
        """One station record with a single ordnance-datum tide measure."""
        ident = f"http://environment.data.gov.uk/flood-monitoring/id/measures/{ref}-level-tidal_level-Mean-15_min-mAOD"
        return {"label": label, "stationReference": ref, "measures": [{"@id": ident}]}

    return json.dumps(
        {"items": [item("Newlyn", "E70124"), item("Avonmouth Portbury", "E72639")]}
    )


def ea_readings_text() -> str:
    """Two days of 15-minute tide readings ending a few minutes ago."""
    end = datetime.now(sources.UTC).replace(second=0, microsecond=0) - timedelta(
        minutes=8
    )
    end -= timedelta(minutes=end.minute % 15)
    items = []
    for step in range(48 * 4):
        stamp = end - timedelta(minutes=15 * step)
        hours = (stamp - end).total_seconds() / 3600
        items.append(
            {
                "dateTime": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "value": round(0.2 + 1.5 * np.sin(2 * np.pi * hours / 12.42), 3),
            }
        )
    return json.dumps({"items": items})


def make_http_get(
    colorado_end: float = 2026.45, down: tuple[str, ...] = ()
) -> Callable[..., str]:
    """Build a stand-in for ``sources.http_get`` that serves the fakes.

    Args:
        colorado_end: Decimal year the satellite file runs to.
        down: URL fragments that should fail, to simulate an outage.

    Returns:
        A function with the same signature as ``sources.http_get``.
    """

    def fake(url: str, params: dict[str, str] | None = None) -> str:
        """Serve the fake for ``url``, or fail if that source is 'down'."""
        if any(fragment in url for fragment in down):
            raise sources.SourceError(f"simulated outage for {url}")
        if url == sources.COLORADO_INDEX:
            return '<a href="/data/2026_rel2">2026_rel2</a><a href="/data/2020_rel1">2020_rel1</a>'
        if re.search(r"colorado.edu/data/\d{4}rel\d+$", url):
            return f'<a href="{COLORADO_FILE_URL}">Raw Data (ASCII)</a>'
        if "gmsl_" in url:
            return colorado_text(colorado_end)
        if "psmsl.org" in url:
            match = re.search(r"/(\d+)\.rlrdata", url)
            assert match is not None, url
            return psmsl_text(match.group(1))
        if "tidesandcurrents" in url:
            return noaa_text()
        if "uhslc" in url:
            return uhslc_text()
        if url == sources.EA_STATIONS_URL:
            return ea_stations_text()
        if "/readings" in url:
            return ea_readings_text()
        raise sources.SourceError(f"no fake for {url}")

    return fake


@pytest.fixture(scope="module", autouse=True)
def offline() -> Iterator[None]:
    """Serve every data source from the fakes for each test module."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("src.sources.http_get", make_http_get())
        st.cache_data.clear()
        yield


@pytest.fixture(autouse=True)
def fresh_cache() -> None:
    """Start each test with empty Streamlit caches."""
    st.cache_data.clear()
