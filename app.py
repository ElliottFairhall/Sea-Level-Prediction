"""Sea level, from 1880 to now: entry point."""

from datetime import datetime

import pandas as pd
import streamlit as st

from src import analysis, data, sources
from src.ui import (
    DEFAULT_MODERN_START,
    age_text,
    render_config,
    render_credits,
    render_extremes,
    render_footer,
    render_freshness,
    render_intro,
    render_limits,
    render_live,
    render_scenarios,
    render_sidebar,
    render_station,
    render_stitch,
    render_topbar,
    render_trend_section,
)


def _row(source: str, latest: pd.Timestamp | None, note: str = "") -> dict[str, str]:
    """One line of the freshness table."""
    if latest is None:
        return {
            "Source": source,
            "Latest data": "unavailable",
            "Behind today": note or "-",
        }
    return {
        "Source": source,
        "Latest data": f"{latest:%d %b %Y}",
        "Behind today": note or age_text(latest),
    }


def _live_row() -> dict[str, str]:
    """The freshness line for the Environment Agency feed."""
    label = "Live tide gauges (Environment Agency)"
    stations, _ = data.load_ea_stations()
    if not stations:
        return _row(label, None)
    measure = next(
        (s["measure"] for s in stations if s["label"] == "Newlyn"),
        stations[0]["measure"],
    )
    readings, _ = data.load_ea_readings(measure)
    if readings is None or readings.empty:
        return _row(label, None)
    latest = readings["time"].max()
    minutes = int(
        (datetime.now(sources.UTC) - latest.to_pydatetime()).total_seconds() // 60
    )
    return {
        "Source": label,
        "Latest data": f"{latest:%d %b %Y %H:%M} UTC",
        "Behind today": f"{minutes} minutes",
    }


def freshness_rows(
    epa: pd.DataFrame, satellite: pd.DataFrame | None
) -> list[dict[str, str]]:
    """Describe how current each data source is, from what it actually returned.

    Args:
        epa: The bundled long record.
        satellite: The satellite series, or None if it failed to load.

    Returns:
        Rows for the freshness table.
    """
    rows = [
        _row(
            "Global satellite record (University of Colorado)",
            None if satellite is None else satellite["date"].max(),
        )
    ]
    for label, key in (
        ("UK tide gauges (PSMSL), shown for Newlyn", "psmsl-202"),
        ("US tide gauges (NOAA), shown for The Battery", "noaa-8518750"),
    ):
        monthly, _ = data.load_station(key)
        rows.append(_row(label, None if monthly is None else monthly["date"].max()))
    daily, _ = data.load_newlyn_daily()
    rows.append(
        _row(
            "Newlyn daily readings (UHSLC)",
            None if daily is None else daily["date"].max(),
            note="series ends here",
        )
    )
    rows.append(_live_row())
    rows.append(
        _row(
            "Original reconstruction (EPA / CSIRO)",
            pd.Timestamp(int(epa["year"].max()), 12, 31),
            note="fixed file, no longer updated",
        )
    )
    return rows


def _modern_rate(sat_annual: pd.DataFrame | None) -> float | None:
    """Recent global rate of rise (mm a year) from the satellite years, if loaded."""
    if sat_annual is None or len(sat_annual) < 4:
        return None
    return analysis.trend(sat_annual["year"], sat_annual["level_mm"]).slope


def _uk_lag() -> str:
    """How far behind the UK gauge records are, from what Newlyn returned."""
    monthly, _ = data.load_station("psmsl-202")
    if monthly is None:
        return "typically about a year"
    return age_text(monthly["date"].max()).replace(" ago", "")


def main() -> None:
    """Render the page. Each section copes with a missing source on its own."""
    render_config()
    render_topbar()
    units = render_sidebar()

    epa = data.load_epa()
    raw, satellite_error = data.load_satellite()
    stitched = sat_annual = None
    modern_start = DEFAULT_MODERN_START
    if raw is not None:
        sat_annual = analysis.annual_means(raw, analysis.MIN_SATELLITE_CYCLES)
        stitched = analysis.stitch(epa[["year", "level_mm"]], sat_annual)
        modern_start = int(sat_annual["year"].min())

    render_intro(stitched, epa)
    render_freshness(freshness_rows(epa, raw))

    if stitched is not None and raw is not None:
        render_stitch(epa, raw, stitched, units)
        render_trend_section(stitched.series, units, modern_start)
    else:
        st.warning(satellite_error)

    render_station(sat_annual, units)

    daily, daily_error = data.load_newlyn_daily()
    if daily is not None:
        render_extremes(daily, units)
    else:
        st.warning(daily_error)

    if stitched is not None:
        render_scenarios(stitched.series, units, modern_start)

    render_live(units, _modern_rate(sat_annual))
    render_limits(_uk_lag())
    release = (
        sources.release_name(raw.attrs.get("source_url")) if raw is not None else None
    )
    render_credits(release)
    render_footer()


if __name__ == "__main__":
    main()
