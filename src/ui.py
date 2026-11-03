"""Page sections for the sea level app."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src import analysis, charts, data, sources
from src.analysis import Stitch, to_units

SITE_URL = "https://www.data-flakes.dev"
ROOT = Path(__file__).resolve().parent.parent
LATE_YEARS = 30
IPCC_URL = "https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-9/"
DEFAULT_MODERN_START = 1993


def length(mm: float, units: str, signed: bool = False) -> str:
    """Format a length for display.

    Args:
        mm: Length in millimetres.
        units: ``"mm"`` or ``"inches"``.
        signed: Always show a plus or minus sign.

    Returns:
        For example ``"272 mm"`` or ``"10.71 inches"``.
    """
    value = to_units(mm, units)
    sign = "+" if signed else ""
    return f"{value:{sign}.2f} inches" if units == "inches" else f"{value:{sign}.0f} mm"


def rate(mm_per_year: float, units: str) -> str:
    """Format a rate for display, for example ``"3.35 mm a year"``."""
    value = to_units(mm_per_year, units)
    return (
        f"{value:.3f} inches a year" if units == "inches" else f"{value:.2f} mm a year"
    )


def per_year(mm_per_year: float, units: str) -> str:
    """A bare rate number for a metric whose label carries the unit."""
    return f"{to_units(mm_per_year, units):.2f}"


def age_text(latest: pd.Timestamp) -> str:
    """Say how long ago a date was, in plain words."""
    days = (pd.Timestamp.now() - latest).days
    if days < 2:
        return "today"
    if days < 60:
        return f"{days} days ago"
    if days < 730:
        return f"{days / 30.44:.0f} months ago"
    return f"{days / 365.25:.0f} years ago"


def render_config() -> None:
    """Set the page configuration and load the stylesheet."""
    st.set_page_config(
        page_title="Sea level, from 1880 to now",
        page_icon=str(ROOT / "assets" / "images" / "favicon.png"),
        layout="wide",
    )
    css = (ROOT / "styles" / "main.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_topbar() -> None:
    """Show a plain breadcrumb back to the Data Flakes site."""
    st.markdown(
        f"""
        <nav class="site-bar">
            <a href="{SITE_URL}/" target="_blank">Data Flakes</a>
            <span>/</span>
            <a href="{SITE_URL}/projects/" target="_blank">Projects</a>
            <span>/</span>
            <span>Sea level</span>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render the settings.

    Returns:
        The chosen display unit, ``"mm"`` or ``"inches"``.
    """
    with st.sidebar:
        st.header("Settings")
        units = st.segmented_control(
            "Units", ["mm", "inches"], default="mm", key="units"
        )
        if st.button("Refresh the data"):
            st.cache_data.clear()
        st.caption(
            "Data is fetched when the page loads and kept for up to six hours; "
            "the live tide panel for five minutes."
        )
    return str(units or "mm")


def render_intro(stitched: Stitch | None, epa: pd.DataFrame) -> None:
    """Introduce the page with figures taken from the data.

    Args:
        stitched: The joined global record, or None if satellites are unavailable.
        epa: The bundled long record.
    """
    st.title("Sea level, from 1880 to now")
    if stitched is None:
        st.write(
            "This page rebuilds an old, fixed sea level dataset into something "
            "current. The satellite record could not be loaded, so only the "
            f"long record, which ends in {int(epa['year'].max())}, is shown for now."
        )
        return
    series = stitched.series
    rise = series["level_mm"].iloc[-1] - series["level_mm"].iloc[0]
    last = int(series["year"].max())
    first = int(series["year"].min())
    st.write(
        f"Global sea level has risen about {rise / 10:.0f} cm since {first}. This page "
        f"takes a tide gauge record that stopped in {int(epa['year'].max())}, joins "
        "it to satellite data that is updated every couple of months, and uses "
        "both to ask how fast the sea is rising, how sure we can be, and what it "
        "means on the coast you care about. Every number below is calculated from "
        f"the data when the page loads; the global figures run to {last}."
    )


def render_freshness(rows: list[dict[str, str]]) -> None:
    """Show how current each data source is.

    Args:
        rows: Dicts with ``Source``, ``Latest data`` and ``Behind today``.
    """
    st.subheader("How up to date is the data?")
    st.caption(
        "Different instruments report on different schedules. "
        "This is what each one gave when the page loaded."
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_stitch(
    epa: pd.DataFrame, raw: pd.DataFrame, stitched: Stitch, units: str
) -> None:
    """Show the two global records joined into one.

    Args:
        epa: The bundled long record.
        raw: The satellite 10-day series.
        stitched: The result of joining them.
        units: Display unit.
    """
    st.subheader("One ocean, two instruments")
    st.write(
        "For well over a century sea level was reconstructed from tide gauges. "
        f"Since {int(raw['date'].dt.year.min())} radar satellites have measured the whole ocean "
        "every ten days. They measure the same thing from a different baseline, so they cannot "
        f"simply be laid end to end. They share {stitched.overlap[0]} to {stitched.overlap[1]}, "
        "and that overlap is what lets us join them."
    )
    st.plotly_chart(
        charts.global_record(epa, raw, stitched.offset_mm, stitched.overlap, units),
        width="stretch",
    )
    _stitch_metrics(stitched, units)
    _stitch_technique()


def _stitch_metrics(stitched: Stitch, units: str) -> None:
    """Show how the join was made and how well the two records agree.

    Args:
        stitched: The result of joining the records.
        units: Display unit.
    """
    cols = st.columns(3)
    cols[0].metric(
        "Shift applied to the satellites",
        length(stitched.offset_mm, units, signed=True),
        help="The average gap between the records over their overlap.",
    )
    cols[1].metric(
        "Typical disagreement afterwards",
        length(stitched.rmse_mm, units),
        help="Root-mean-square gap between the records once aligned.",
    )
    cols[2].metric(
        "Year-to-year agreement",
        f"{stitched.correlation:.2f}",
        help="Correlation of the two records' annual changes. 1 is perfect agreement.",
    )
    st.caption(
        "The records agree closely on the overall rise but less on individual years, "
        "which is what you would expect from two different methods."
    )


def _stitch_technique() -> None:
    """Explain how two records with different baselines are joined."""
    with st.expander(
        "Technique: how to join two records that disagree on the baseline"
    ):
        st.write(
            "1. Average each record by calendar year, so both are annual.\n"
            "2. Over the years both exist, find the average gap between them. "
            "That gap is the offset.\n"
            "3. Add the offset to the satellite record so both sit on one scale.\n"
            "4. Check the join: how big is the remaining disagreement, and do the "
            "two move together from year to year?\n\n"
            "The check matters. A join that looks smooth can hide two records that "
            "disagree. The large shift here exists because the tide gauge series is "
            "measured from 1880 while the satellites start from their own first year."
        )


def render_trend_section(series: pd.DataFrame, units: str, modern_start: int) -> None:
    """Show rates over time, the straight-line versus curved fit and acceleration.

    Args:
        series: The stitched annual series.
        units: Display unit.
        modern_start: First year of the satellite record.
    """
    st.subheader("Is the rise speeding up?")
    window = st.slider(
        "Window length (years)",
        10,
        30,
        20,
        key="window",
        help="Each point on the chart is the trend over this many preceding years.",
    )
    rolling = analysis.rolling_trends(series, window)
    st.plotly_chart(
        charts.rolling(rolling, window, units, modern_start), width="stretch"
    )
    first, last = rolling.iloc[0], rolling.iloc[-1]
    st.write(
        f"Looking back {window} years from {int(first['end_year'])}, sea level was rising "
        f"about {rate(first['slope'], units)}. Looking back from {int(last['end_year'])} it is "
        f"{rate(last['slope'], units)}, around {last['slope'] / first['slope']:.1f} times as fast."
    )
    last_year = int(series["year"].max())
    eras = analysis.era_trends(
        series, [(1900, 1950), (1950, modern_start - 1), (modern_start, last_year)]
    )
    bounds = zip(eras["low"], eras["high"], strict=True)
    table = eras.assign(
        Rate=eras["slope"].map(lambda v: rate(v, units)),
        Interval=[
            f"{to_units(a, units):.2f} to {to_units(b, units):.2f}" for a, b in bounds
        ],
    )[["era", "Rate", "Interval", "years"]].rename(
        columns={
            "era": "Period",
            "years": "Years of data",
            "Interval": f"95% interval ({units} a year)",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    render_fit(series, units)


def render_fit(series: pd.DataFrame, units: str) -> None:
    """Compare a straight line with a curve and explain the interval.

    Args:
        series: The stitched annual series.
        units: Display unit.
    """
    line = analysis.trend(series["year"], series["level_mm"])
    curve = analysis.acceleration(series)
    st.markdown("**A straight line or a curve?**")
    st.plotly_chart(charts.fits(series, line, curve, units), width="stretch")
    start = curve.curve.iloc[1] - curve.curve.iloc[0]
    end = curve.curve.iloc[-1] - curve.curve.iloc[-2]
    cols = st.columns(3)
    cols[0].metric(
        f"Whole-record rate ({units} a year)",
        per_year(line.slope, units),
    )
    cols[1].metric(
        f"95% range ({units} a year)",
        f"{per_year(line.low, units)} to {per_year(line.high, units)}",
        help=f"A textbook formula would claim only plus or minus {to_units(line.naive_half_width, units):.2f}.",
    )
    cols[2].metric(
        f"Acceleration ({units} a year per year)",
        f"{to_units(curve.value, units):.3f}",
        help=f"95% range {to_units(curve.low, units):.3f} to {to_units(curve.high, units):.3f}.",
    )
    verdict = (
        "clearly above zero" if curve.low > 0 else "not clearly different from zero"
    )
    st.write(
        "A straight line misses the shape: the curve says the rate has gone from about "
        f"{rate(start, units)} to {rate(end, units)}, a steady speed-up of "
        f"{to_units(curve.value, units):.3f} {units} a year every year "
        f"(95% range {to_units(curve.low, units):.3f} to {to_units(curve.high, units):.3f}), "
        f"which is {verdict}."
    )
    with st.expander("Technique: why the honest range is wider than the naive one"):
        st.write(
            "A textbook trend line treats every year as an independent measurement. "
            "Sea level is smooth, so one year tells you a lot about the next "
            f"(here the leftovers from the line have a year-to-year correlation of {line.lag1:.2f}). "
            f"Ignoring that gives a range of only plus or minus {to_units(line.naive_half_width, units):.2f}.\n\n"
            f"Instead, the leftovers are cut into chunks of {line.block} consecutive years, "
            "shuffled to build a new plausible series, and the line is refitted. Doing that "
            "1,000 times and taking the middle 95% of the slopes gives plus or minus "
            f"{to_units(line.half_width, units):.2f}. This is a moving-block bootstrap: keeping "
            "the chunks intact keeps the memory in the data.\n\n"
            "Part of that memory is real (the ocean changes slowly) and part comes from the "
            "straight line being the wrong shape. Either way the wider range is the safer one to quote."
        )


def station_label(key: str) -> str:
    """Return a picker label such as ``"Newlyn, Cornwall (UK)"``."""
    station = sources.STATION_BY_KEY[key]
    return f"{station.name} ({station.region})"


def land_sentence(lm: dict[str, float], units: str) -> str:
    """Say in words what the gauge-versus-ocean gap suggests about the land.

    Args:
        lm: Output of ``analysis.land_motion``.
        units: Display unit.

    Returns:
        A sentence that claims movement only if the interval excludes zero.
    """
    span = f"{to_units(lm['gap_low'], units):+.2f} to {to_units(lm['gap_high'], units):+.2f}"
    if lm["gap_low"] > 0:
        return (
            f"The gauge records about {rate(lm['gap'], units)} more rise than the open ocean "
            f"(95% range {span}), which suggests the land here is sinking."
        )
    if lm["gap_high"] < 0:
        return (
            f"The gauge records about {rate(-lm['gap'], units)} less rise than the open ocean "
            f"(95% range {span}), which suggests the land here is rising."
        )
    return (
        f"The gauge's rise differs from the ocean's by {rate(lm['gap'], units)}, but the 95% "
        f"range ({span}) includes zero, so these records cannot show whether the land is moving."
    )


def render_station(sat_annual: pd.DataFrame | None, units: str) -> None:
    """Show one tide gauge against the global ocean.

    Args:
        sat_annual: Annual global satellite series, or None if unavailable.
        units: Display unit.
    """
    st.subheader("Your coast")
    keys = [s.key for s in sources.STATIONS]
    key = st.selectbox("Tide gauge", keys, format_func=station_label, key="station")
    station = sources.STATION_BY_KEY[key]
    monthly, error = data.load_station(key)
    if monthly is None:
        st.warning(error)
        return
    anomalies, baseline = analysis.baseline_anomalies(monthly)
    adjusted = analysis.deseasonalise(anomalies)
    annual = analysis.station_annual(adjusted, datetime.now(sources.UTC).year)
    fit = analysis.trend(annual["year"], annual["level_mm"])
    modern_start = (
        int(sat_annual["year"].min())
        if sat_annual is not None
        else DEFAULT_MODERN_START
    )
    modern = annual[annual["year"] >= modern_start]
    latest = monthly["date"].max()
    st.write(
        f"{station.name}: monthly records from {int(annual['year'].min())} to "
        f"{latest:%B %Y} ({age_text(latest)}), from {station.provider}. "
        f"Shown as change from {baseline}."
    )
    st.plotly_chart(
        charts.station(adjusted, fit, station.name, baseline, units), width="stretch"
    )
    _station_metrics(annual, fit, modern, sat_annual, modern_start, units)
    _station_technique()


def _station_metrics(
    annual: pd.DataFrame,
    fit: analysis.Trend,
    modern: pd.DataFrame,
    sat_annual: pd.DataFrame | None,
    modern_start: int,
    units: str,
) -> None:
    """Show a gauge's rates and what its gap from the global ocean suggests.

    Args:
        annual: The gauge's annual means.
        fit: Trend over the whole record.
        modern: The gauge's annual means from the start of the satellite era.
        sat_annual: Annual global satellite series, or None if unavailable.
        modern_start: First year of the satellite record.
        units: Display unit.
    """
    cols = st.columns(3)
    cols[0].metric(
        f"Whole record ({units} a year)",
        per_year(fit.slope, units),
        help=f"95% range plus or minus {to_units(fit.half_width, units):.2f}.",
    )
    if len(modern) >= 4:
        modern_fit = analysis.trend(modern["year"], modern["level_mm"])
        cols[1].metric(
            f"Since {modern_start} ({units} a year)", per_year(modern_fit.slope, units)
        )
    if sat_annual is None:
        return
    global_fit = analysis.trend(sat_annual["year"], sat_annual["level_mm"])
    cols[2].metric(
        f"Global ocean since {modern_start} ({units} a year)",
        per_year(global_fit.slope, units),
    )
    lm = analysis.land_motion(annual, sat_annual)
    if lm is None:
        st.caption(
            "This record overlaps the satellite era by too few years to compare with the open ocean."
        )
        return
    st.write(land_sentence(lm, units))
    st.plotly_chart(charts.gap(annual, sat_annual, lm, units), width="stretch")


def _station_technique() -> None:
    """Explain seasonal removal, baselines and the land-movement estimate."""
    with st.expander(
        "Techniques: removing the seasons, and what a gauge really measures"
    ):
        st.write(
            "**Seasons.** Monthly sea level rises and falls with the year. To see the "
            "long-term change, the straight-line trend is removed first, the average for "
            "each calendar month is found from what is left, and that seasonal pattern is "
            "subtracted from the original.\n\n"
            "**Baselines.** Gauges measure against local reference levels, so absolute numbers "
            "cannot be compared. Each record is shown as change from its own average over "
            "a fixed reference period (1993 to 2012) where it covers that period, otherwise "
            "its whole-record average.\n\n"
            "**Land movement.** A gauge measures the sea relative to the land it stands on. "
            "Still-rebounding land (northern Britain, after the last ice age) lowers the apparent "
            "rise, and sinking land (Galveston, for example) raises it. The trend of the gap between "
            "a gauge and the global ocean estimates that movement, but it also absorbs regional "
            "ocean differences and is noisy over short records, so the page only draws a conclusion "
            "when the range excludes zero."
        )


def render_extremes(daily: pd.DataFrame, units: str) -> None:
    """Show how often Newlyn's daily sea level crosses a fixed early threshold.

    Args:
        daily: Newlyn daily series.
        units: Display unit.
    """
    st.subheader("How often is the sea high? A century at Newlyn")
    baseline = (1915, 1944)
    counts, threshold = analysis.exceedance_days(daily, baseline)
    last = int(counts["year"].max())
    late = (last - LATE_YEARS + 1, last)
    base_mean = counts[counts["year"].between(*baseline)]["days_above"].mean()
    late_mean = counts[counts["year"].between(*late)]["days_above"].mean()
    shifts = analysis.percentile_shift(daily, baseline, late)
    st.write(
        f"Take the daily average sea level at Newlyn in {baseline[0]} to {baseline[1]} and find the "
        f"level that was exceeded on just 1 day in 100. That is {length(threshold, units)} on the gauge's own scale. "
        f"It was topped on about {base_mean:.1f} days a year then, and on about {late_mean:.1f} days a year in "
        f"{late[0]} to {late[1]}: {late_mean / base_mean:.0f} times as often."
    )
    st.plotly_chart(
        charts.extremes(counts, threshold, baseline, units), width="stretch"
    )
    spread = shifts["shift_mm"].max() - shifts["shift_mm"].min()
    if spread < 0.25 * shifts["shift_mm"].mean():
        reading = (
            "Every part of the distribution moved up by almost the same amount, about "
            f"{length(shifts['shift_mm'].mean(), units)}, so it looks like the whole sea shifted upwards "
            "rather than the highs becoming more extreme."
        )
    else:
        reading = "The high end shifted by a different amount from the middle, so more than the average changed."
    st.plotly_chart(charts.distribution(daily, baseline, late, units), width="stretch")
    st.write(reading)
    st.caption(
        f"This record ends in {last}: the Newlyn daily series available from UHSLC stops there."
    )
    with st.expander("Technique: counting extremes against a fixed line"):
        st.write(
            "The threshold is set once, from an early period, and never moved. Counting later days above "
            "the same line shows how often yesterday's rare event has become ordinary. "
            "A moving threshold would hide the change.\n\n"
            "**Limits.** These are daily averages, which smooth out the short peaks of storm surges and "
            "high tides, so this says how the baseline sea level has moved, not how high the worst waves "
            "or surges got. It is not a flood-risk estimate."
        )


def render_scenarios(series: pd.DataFrame, units: str, modern_start: int) -> None:
    """Show two simple extrapolations with their statistical bands.

    Args:
        series: The stitched annual series.
        units: Display unit.
        modern_start: First year of the satellite record.
    """
    st.subheader("Where could this go?")
    last_year = int(series["year"].max())
    horizon = st.slider(
        "Look ahead to", last_year + 5, 2100, min(last_year + 25, 2100), key="horizon"
    )
    paths = analysis.scenarios(series, modern_start, horizon)
    st.plotly_chart(charts.scenarios(series, paths, units), width="stretch")
    rows = []
    for name in ("Steady", "Speeding up"):
        part = paths[paths["scenario"] == name].set_index("year")
        rows.append(
            {
                "Model": name,
                f"Level in {horizon} (relative to 1880)": f"{length(part.loc[horizon, 'level_mm'], units)} "
                f"({to_units(part.loc[horizon, 'low'], units):.0f} to {to_units(part.loc[horizon, 'high'], units):.0f})",
                f"Rise from {last_year} to {horizon}": length(
                    part.loc[horizon, "level_mm"] - part.loc[last_year, "level_mm"],
                    units,
                    signed=True,
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info(
        "These are extrapolations of the past, not forecasts. 'Steady' continues the straight-line rate since "
        f"{modern_start}; 'Speeding up' continues the curve fitted to the whole record. The shaded bands "
        "show only the statistical uncertainty of each fit, and each line starts from its own fitted value, "
        "not from the last data point. Neither model knows about ice sheets, emissions or "
        f"policy. For physically based projections see the [IPCC Sixth Assessment Report, chapter 9]({IPCC_URL})."
    )


def render_live(units: str, modern_rate_mm: float | None) -> None:
    """Show the last two days of a live tide gauge and put the tide in context.

    Args:
        units: Display unit.
        modern_rate_mm: Recent global rate of rise in mm a year, or None.
    """
    st.subheader("Right now")
    stations, error = data.load_ea_stations()
    if not stations:
        st.warning(error)
        return
    labels = [s["label"] for s in stations]
    default = labels.index("Newlyn") if "Newlyn" in labels else 0
    choice = st.selectbox("Live tide gauge", labels, index=default, key="ea_station")
    measure = next(s["measure"] for s in stations if s["label"] == choice)
    readings, error = data.load_ea_readings(measure)
    if readings is None or readings.empty:
        st.warning(error or "No recent readings were returned for this gauge.")
        return
    high, low, span = analysis.tidal_range(readings)
    latest = readings["time"].max()
    st.plotly_chart(charts.tide(readings, choice), width="stretch")
    cols = st.columns(3)
    cols[0].metric("Highest, last 24 hours", f"{high:.2f} m")
    cols[1].metric("Lowest, last 24 hours", f"{low:.2f} m")
    cols[2].metric("Tidal range", f"{span:.2f} m")
    if modern_rate_mm:
        context = (
            f"Global sea level has recently been rising by about {rate(modern_rate_mm, units)}, so a "
            f"single day's tide is about {span * 1000 / modern_rate_mm:,.0f} times larger than a year's rise. "
        )
    else:
        context = "A year's rise in sea level is a few millimetres, thousands of times smaller than a day's tide. "
    st.write(
        f"The tide here moved {span:.1f} m in a day. {context}"
        "That is why scientists average over months and years to see it. Latest reading: "
        f"{latest:%d %B %Y, %H:%M} UTC. Readings are provisional and unvalidated."
    )
    st.caption(
        "This uses Environment Agency tide gauge data from the real-time data API (Beta). "
        "Open Government Licence v3.0."
    )


def render_limits(uk_lag: str) -> None:
    """State plainly what the page can and cannot tell you.

    Args:
        uk_lag: How far the UK gauge records are behind today, in words.
    """
    st.subheader("What this page can and cannot tell you")
    can, cannot = st.columns(2)
    can.markdown(
        "**Can do**\n\n"
        "- Join two records onto one scale and show how well they agree.\n"
        "- Give rates of rise with an honest uncertainty.\n"
        "- Show whether the rate is changing, over any window you choose.\n"
        "- Compare a coastal gauge with the global ocean.\n"
        "- Count how often a fixed high level is reached at Newlyn.\n"
        "- Carry on working when one source is down."
    )
    cannot.markdown(
        "**Cannot do**\n\n"
        "- **Forecast.** The scenarios extend past trends only.\n"
        "- **Explain why** sea level rises; warming water and melting ice are not separated.\n"
        "- **Estimate flood risk.** Monthly and daily averages hide tides and storm surges.\n"
        f"- **Show UK gauges up to today.** The long UK records are {uk_lag} behind; "
        "only the live panel is current, and it covers two days.\n"
        "- **Pin down land movement** precisely; the estimate is rough and often inconclusive.\n"
        "- **Treat live readings as checked.** They are provisional."
    )


def render_credits(release: str | None) -> None:
    """Credit every source, its role and its terms.

    Args:
        release: The Colorado release in use, if known.
    """
    st.subheader("Data and credits")
    st.write(
        "This page exists because these organisations publish their data openly. "
        "Please credit them if you reuse anything here."
    )
    for source in sources.credits(release=release):
        st.markdown(
            f"**[{source['name']}]({source['url']})**  \n"
            f"{source['role']}  \n"
            f"*Credit:* {source['credit']}  \n"
            f"*Terms:* {source['terms']}"
        )


def render_footer() -> None:
    """Show a one-line footer linking back to the Data Flakes site."""
    st.markdown(
        f"""
        <footer class="site-foot">
            Built by Elliott Fairhall. More projects and write-ups at
            <a href="{SITE_URL}/" target="_blank">data-flakes.dev</a>.
        </footer>
        """,
        unsafe_allow_html=True,
    )
