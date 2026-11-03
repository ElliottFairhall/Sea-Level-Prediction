"""Plotly figures for the sea level page, in the Data Flakes style."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.analysis import Acceleration, Trend, to_units

BLUE = "#5E8FBA"
DEEP = "#355C7D"
ROSE = "#F67280"
PLUM = "#C06C84"
PEACH = "#F8B195"
GREY = "#94A3B8"
FONT = {"family": "Inter, sans-serif", "size": 13}
GRID = "rgba(128,128,128,0.18)"


def _style(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"t": 8, "b": 8, "l": 8, "r": 8},
        font=FONT,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _unit_label(units: str) -> str:
    return "inches" if units == "inches" else "mm"


def _fmt(units: str) -> str:
    return "%{y:.2f}" if units == "inches" else "%{y:.0f}"


def global_record(
    epa: pd.DataFrame,
    satellite: pd.DataFrame,
    offset_mm: float,
    overlap: tuple[int, int],
    units: str,
) -> go.Figure:
    """The two global records on one scale, with the overlap shaded.

    Args:
        epa: Columns ``year``, ``level_mm``, ``lower_mm`` and ``upper_mm``.
        satellite: Columns ``date`` and ``level_mm`` (10-day, before alignment).
        offset_mm: Amount added to the satellite record to align it.
        overlap: First and last year the records share.
        units: ``"mm"`` or ``"inches"``.

    Returns:
        The figure.
    """
    cv = lambda s: s.map(lambda v: to_units(v, units))  # noqa: E731
    years = pd.to_datetime(epa["year"].astype(str) + "-07-01")
    fig = go.Figure()
    fig.add_scatter(
        x=years,
        y=cv(epa["upper_mm"]),
        mode="lines",
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(
        x=years,
        y=cv(epa["lower_mm"]),
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(94,143,186,0.2)",
        hoverinfo="skip",
        name="Reported uncertainty",
    )
    fig.add_scatter(
        x=years,
        y=cv(epa["level_mm"]),
        mode="lines",
        name="Tide gauge reconstruction",
        line={"color": BLUE, "width": 2.5},
        hovertemplate=_fmt(units),
    )
    fig.add_scatter(
        x=satellite["date"],
        y=cv(satellite["level_mm"] + offset_mm),
        mode="lines",
        name="Satellite altimetry (aligned)",
        line={"color": ROSE, "width": 2},
        hovertemplate=_fmt(units),
    )
    fig.add_vrect(
        x0=f"{overlap[0]}-01-01",
        x1=f"{overlap[1]}-12-31",
        fillcolor="rgba(128,128,128,0.12)",
        line_width=0,
        annotation_text="Both records",
        annotation_position="top left",
    )
    _style(fig, 420)
    fig.update_yaxes(title=f"Sea level relative to 1880 ({_unit_label(units)})")
    return fig


def fits(
    series: pd.DataFrame, line: Trend, curve: Acceleration, units: str
) -> go.Figure:
    """Annual points with a straight-line fit and a curved fit.

    Args:
        series: Annual ``year`` and ``level_mm``.
        line: The straight-line fit over the whole series.
        curve: The quadratic fit.
        units: ``"mm"`` or ``"inches"``.

    Returns:
        The figure.
    """
    cv = lambda v: to_units(v, units)  # noqa: E731
    fig = go.Figure()
    fig.add_scatter(
        x=series["year"],
        y=series["level_mm"].map(cv),
        mode="markers",
        name="Annual average",
        marker={"color": GREY, "size": 6},
        hovertemplate=_fmt(units),
    )
    fig.add_scatter(
        x=series["year"],
        y=[cv(line.at(y)) for y in series["year"]],
        mode="lines",
        name="Straight line",
        line={"color": PLUM, "width": 2, "dash": "dash"},
        hovertemplate=_fmt(units),
    )
    fig.add_scatter(
        x=curve.curve.index,
        y=curve.curve.map(cv),
        mode="lines",
        name="Curve",
        line={"color": ROSE, "width": 3},
        hovertemplate=_fmt(units),
    )
    _style(fig, 380)
    fig.update_yaxes(title=f"Sea level relative to 1880 ({_unit_label(units)})")
    return fig


def rolling(
    rt: pd.DataFrame, window: int, units: str, satellite_start: int
) -> go.Figure:
    """Rate of rise in a sliding window of years, with its interval.

    Args:
        rt: Columns ``end_year``, ``slope``, ``low`` and ``high`` (mm per year).
        window: Window length in years.
        units: ``"mm"`` or ``"inches"``.
        satellite_start: Year the satellite record begins, marked on the chart.

    Returns:
        The figure.
    """
    cv = lambda s: s.map(lambda v: to_units(v, units))  # noqa: E731
    fig = go.Figure()
    fig.add_scatter(
        x=rt["end_year"],
        y=cv(rt["high"]),
        mode="lines",
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(
        x=rt["end_year"],
        y=cv(rt["low"]),
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(246,114,128,0.2)",
        hoverinfo="skip",
        name="95% interval",
    )
    fig.add_scatter(
        x=rt["end_year"],
        y=cv(rt["slope"]),
        mode="lines",
        name=f"Rate over the previous {window} years",
        line={"color": ROSE, "width": 3},
        hovertemplate="%{y:.2f}",
    )
    fig.add_vline(
        x=satellite_start,
        line_dash="dot",
        line_color=GREY,
        annotation_text="Satellites begin",
    )
    _style(fig, 360)
    fig.update_xaxes(
        title=f"Window ends in year (each point looks back {window} years)"
    )
    fig.update_yaxes(title=f"Rate ({_unit_label(units)} per year)")
    return fig


def station(
    frame: pd.DataFrame, trend_line: Trend, name: str, baseline: str, units: str
) -> go.Figure:
    """A gauge's monthly values, 12-month smooth and long-term trend.

    Args:
        frame: Columns ``date``, ``anomaly_mm`` and ``adjusted_mm``.
        trend_line: Trend fitted to the annual means.
        name: Place name.
        baseline: Plain description of the baseline.
        units: ``"mm"`` or ``"inches"``.

    Returns:
        The figure.
    """
    cv = lambda s: s.map(lambda v: to_units(v, units))  # noqa: E731
    smooth = frame["adjusted_mm"].rolling(12, center=True, min_periods=9).mean()
    decimal = frame["date"].dt.year + frame["date"].dt.day_of_year / 365.25
    fig = go.Figure()
    fig.add_scatter(
        x=frame["date"],
        y=cv(frame["adjusted_mm"]),
        mode="markers",
        name="Monthly (seasonal cycle removed)",
        marker={"color": "rgba(94,143,186,0.35)", "size": 4},
        hovertemplate=_fmt(units),
    )
    fig.add_scatter(
        x=frame["date"],
        y=cv(smooth),
        mode="lines",
        name="12-month average",
        line={"color": BLUE, "width": 2.5},
        hovertemplate=_fmt(units),
    )
    fig.add_scatter(
        x=frame["date"],
        y=cv(trend_line.slope * decimal + trend_line.intercept),
        mode="lines",
        name="Long-term trend",
        line={"color": ROSE, "width": 2.5, "dash": "dash"},
        hovertemplate=_fmt(units),
    )
    _style(fig, 380)
    fig.update_yaxes(title=f"Change ({_unit_label(units)})")
    return fig


def gap(
    local: pd.DataFrame, global_: pd.DataFrame, rates: dict[str, float], units: str
) -> go.Figure:
    """Local annual sea level against the global ocean over the years both exist.

    Args:
        local: Annual ``year`` and ``level_mm`` for the gauge.
        global_: Annual ``year`` and ``level_mm`` for the global satellite record.
        rates: Output of ``analysis.land_motion``.
        units: ``"mm"`` or ``"inches"``.

    Returns:
        The figure.
    """
    both = local.merge(global_, on="year", suffixes=("_l", "_g"))
    cv = lambda s: s.map(lambda v: to_units(v, units))  # noqa: E731
    fig = go.Figure()
    fig.add_scatter(
        x=both["year"],
        y=cv(both["level_mm_l"] - both["level_mm_l"].mean()),
        mode="lines+markers",
        name="This gauge",
        line={"color": BLUE, "width": 2.5},
        hovertemplate=_fmt(units),
    )
    fig.add_scatter(
        x=both["year"],
        y=cv(both["level_mm_g"] - both["level_mm_g"].mean()),
        mode="lines+markers",
        name="Global ocean (satellites)",
        line={"color": ROSE, "width": 2.5},
        hovertemplate=_fmt(units),
    )
    _style(fig, 330)
    fig.update_yaxes(title=f"Change in annual average ({_unit_label(units)})")
    return fig


def extremes(
    counts: pd.DataFrame, threshold_mm: float, baseline: tuple[int, int], units: str
) -> go.Figure:
    """Days per year above a fixed early threshold.

    Args:
        counts: Columns ``year`` and ``days_above``.
        threshold_mm: The threshold in millimetres.
        baseline: First and last year that defined the threshold.
        units: ``"mm"`` or ``"inches"``.

    Returns:
        The figure.
    """
    base = counts[counts["year"].between(*baseline)]["days_above"].mean()
    colours = [GREY if y <= baseline[1] else ROSE for y in counts["year"]]
    fig = go.Figure(
        go.Bar(
            x=counts["year"],
            y=counts["days_above"],
            marker_color=colours,
            hovertemplate="%{y} days<extra></extra>",
        )
    )
    fig.add_hline(y=base, line_dash="dot", line_color=GREY)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        text=(
            f"Grey: the {baseline[0]} to {baseline[1]} baseline, "
            f"about {base:.1f} days a year (dotted line)"
        ),
    )
    _style(fig, 340)
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_yaxes(title="Days above the baseline's 99th percentile")
    fig.update_xaxes(title="Year")
    return fig


def distribution(
    daily: pd.DataFrame, early: tuple[int, int], late: tuple[int, int], units: str
) -> go.Figure:
    """Overlaid histograms of daily levels in two periods.

    Args:
        daily: Columns ``date`` and ``level_mm``.
        early: First and last year of the early period.
        late: First and last year of the late period.
        units: ``"mm"`` or ``"inches"``.

    Returns:
        The figure.
    """
    years = daily["date"].dt.year
    fig = go.Figure()
    for label, span, colour in (
        (f"{early[0]} to {early[1]}", early, GREY),
        (f"{late[0]} to {late[1]}", late, ROSE),
    ):
        values = daily[years.between(*span)]["level_mm"].map(
            lambda v: to_units(v, units)
        )
        fig.add_histogram(
            x=values,
            name=label,
            opacity=0.65,
            nbinsx=60,
            marker_color=colour,
            histnorm="percent",
        )
    _style(fig, 340)
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(
        title=f"Daily average sea level at Newlyn ({_unit_label(units)}, gauge datum)"
    )
    fig.update_yaxes(title="Share of days (%)")
    return fig


def scenarios(history: pd.DataFrame, paths: pd.DataFrame, units: str) -> go.Figure:
    """History and two simple projections with their statistical bands.

    Args:
        history: Annual ``year`` and ``level_mm``.
        paths: Output of ``analysis.scenarios``.
        units: ``"mm"`` or ``"inches"``.

    Returns:
        The figure.
    """
    cv = lambda s: s.map(lambda v: to_units(v, units))  # noqa: E731
    fig = go.Figure()
    fig.add_scatter(
        x=history["year"],
        y=cv(history["level_mm"]),
        mode="lines",
        name="Observed",
        line={"color": DEEP, "width": 2.5},
        hovertemplate=_fmt(units),
    )
    for name, colour, fill in (
        ("Steady", PLUM, "rgba(192,108,132,0.2)"),
        ("Speeding up", ROSE, "rgba(246,114,128,0.2)"),
    ):
        part = paths[paths["scenario"] == name]
        fig.add_scatter(
            x=part["year"],
            y=cv(part["high"]),
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_scatter(
            x=part["year"],
            y=cv(part["low"]),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=fill,
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_scatter(
            x=part["year"],
            y=cv(part["level_mm"]),
            mode="lines",
            name=name,
            line={"color": colour, "width": 2.5, "dash": "dash"},
            hovertemplate=_fmt(units),
        )
    _style(fig, 400)
    fig.update_yaxes(title=f"Sea level relative to 1880 ({_unit_label(units)})")
    return fig


def tide(readings: pd.DataFrame, label: str) -> go.Figure:
    """Live tide readings with the high and low marked.

    Args:
        readings: Columns ``time`` and ``level_m``.
        label: Station name.

    Returns:
        The figure.
    """
    fig = go.Figure()
    fig.add_scatter(
        x=readings["time"],
        y=readings["level_m"],
        mode="lines",
        name=label,
        line={"color": BLUE, "width": 2.5},
        hovertemplate="%{y:.2f} m",
    )
    hi, lo = (
        readings.loc[readings["level_m"].idxmax()],
        readings.loc[readings["level_m"].idxmin()],
    )
    fig.add_scatter(
        x=[hi["time"], lo["time"]],
        y=[hi["level_m"], lo["level_m"]],
        mode="markers",
        name="Highest and lowest",
        marker={"color": ROSE, "size": 10},
        hovertemplate="%{y:.2f} m",
    )
    _style(fig, 320)
    fig.update_yaxes(title="Metres above ordnance datum")
    fig.update_xaxes(title="Time (UTC)")
    return fig
