"""Analysis techniques for the sea level page.

Each function is small, pure and documented with the idea behind it, because the
page is also meant to teach how these records are handled:

* stitching two records that use different baselines
* trends with honest uncertainty (a moving-block bootstrap)
* rolling trends and acceleration
* removing the seasonal cycle from monthly gauge data
* comparing a local gauge with the global ocean to estimate land movement
* counting extreme days against a fixed baseline
* simple scenarios, with their statistical uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

MM_PER_INCH = 25.4
MIN_SATELLITE_CYCLES = 30
MIN_STATION_MONTHS = 9
MIN_OVERLAP_YEARS = 12
ALPHA = 0.05
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class Trend:
    """A straight-line fit with its uncertainty.

    Attributes:
        slope: Rate of change per year, in the units of the data.
        low: Lower end of the 95% interval for the slope.
        high: Upper end of the 95% interval for the slope.
        intercept: Fitted value at year 0.
        n: Number of points used.
        block: Length of the blocks of consecutive residuals that were
            resampled to build the interval.
        lag1: Lag-1 autocorrelation of the residuals.
        naive_half_width: Half-width of the textbook interval, which assumes
            every point is independent, for comparison.
    """

    slope: float
    low: float
    high: float
    intercept: float
    n: int
    block: int
    lag1: float
    naive_half_width: float

    @property
    def half_width(self) -> float:
        """Half the width of the 95% interval."""
        return (self.high - self.low) / 2

    def at(self, year: float) -> float:
        """Value of the fitted line at ``year``."""
        return self.slope * year + self.intercept


@dataclass(frozen=True)
class Stitch:
    """Two records joined into one.

    Attributes:
        series: Annual values with ``year``, ``level_mm`` and ``source`` columns.
        offset_mm: Amount added to the satellite record to match the long record.
        rmse_mm: Typical disagreement over the overlap after alignment.
        correlation: Correlation of the two records' year-to-year changes.
        overlap: First and last year of the overlap.
    """

    series: pd.DataFrame
    offset_mm: float
    rmse_mm: float
    correlation: float
    overlap: tuple[int, int]


@dataclass(frozen=True)
class Acceleration:
    """A quadratic fit's curvature.

    Attributes:
        value: Acceleration in mm per year per year.
        low: Lower end of the 95% interval.
        high: Upper end of the 95% interval.
        curve: Fitted values for each input year.
    """

    value: float
    low: float
    high: float
    curve: pd.Series


def load_epa(path: Path) -> pd.DataFrame:
    """Read the bundled EPA / CSIRO file and convert it to millimetres.

    Args:
        path: Location of ``epa-sea-level.csv`` (inches, relative to 1880).

    Returns:
        Columns ``year``, ``level_mm``, ``lower_mm`` and ``upper_mm``.
    """
    raw = pd.read_csv(path).sort_values("Year")
    return pd.DataFrame(
        {
            "year": raw["Year"].astype(int),
            "level_mm": raw["CSIRO Adjusted Sea Level"] * MM_PER_INCH,
            "lower_mm": raw["Lower Error Bound"] * MM_PER_INCH,
            "upper_mm": raw["Upper Error Bound"] * MM_PER_INCH,
        }
    ).reset_index(drop=True)


def annual_means(
    frame: pd.DataFrame, min_count: int, last_complete_year: int | None = None
) -> pd.DataFrame:
    """Average a dated series by calendar year, dropping thin years.

    Args:
        frame: Columns ``date`` and ``level_mm``.
        min_count: Fewest readings a year needs to be kept.
        last_complete_year: Drop later years (the current, partial one).

    Returns:
        Columns ``year`` and ``level_mm``.
    """
    grouped = frame.groupby(frame["date"].dt.year)["level_mm"].agg(["mean", "count"])
    kept = grouped[grouped["count"] >= min_count]
    if last_complete_year is not None:
        kept = kept[kept.index <= last_complete_year]
    return pd.DataFrame(
        {"year": kept.index.astype(int), "level_mm": kept["mean"].to_numpy()}
    )


def stitch(long_record: pd.DataFrame, satellite: pd.DataFrame) -> Stitch:
    """Join a long record and a satellite record into one annual series.

    The two measure the same quantity but start from different baselines (the
    long record from 1880, the satellite record from 1993). Where they overlap,
    the average gap between them is the offset; adding it to the satellite
    record puts both on one scale. The leftover disagreement tells you how far
    to trust the join.

    Args:
        long_record: Annual ``year`` and ``level_mm`` from the long record.
        satellite: Annual ``year`` and ``level_mm`` from the satellite record.

    Returns:
        The stitched series and the checks on the join.

    Raises:
        ValueError: If the records overlap in fewer than three years.
    """
    both = long_record.merge(satellite, on="year", suffixes=("_long", "_sat"))
    if len(both) < 3:
        raise ValueError("the records need at least three overlapping years")
    offset = float((both["level_mm_long"] - both["level_mm_sat"]).mean())
    residual = both["level_mm_long"] - (both["level_mm_sat"] + offset)
    rmse = float(np.sqrt((residual**2).mean()))
    long_steps = both["level_mm_long"].diff().dropna().to_numpy()
    sat_steps = both["level_mm_sat"].diff().dropna().to_numpy()
    if np.allclose(long_steps, sat_steps):
        correlation = 1.0
    elif long_steps.std() == 0 or sat_steps.std() == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(long_steps, sat_steps)[0, 1])
    first_sat = int(satellite["year"].min())
    older = long_record[long_record["year"] < first_sat].assign(
        source="Tide gauge reconstruction"
    )
    newer = satellite.assign(
        level_mm=satellite["level_mm"] + offset, source="Satellite"
    )
    series = pd.concat(
        [older, newer[["year", "level_mm", "source"]]], ignore_index=True
    )
    return Stitch(
        series=series,
        offset_mm=offset,
        rmse_mm=rmse,
        correlation=correlation,
        overlap=(int(both["year"].min()), int(both["year"].max())),
    )


def _lag1(residuals: np.ndarray) -> float:
    """Lag-1 autocorrelation of a series, floored at zero."""
    if len(residuals) < 4 or np.allclose(residuals.std(), 0):
        return 0.0
    r = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    return float(max(r, 0.0)) if np.isfinite(r) else 0.0


def _block_length(n: int) -> int:
    """Block length for the bootstrap, growing slowly with the series length."""
    return max(2, int(round(2 * n ** (1 / 3))))


def bootstrap_fit(
    x: np.ndarray, y: np.ndarray, degree: int, draws: int = BOOTSTRAP_DRAWS
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Fit a polynomial and bootstrap its coefficients in blocks.

    The usual error formulas assume every point is independent. Sea level is
    smooth, so neighbouring points repeat each other. A moving-block bootstrap
    keeps that memory: it cuts the leftover wiggles (the residuals) into
    chunks of consecutive years, shuffles the chunks to build new plausible
    series, refits each one, and reads the uncertainty off the spread.

    Args:
        x: Years.
        y: Values.
        degree: Polynomial degree (1 for a line, 2 for a curve).
        draws: Number of bootstrap series.

    Returns:
        The fitted coefficients (highest power first, years centred on their
        mean), the bootstrap coefficients (``draws`` by ``degree + 1``), the
        mean year used to centre, and the block length.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centre = float(x.mean())
    design = np.vander(x - centre, degree + 1)
    pinv = np.linalg.pinv(design)
    coef = pinv @ y
    fitted = design @ coef
    residuals = y - fitted
    n, block = len(y), _block_length(len(y))
    blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    starts = rng.integers(0, n - block + 1, size=(draws, blocks))
    index = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]
    simulated = fitted[None, :] + residuals[index]
    return coef, simulated @ pinv.T, centre, block


def _interval(samples: np.ndarray) -> tuple[float, float]:
    """The middle 95% of a set of bootstrap values."""
    low, high = np.percentile(samples, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def trend(x: ArrayLike, y: ArrayLike) -> Trend:
    """Fit a straight line and give an honest 95% interval for the slope.

    Args:
        x: Years.
        y: Values.

    Returns:
        The fit. ``naive_half_width`` is what the textbook formula would claim.

    Raises:
        ValueError: If fewer than four points are given.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4:
        raise ValueError("a trend needs at least four points")
    coef, boots, centre, block = bootstrap_fit(x, y, 1)
    low, high = _interval(boots[:, 0])
    ols = stats.linregress(x, y)
    residuals = y - (coef[0] * (x - centre) + coef[1])
    t_naive = stats.t.ppf(1 - ALPHA / 2, len(x) - 2)
    return Trend(
        slope=float(coef[0]),
        low=low,
        high=high,
        intercept=float(coef[1] - coef[0] * centre),
        n=len(x),
        block=block,
        lag1=_lag1(residuals),
        naive_half_width=float(t_naive * ols.stderr),
    )


def rolling_trends(series: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Trend over a sliding window of years, to show whether the rate changes.

    Args:
        series: Annual ``year`` and ``level_mm``.
        window: Years in each window.

    Returns:
        Columns ``end_year``, ``slope``, ``low`` and ``high`` (mm per year).
    """
    rows = []
    years = series["year"].to_numpy()
    values = series["level_mm"].to_numpy()
    for end in range(window - 1, len(series)):
        sl = slice(end - window + 1, end + 1)
        if years[end] - years[end - window + 1] != window - 1:
            continue
        fit = trend(years[sl], values[sl])
        rows.append((int(years[end]), fit.slope, fit.low, fit.high))
    return pd.DataFrame(rows, columns=["end_year", "slope", "low", "high"])


def era_trends(series: pd.DataFrame, eras: list[tuple[int, int]]) -> pd.DataFrame:
    """Trends over named stretches of years.

    Args:
        series: Annual ``year`` and ``level_mm``.
        eras: ``(first_year, last_year)`` pairs, inclusive.

    Returns:
        Columns ``era``, ``slope``, ``low``, ``high`` and ``years``.
    """
    rows = []
    for first, last in eras:
        part = series[series["year"].between(first, last)]
        if len(part) >= 4:
            fit = trend(part["year"], part["level_mm"])
            rows.append((f"{first} to {last}", fit.slope, fit.low, fit.high, len(part)))
    return pd.DataFrame(rows, columns=["era", "slope", "low", "high", "years"])


def acceleration(series: pd.DataFrame) -> Acceleration:
    """Fit a curve to see whether the rate of rise is itself changing.

    A quadratic ``a + b t + c t^2`` has a rate that changes steadily, at ``2c``
    millimetres per year per year. The interval comes from the same
    moving-block bootstrap as :func:`trend`.

    Args:
        series: Annual ``year`` and ``level_mm``.

    Returns:
        The acceleration, its 95% interval and the fitted curve.
    """
    t = series["year"].to_numpy(dtype=float)
    coef, boots, centre, _ = bootstrap_fit(t, series["level_mm"].to_numpy(), 2)
    low, high = _interval(2 * boots[:, 0])
    fitted = np.vander(t - centre, 3) @ coef
    return Acceleration(
        value=float(2 * coef[0]),
        low=low,
        high=high,
        curve=pd.Series(fitted, index=series["year"].to_numpy()),
    )


def baseline_anomalies(
    frame: pd.DataFrame, first: int = 1993, last: int = 2012
) -> tuple[pd.DataFrame, str]:
    """Express a gauge record relative to a common reference period.

    Gauges report levels against different local datums, so absolute numbers
    cannot be compared. Subtracting each record's own average over a reference
    period makes changes comparable. If the gauge does not cover that period
    well, its whole-record average is used instead.

    Args:
        frame: Columns ``date`` and ``level_mm``.
        first: First year of the reference period.
        last: Last year of the reference period.

    Returns:
        The frame with an added ``anomaly_mm`` column, and a plain description
        of the baseline used.
    """
    period = frame[frame["date"].dt.year.between(first, last)]
    if len(period) >= 0.8 * 12 * (last - first + 1):
        base, label = period["level_mm"].mean(), f"its {first} to {last} average"
    else:
        base, label = frame["level_mm"].mean(), "its whole-record average"
    return frame.assign(anomaly_mm=frame["level_mm"] - base), label


def deseasonalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the repeating yearly cycle from a monthly series.

    A long-term trend would leak into month averages, so the straight-line trend
    is taken out first, the average for each calendar month is found from what
    is left, and that seasonal pattern is subtracted from the original.

    Args:
        frame: Columns ``date`` and ``anomaly_mm``.

    Returns:
        The frame with ``seasonal_mm`` and ``adjusted_mm`` columns added.
    """
    years = frame["date"].dt.year + (frame["date"].dt.day_of_year / 365.25)
    line = np.polyval(np.polyfit(years, frame["anomaly_mm"], 1), years)
    detrended = frame["anomaly_mm"] - line
    seasonal = detrended.groupby(frame["date"].dt.month).transform("mean")
    return frame.assign(
        seasonal_mm=seasonal, adjusted_mm=frame["anomaly_mm"] - seasonal
    )


def station_annual(frame: pd.DataFrame, last_complete_year: int) -> pd.DataFrame:
    """Annual means of a deseasonalised gauge, requiring most months present.

    Args:
        frame: Columns ``date`` and ``adjusted_mm``.
        last_complete_year: Drop later years.

    Returns:
        Columns ``year`` and ``level_mm``.
    """
    tidy = frame[["date", "adjusted_mm"]].rename(columns={"adjusted_mm": "level_mm"})
    return annual_means(tidy, MIN_STATION_MONTHS, last_complete_year)


def land_motion(
    local_annual: pd.DataFrame, global_annual: pd.DataFrame
) -> dict[str, float] | None:
    """Estimate how the land is moving, from the gap between gauge and ocean.

    A gauge measures the sea relative to the land it stands on. If the land sinks
    the gauge sees more rise than the open ocean; if it rises, less. The trend of
    (gauge minus global ocean) over the years both exist is therefore a rough
    estimate of vertical land motion. It also absorbs regional ocean differences,
    so treat it as indicative.

    Args:
        local_annual: Annual ``year`` and ``level_mm`` for the gauge.
        global_annual: Annual ``year`` and ``level_mm`` for the global satellite
            record.

    Returns:
        ``local``, ``global_`` and ``gap`` rates with the gap's interval (mm per
        year), or None if fewer than ``MIN_OVERLAP_YEARS`` years overlap.
    """
    both = local_annual.merge(global_annual, on="year", suffixes=("_local", "_global"))
    if len(both) < MIN_OVERLAP_YEARS:
        return None
    local = trend(both["year"], both["level_mm_local"])
    glob = trend(both["year"], both["level_mm_global"])
    gap = trend(both["year"], both["level_mm_local"] - both["level_mm_global"])
    return {
        "years": float(len(both)),
        "first": float(both["year"].min()),
        "last": float(both["year"].max()),
        "local": local.slope,
        "global_": glob.slope,
        "gap": gap.slope,
        "gap_low": gap.low,
        "gap_high": gap.high,
    }


def exceedance_days(
    daily: pd.DataFrame,
    baseline: tuple[int, int] = (1915, 1944),
    percentile: float = 99.0,
    min_days: int = 300,
) -> tuple[pd.DataFrame, float]:
    """Count unusually high days each year against a fixed early threshold.

    The threshold is the given percentile of the baseline years, so by
    construction about ``100 - percentile`` percent of baseline days exceed it.
    Counting later days above the same line shows how often yesterday's rare
    high becomes ordinary.

    Args:
        daily: Columns ``date`` and ``level_mm``.
        baseline: First and last year defining the threshold.
        percentile: Percentile of baseline days used as the threshold.
        min_days: Fewest valid days a year needs to be counted.

    Returns:
        A frame of ``year``, ``days_above`` and ``valid_days`` (one row per
        sufficiently complete year) and the threshold in millimetres.
    """
    years = daily["date"].dt.year
    base = daily[years.between(*baseline)]["level_mm"]
    threshold = float(np.percentile(base, percentile))
    flagged = daily.assign(above=daily["level_mm"] > threshold, year=years)
    counts = flagged.groupby("year").agg(
        days_above=("above", "sum"), valid_days=("above", "size")
    )
    counts = counts[counts["valid_days"] >= min_days].reset_index()
    return counts, threshold


def percentile_shift(
    daily: pd.DataFrame,
    early: tuple[int, int],
    late: tuple[int, int],
    levels: tuple[int, ...] = (5, 50, 95, 99),
) -> pd.DataFrame:
    """Compare chosen percentiles between two periods.

    Args:
        daily: Columns ``date`` and ``level_mm``.
        early: First and last year of the early period.
        late: First and last year of the late period.
        levels: Percentiles to compare.

    Returns:
        Columns ``percentile``, ``early_mm``, ``late_mm`` and ``shift_mm``.
    """
    years = daily["date"].dt.year
    a = daily[years.between(*early)]["level_mm"]
    b = daily[years.between(*late)]["level_mm"]
    rows = [(q, float(np.percentile(a, q)), float(np.percentile(b, q))) for q in levels]
    out = pd.DataFrame(rows, columns=["percentile", "early_mm", "late_mm"])
    out["shift_mm"] = out["late_mm"] - out["early_mm"]
    return out


def _projection(
    x: np.ndarray, y: np.ndarray, degree: int, future: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central path and 95% band of a polynomial fit extended to future years."""
    coef, boots, centre, _ = bootstrap_fit(x, y, degree)
    design = np.vander(future - centre, degree + 1)
    central = design @ coef
    paths = design @ boots.T
    low, high = np.percentile(paths, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], axis=1)
    return central, low, high


def scenarios(series: pd.DataFrame, modern_start: int, to_year: int) -> pd.DataFrame:
    """Extend two simple models to a future year, with statistical bands.

    "Steady" continues the straight-line rate since ``modern_start``.
    "Speeding up" continues the curve fitted to the whole record. The bands show
    only the statistical uncertainty of each fit, not what the climate will do.

    Args:
        series: Annual ``year`` and ``level_mm``.
        modern_start: First year used for the straight-line model.
        to_year: Last year to project to.

    Returns:
        Columns ``year``, ``scenario``, ``level_mm``, ``low`` and ``high`` for
        every year from the last observed year to ``to_year``.
    """
    last = int(series["year"].max())
    future = np.arange(last, to_year + 1, dtype=float)
    recent = series[series["year"] >= modern_start]
    models = (
        (
            "Steady",
            recent["year"].to_numpy(dtype=float),
            recent["level_mm"].to_numpy(),
            1,
        ),
        (
            "Speeding up",
            series["year"].to_numpy(dtype=float),
            series["level_mm"].to_numpy(),
            2,
        ),
    )
    frames = []
    for name, x, y, degree in models:
        central, low, high = _projection(x, y, degree, future)
        frames.append(
            pd.DataFrame(
                {
                    "year": future,
                    "scenario": name,
                    "level_mm": central,
                    "low": low,
                    "high": high,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def tidal_range(readings: pd.DataFrame, hours: int = 24) -> tuple[float, float, float]:
    """Highest, lowest and range of tide readings over the latest hours.

    Args:
        readings: Columns ``time`` and ``level_m``.
        hours: Length of the window ending at the latest reading.

    Returns:
        ``(high_m, low_m, range_m)``.

    Raises:
        ValueError: If there are no readings.
    """
    if readings.empty:
        raise ValueError("no readings")
    window = readings[
        readings["time"] >= readings["time"].max() - pd.Timedelta(hours=hours)
    ]
    high, low = float(window["level_m"].max()), float(window["level_m"].min())
    return high, low, high - low


def to_units(mm: float, units: str) -> float:
    """Convert millimetres to the chosen display unit.

    Args:
        mm: A length in millimetres.
        units: ``"mm"`` or ``"inches"``.

    Returns:
        The length in that unit.
    """
    return mm / MM_PER_INCH if units == "inches" else mm
