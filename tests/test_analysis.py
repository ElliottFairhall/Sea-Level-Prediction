"""Tests for the analysis techniques, using data with known answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import analysis

EPA_FILE = (
    Path(__file__).resolve().parent.parent / "assets" / "data" / "epa-sea-level.csv"
)


def annual(
    first: int,
    last: int,
    slope: float,
    base: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """An annual series with a known slope and optional noise."""
    years = np.arange(first, last + 1)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "year": years,
            "level_mm": base
            + slope * (years - first)
            + rng.normal(0, noise, len(years)),
        }
    )


def ar1(n: int, phi: float, sigma: float, seed: int) -> np.ndarray:
    """Autocorrelated noise: each value remembers the one before."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.normal(0, sigma)
    return out


def test_epa_file_is_converted_from_inches_to_millimetres() -> None:
    """The bundled file is in inches; the page works in millimetres."""
    epa = analysis.load_epa(EPA_FILE)
    raw = pd.read_csv(EPA_FILE)
    assert epa["year"].iloc[0] == 1880
    assert epa["level_mm"].iloc[-1] == pytest.approx(
        raw["CSIRO Adjusted Sea Level"].iloc[-1] * 25.4
    )
    assert epa["year"].max() == 2013


def test_annual_means_drop_thin_years_and_the_partial_current_year() -> None:
    """A year with too few readings is not averaged."""
    dates = list(pd.date_range("2020-01-01", periods=40, freq="10D")) + list(
        pd.date_range("2021-01-01", periods=5, freq="10D")
    )
    frame = pd.DataFrame({"date": dates, "level_mm": 1.0})
    out = analysis.annual_means(frame, min_count=30)
    assert out["year"].tolist() == [2020]


def test_stitching_recovers_a_known_offset() -> None:
    """If the satellites are the long record minus 100 mm, the shift found is 100."""
    long_record = annual(1880, 2013, 1.7)
    satellite = annual(1993, 2025, 1.7, base=1.7 * 113 - 100)
    satellite["level_mm"] = [1.7 * (y - 1880) - 100 for y in satellite["year"]]
    joined = analysis.stitch(long_record, satellite)
    assert joined.offset_mm == pytest.approx(100.0)
    assert joined.rmse_mm == pytest.approx(0.0, abs=1e-9)
    assert joined.overlap == (1993, 2013)
    assert joined.series["year"].is_monotonic_increasing
    assert joined.series["year"].is_unique
    assert joined.series["year"].min() == 1880
    assert joined.series["year"].max() == 2025


def test_stitched_series_has_no_jump_when_the_records_agree() -> None:
    """After alignment the year-to-year steps stay smooth across the join."""
    long_record = annual(1880, 2013, 2.0)
    satellite = pd.DataFrame(
        {
            "year": range(1993, 2026),
            "level_mm": [2.0 * (y - 1880) - 50 for y in range(1993, 2026)],
        }
    )
    steps = analysis.stitch(long_record, satellite).series["level_mm"].diff().dropna()
    assert steps.max() == pytest.approx(2.0)
    assert steps.min() == pytest.approx(2.0)


def test_stitching_needs_an_overlap() -> None:
    """Two records that barely overlap cannot be joined reliably."""
    with pytest.raises(ValueError, match="overlapping"):
        analysis.stitch(annual(1880, 1995, 1.0), annual(1994, 2025, 1.0))


def test_trend_recovers_the_true_slope() -> None:
    """A noisy line's slope is found and the interval contains the truth."""
    series = annual(1900, 2000, 1.5, noise=3.0)
    fit = analysis.trend(series["year"], series["level_mm"])
    assert fit.slope == pytest.approx(1.5, abs=0.1)
    assert fit.low < 1.5 < fit.high
    assert fit.n == 101


def test_trend_is_repeatable() -> None:
    """The bootstrap uses a fixed seed so the same data gives the same interval."""
    series = annual(1900, 1990, 1.0, noise=4.0)
    first = analysis.trend(series["year"], series["level_mm"])
    second = analysis.trend(series["year"], series["level_mm"])
    assert (first.low, first.high) == (second.low, second.high)


def test_trend_needs_enough_points() -> None:
    """Three points cannot support a trend."""
    with pytest.raises(ValueError, match="four points"):
        analysis.trend([1, 2, 3], [1, 2, 3])


def test_honest_interval_is_wider_than_naive_for_autocorrelated_noise() -> None:
    """Remembered noise makes the textbook interval too narrow."""
    years = np.arange(1900, 2000)
    y = 1.0 * (years - 1900) + ar1(len(years), phi=0.9, sigma=2.0, seed=4)
    fit = analysis.trend(years, y)
    assert fit.lag1 > 0.5
    assert fit.half_width > 1.5 * fit.naive_half_width


def test_honest_interval_covers_the_truth_most_of_the_time() -> None:
    """Over many simulated records the 95% interval catches the true slope often enough."""
    hits = 0
    runs = 60
    for seed in range(runs):
        years = np.arange(1950, 2000)
        y = 2.0 * (years - 1950) + ar1(len(years), phi=0.6, sigma=3.0, seed=seed)
        fit = analysis.trend(years, y)
        hits += fit.low <= 2.0 <= fit.high
    assert hits / runs >= 0.80


def test_rolling_trend_rises_when_the_series_accelerates() -> None:
    """On a curved series each window's rate is larger than the one before."""
    years = np.arange(1900, 2000)
    series = pd.DataFrame(
        {"year": years, "level_mm": 0.5 * (years - 1900) + 0.02 * (years - 1900) ** 2}
    )
    rolling = analysis.rolling_trends(series, 20)
    assert rolling["slope"].is_monotonic_increasing
    assert rolling["end_year"].iloc[0] == 1919
    assert len(rolling) == 100 - 20 + 1


def test_rolling_trend_skips_windows_with_gaps() -> None:
    """A missing year means a window does not span the stated number of years."""
    series = annual(1900, 1960, 1.0)
    series = series[series["year"] != 1930]
    rolling = analysis.rolling_trends(series, 20)
    assert not rolling["end_year"].between(1930, 1949).any()


def test_era_trends_report_each_period() -> None:
    """Three eras with different slopes are each recovered."""
    years = np.arange(1900, 2021)
    level = np.where(years < 1950, 1.0 * (years - 1900), 50 + 2.0 * (years - 1950))
    level = np.where(years >= 1990, 50 + 80 + 4.0 * (years - 1990), level)
    series = pd.DataFrame({"year": years, "level_mm": level.astype(float)})
    eras = analysis.era_trends(series, [(1900, 1949), (1950, 1989), (1990, 2020)])
    assert eras["slope"].round(2).tolist() == [1.0, 2.0, 4.0]


def test_acceleration_recovers_a_known_curve() -> None:
    """Level = 0.5 t^2 per year squared has acceleration 1.0."""
    years = np.arange(1900, 2000)
    series = pd.DataFrame(
        {"year": years, "level_mm": 0.5 * (years - 1950.0) ** 2 + 3 * (years - 1950)}
    )
    acc = analysis.acceleration(series)
    assert acc.value == pytest.approx(1.0, abs=1e-6)
    assert acc.low <= 1.0 <= acc.high


def test_acceleration_is_not_claimed_for_a_straight_line() -> None:
    """A noisy straight line has an interval that includes zero acceleration."""
    series = annual(1900, 2000, 2.0, noise=4.0, seed=11)
    acc = analysis.acceleration(series)
    assert acc.low <= 0 <= acc.high


def test_baseline_anomalies_use_the_reference_period_when_covered() -> None:
    """Covered gauges are compared with their 1993 to 2012 average."""
    dates = pd.date_range("1980-01-15", "2020-12-15", freq="MS") + pd.Timedelta(days=14)
    frame = pd.DataFrame({"date": dates, "level_mm": 7000.0})
    out, label = analysis.baseline_anomalies(frame)
    assert "1993 to 2012" in label
    assert out["anomaly_mm"].abs().max() == pytest.approx(0.0)


def test_baseline_anomalies_fall_back_for_short_records() -> None:
    """A gauge with no data in the reference period uses its whole-record average."""
    dates = pd.date_range("1920-01-15", "1950-12-15", freq="MS")
    frame = pd.DataFrame(
        {"date": dates, "level_mm": np.linspace(7000, 7100, len(dates))}
    )
    out, label = analysis.baseline_anomalies(frame)
    assert "whole-record" in label
    assert out["anomaly_mm"].mean() == pytest.approx(0.0, abs=1e-9)


def test_deseasonalising_removes_the_yearly_cycle_and_keeps_the_trend() -> None:
    """A seasonal wave on a rising line comes out as the line."""
    dates = pd.date_range("1960-01-15", "2010-12-15", freq="MS")
    t = np.arange(len(dates)) / 12
    seasonal = 80 * np.sin(2 * np.pi * (dates.month - 1) / 12)
    frame = pd.DataFrame({"date": dates, "anomaly_mm": 3.0 * t + seasonal})
    out = analysis.deseasonalise(frame)
    residual = out["adjusted_mm"] - 3.0 * t
    assert residual.std() < 5.0
    fitted = np.polyfit(t, out["adjusted_mm"], 1)[0]
    assert fitted == pytest.approx(3.0, abs=0.05)


def test_land_motion_finds_a_gauge_that_rises_faster_than_the_ocean() -> None:
    """A gauge gaining 1.5 mm a year on the ocean suggests the land is sinking."""
    ocean = annual(1993, 2025, 3.0, noise=0.5, seed=1)
    gauge = ocean.assign(level_mm=ocean["level_mm"] + 1.5 * (ocean["year"] - 1993))
    result = analysis.land_motion(gauge, ocean)
    assert result is not None
    assert result["gap"] == pytest.approx(1.5, abs=0.01)
    assert result["gap_low"] > 0


def test_land_motion_is_inconclusive_for_a_gauge_that_tracks_the_ocean() -> None:
    """If gauge and ocean move together the gap interval includes zero."""
    ocean = annual(1993, 2025, 3.0, noise=1.0, seed=2)
    gauge = ocean.assign(
        level_mm=ocean["level_mm"] + np.random.default_rng(3).normal(0, 4, len(ocean))
    )
    result = analysis.land_motion(gauge, ocean)
    assert result is not None
    assert result["gap_low"] <= 0 <= result["gap_high"]


def test_land_motion_needs_enough_overlap() -> None:
    """Fewer than a dozen shared years gives no estimate."""
    assert (
        analysis.land_motion(annual(2010, 2020, 3.0), annual(1993, 2025, 3.0)) is None
    )


def test_exceedance_baseline_matches_the_percentile() -> None:
    """By construction about 1% of baseline days exceed the 99th percentile."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("1915-01-01", "1990-12-31")
    daily = pd.DataFrame(
        {"date": dates, "level_mm": 3000 + rng.normal(0, 100, len(dates))}
    )
    counts, threshold = analysis.exceedance_days(daily, (1915, 1944))
    early = counts[counts["year"].between(1915, 1944)]["days_above"].mean()
    assert early == pytest.approx(3.65, abs=1.0)
    assert threshold == pytest.approx(3000 + 2.33 * 100, abs=15)


def test_exceedances_increase_when_the_level_rises() -> None:
    """Lifting later years by one standard deviation makes the old extreme routine."""
    rng = np.random.default_rng(6)
    dates = pd.date_range("1915-01-01", "1990-12-31")
    level = 3000 + rng.normal(0, 100, len(dates)) + np.where(dates.year >= 1960, 100, 0)
    counts, _ = analysis.exceedance_days(
        pd.DataFrame({"date": dates, "level_mm": level}), (1915, 1944)
    )
    assert (
        counts[counts["year"] >= 1960]["days_above"].mean()
        > 5 * counts[counts["year"] <= 1944]["days_above"].mean()
    )


def test_exceedance_ignores_years_with_too_few_days() -> None:
    """A year with a handful of readings is not counted."""
    dates = list(pd.date_range("1915-01-01", "1944-12-31")) + list(
        pd.date_range("1950-01-01", periods=20)
    )
    counts, _ = analysis.exceedance_days(
        pd.DataFrame({"date": dates, "level_mm": 3000.0}), (1915, 1944)
    )
    assert 1950 not in counts["year"].tolist()


def test_percentile_shift_reports_a_uniform_lift() -> None:
    """Moving every value up by 100 mm shifts every percentile by 100 mm."""
    rng = np.random.default_rng(8)
    dates = pd.date_range("1915-01-01", "2016-12-31")
    base = rng.normal(3000, 100, len(dates))
    daily = pd.DataFrame(
        {"date": dates, "level_mm": np.where(dates.year >= 1990, base + 100, base)}
    )
    shift = analysis.percentile_shift(daily, (1915, 1944), (1990, 2016))
    assert shift["shift_mm"].between(80, 120).all()
    assert shift["percentile"].tolist() == [5, 50, 95, 99]


def test_scenarios_extend_from_the_last_year_with_a_band() -> None:
    """Each model runs from the last observed year to the chosen year, band around the line."""
    series = annual(1880, 2025, 2.0, noise=2.0, seed=9)
    paths = analysis.scenarios(series, 1993, 2060)
    assert set(paths["scenario"]) == {"Steady", "Speeding up"}
    assert paths["year"].min() == 2025
    assert paths["year"].max() == 2060
    assert (paths["low"] <= paths["level_mm"]).all()
    assert (paths["level_mm"] <= paths["high"]).all()


def test_both_scenarios_agree_when_the_rate_never_changed() -> None:
    """On a straight line the curve has nothing to add, so the models match."""
    series = annual(1880, 2025, 2.0, noise=0.1, seed=10)
    paths = analysis.scenarios(series, 1993, 2050)
    final = paths[paths["year"] == 2050].set_index("scenario")["level_mm"]
    assert final["Steady"] == pytest.approx(final["Speeding up"], abs=5.0)


def test_scenarios_diverge_when_the_series_is_curved() -> None:
    """A speeding-up series projects higher under the curve than under the straight line."""
    years = np.arange(1880, 2026)
    series = pd.DataFrame(
        {"year": years, "level_mm": 1.0 * (years - 1880) + 0.02 * (years - 1880) ** 2}
    )
    paths = analysis.scenarios(series, 1993, 2100)
    final = paths[paths["year"] == 2100].set_index("scenario")["level_mm"]
    assert final["Speeding up"] > final["Steady"] + 20


def test_tidal_range_over_the_latest_day() -> None:
    """A tide of amplitude 1.5 m has a range of about 3 m."""
    times = pd.date_range("2026-11-01", periods=48 * 4, freq="15min", tz="UTC")
    readings = pd.DataFrame(
        {"time": times, "level_m": 1.5 * np.sin(2 * np.pi * np.arange(len(times)) / 50)}
    )
    high, low, span = analysis.tidal_range(readings)
    assert span == pytest.approx(3.0, abs=0.05)
    assert high > 0 > low
    with pytest.raises(ValueError):
        analysis.tidal_range(readings.iloc[0:0])


@pytest.mark.parametrize(("units", "expected"), [("mm", 25.4), ("inches", 1.0)])
def test_unit_conversion(units: str, expected: float) -> None:
    """Millimetres convert to inches at 25.4 to one."""
    assert analysis.to_units(25.4, units) == pytest.approx(expected)
