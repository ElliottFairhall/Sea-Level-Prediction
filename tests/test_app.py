"""Run the whole page against the offline fakes, including outages and new data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from src import sources

from .conftest import make_http_get

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = [
    "How up to date is the data?",
    "One ocean, two instruments",
    "Is the rise speeding up?",
    "Your coast",
    "How often is the sea high? A century at Newlyn",
    "Where could this go?",
    "Right now",
    "What this page can and cannot tell you",
    "Data and credits",
]


def run() -> AppTest:
    """Run the app headlessly."""
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def segmented(at: AppTest, label: str) -> Any:
    """Find a segmented control by its label."""
    return next(
        c for c in at.get("segmented_control") if getattr(c, "label", "") == label
    )


def headings(at: AppTest) -> list[str]:
    """Text of every subheader on the page."""
    return [item.value for item in at.subheader]


def intro_text(at: AppTest) -> str:
    """The introduction paragraph (the first markdown block is the stylesheet)."""
    return str(
        next(
            item.value
            for item in at.markdown
            if "Global sea level has risen" in item.value
            or "rebuilds an old, fixed" in item.value
        )
    )


def test_page_renders_every_section() -> None:
    """With all sources up, all nine sections appear and nothing errors."""
    at = run()
    assert not at.exception, [e.value for e in at.exception]
    assert headings(at) == SECTIONS
    assert at.title[0].value == "Sea level, from 1880 to now"


def test_freshness_table_reports_each_source() -> None:
    """The table says how current each source is, and the live one is minutes old."""
    table = run().dataframe[0].value
    assert len(table) == 6
    live = table[table["Source"].str.contains("Environment Agency")].iloc[0]
    assert "minutes" in live["Behind today"]
    original = table[table["Source"].str.contains("EPA")].iloc[0]
    assert original["Latest data"].endswith("2013")


def test_intro_reports_the_last_global_year_from_the_data() -> None:
    """The year in the intro is the last full satellite year, not a typed number."""
    assert "run to 2025" in intro_text(run())


def test_years_move_when_newer_satellite_data_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feed a satellite file that runs to 2031 and every year on the page follows."""
    monkeypatch.setattr("src.sources.http_get", make_http_get(colorado_end=2031.99))
    at = run()
    assert not at.exception, [e.value for e in at.exception]
    assert "run to 2031" in intro_text(at)
    horizon = at.slider(key="horizon")
    assert horizon.min == 2036
    assert horizon.value == 2056
    assert any(
        "2031" in str(df.value["Latest data"].tolist()) for df in at.dataframe[:1]
    )


def test_citations_carry_todays_year_and_date() -> None:
    """The PSMSL citation is stamped with the current year and retrieval date."""
    text = " ".join(item.value for item in run().markdown)
    year = datetime.now(sources.UTC).year
    assert f"PSMSL), {year}" in text
    assert f"{datetime.now(sources.UTC):%d %B %Y}" in text
    assert "release 2026_rel2" in text


def test_every_source_is_credited() -> None:
    """Each organisation behind the data is named with its terms."""
    text = " ".join(item.value for item in run().markdown)
    for name in (
        "EPA / CSIRO",
        "University of Colorado",
        "PSMSL",
        "NOAA",
        "UHSLC",
        "Environment Agency",
    ):
        assert name in text
    assert (
        "This uses Environment Agency tide gauge data from the real-time data API (Beta)"
        in " ".join(c.value for c in run().caption)
    )


def test_units_toggle_switches_to_inches() -> None:
    """Choosing inches changes metric labels and values."""
    at = run()
    segmented(at, "Units").set_value("inches").run()
    assert not at.exception
    assert any("inches" in metric.label for metric in at.metric)


@pytest.mark.parametrize(
    "key", ["psmsl-830", "psmsl-3", "noaa-9414290", "noaa-8771450"]
)
def test_every_kind_of_station_renders(key: str) -> None:
    """UK and US gauges both work, and the land-motion sentence is shown or explained."""
    at = run()
    at.selectbox(key="station").set_value(key).run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(item.value for item in at.markdown)
    assert ("land here is" in text) or (
        "cannot show whether the land is moving" in text
    )


def test_window_and_horizon_controls_update_the_page() -> None:
    """Changing the window and the look-ahead year re-draws without error."""
    at = run()
    at.slider(key="window").set_value(10).run()
    at.slider(key="horizon").set_value(2100).run()
    assert not at.exception
    assert "Level in 2100" in " ".join(str(c) for c in at.dataframe[-1].value.columns)


def test_extremes_section_reports_a_ratio() -> None:
    """The Newlyn section says how many times more often the threshold is crossed."""
    text = " ".join(item.value for item in run().markdown)
    assert "times as often" in text
    assert "This record ends in 2016" in " ".join(c.value for c in run().caption)


def test_scenario_table_has_both_models() -> None:
    """Both simple models are tabulated with their range."""
    table = run().dataframe[-1].value
    assert table["Model"].tolist() == ["Steady", "Speeding up"]


def test_live_panel_puts_the_tide_in_context() -> None:
    """The live panel shows a tidal range and compares it with a year's rise."""
    at = run()
    labels = [metric.label for metric in at.metric]
    assert "Tidal range" in labels
    assert "times larger than a year's rise" in " ".join(
        item.value for item in at.markdown
    )


def test_satellite_outage_degrades_the_page_but_does_not_stop_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the satellite source is down the page says so and the rest still works."""
    monkeypatch.setattr("src.sources.http_get", make_http_get(down=("colorado.edu",)))
    at = run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(
        "satellite record could not be loaded" in w.value.lower() for w in at.warning
    )
    assert "Your coast" in headings(at)
    assert "One ocean, two instruments" not in headings(at)
    assert "ends in 2013" in intro_text(at)


def test_newlyn_daily_outage_only_loses_the_extremes_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure in one source takes out only the section that needs it."""
    monkeypatch.setattr("src.sources.http_get", make_http_get(down=("uhslc",)))
    at = run()
    assert not at.exception
    assert any("Newlyn daily record" in w.value for w in at.warning)
    assert "One ocean, two instruments" in headings(at)
    assert "How often is the sea high? A century at Newlyn" not in headings(at)


def test_live_feed_outage_is_reported_in_the_freshness_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the Environment Agency down, the table says unavailable and the page continues."""
    monkeypatch.setattr(
        "src.sources.http_get", make_http_get(down=("environment.data.gov.uk",))
    )
    at = run()
    assert not at.exception
    table = at.dataframe[0].value
    live = table[table["Source"].str.contains("Environment Agency")].iloc[0]
    assert live["Latest data"] == "unavailable"
    assert any("live tide gauge list" in w.value.lower() for w in at.warning)


def test_page_survives_every_source_being_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with nothing reachable the bundled data keeps the page up."""
    monkeypatch.setattr(
        "src.sources.http_get",
        make_http_get(
            down=(
                "colorado.edu",
                "psmsl.org",
                "tidesandcurrents",
                "uhslc",
                "environment.data.gov.uk",
            )
        ),
    )
    at = run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Data and credits" in headings(at)
    assert len(at.warning) >= 3
