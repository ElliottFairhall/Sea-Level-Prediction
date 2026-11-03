"""Tests for fetching and parsing the data sources."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import requests

from src import sources

SRC = Path(__file__).resolve().parent.parent / "src"


def test_decimal_years_become_the_right_dates() -> None:
    """1993.5 is early July 1993 and a whole year is January 1st."""
    out = sources.decimal_year_to_timestamp(pd.Series([1993.0, 1993.5, 2000.25]))
    assert out.iloc[0] == pd.Timestamp("1993-01-01")
    assert out.iloc[1].month == 7
    assert out.iloc[2].month == 4


def test_parse_colorado_skips_comments_and_reads_two_columns() -> None:
    """Comment lines are ignored; levels are millimetres."""
    text = "# header\n1993.0100  -21.3\n1993.0387  -21.2\n"
    frame = sources.parse_colorado(text)
    assert list(frame.columns) == ["date", "level_mm"]
    assert frame["level_mm"].tolist() == [-21.3, -21.2]


def test_parse_colorado_rejects_an_empty_file() -> None:
    """A file with no data rows is an error, not an empty result."""
    with pytest.raises(sources.SourceError):
        sources.parse_colorado("# only a comment\n")


def test_parse_psmsl_drops_missing_months() -> None:
    """The -99999 marker is removed and the rest are kept."""
    text = "1915.3750;  7111;  0;  000\n1915.4583;  -99999;  0;  000\n1915.5417;  7090;  0;  000\n"
    frame = sources.parse_psmsl(text)
    assert frame["level_mm"].tolist() == [7111.0, 7090.0]


def test_parse_noaa_converts_metres_to_millimetres() -> None:
    """NOAA reports metres; the page works in millimetres."""
    text = json.dumps({"data": [{"year": "2026", "month": "9", "MSL": "0.222"}]})
    frame = sources.parse_noaa_monthly(text)
    assert frame["level_mm"].iloc[0] == pytest.approx(222.0)
    assert frame["date"].iloc[0] == pd.Timestamp("2026-09-15")


def test_parse_noaa_surfaces_an_api_error() -> None:
    """NOAA's error payload becomes a clear exception."""
    text = json.dumps({"error": {"message": " Wrong Date: range too long "}})
    with pytest.raises(sources.SourceError, match="Wrong Date"):
        sources.parse_noaa_monthly(text)


def test_parse_uhslc_skips_the_units_row_and_missing_days() -> None:
    """The second header row holds units, and NaN days are dropped."""
    text = "time,sea_level,uhslc_id\nUTC,millimeters,\n1915-04-23T12:00:00Z,2948,294\n1915-04-24T12:00:00Z,NaN,294\n"
    frame = sources.parse_uhslc(text)
    assert len(frame) == 1
    assert frame["date"].iloc[0] == pd.Timestamp("1915-04-23")


def test_parse_ea_readings_sorts_and_removes_duplicates() -> None:
    """Readings come back oldest first, once each."""
    items: list[dict] = [
        {"dateTime": "2026-11-03T01:15:00Z", "value": 3.4},
        {"dateTime": "2026-11-03T01:00:00Z", "value": 3.3},
        {"dateTime": "2026-11-03T01:00:00Z", "value": 3.3},
        {"dateTime": "2026-11-03T00:45:00Z", "value": "n/a"},
    ]
    frame = sources.parse_ea_readings(items)
    assert frame["level_m"].tolist() == [3.3, 3.4]


def test_every_station_key_is_unique_and_resolvable() -> None:
    """The picker's keys map back to one station each."""
    keys = [station.key for station in sources.STATIONS]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(sources.STATION_BY_KEY)
    assert {s.provider for s in sources.STATIONS} == {"PSMSL", "NOAA"}


def test_discovery_finds_the_newest_colorado_release() -> None:
    """The latest release listed on the data page is the one used."""
    url = sources.discover_colorado_url()
    assert url.endswith("seasons_rmvd.txt")
    assert sources.release_name(url) == "2026_rel2"


def test_discovery_falls_back_when_the_site_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the data page fails, the known address is used instead."""

    def broken(url: str, params: dict[str, str] | None = None) -> str:
        raise sources.SourceError("down")

    monkeypatch.setattr("src.sources.http_get", broken)
    assert sources.discover_colorado_url() == sources.COLORADO_FALLBACK


def test_fetch_colorado_remembers_which_release_it_used() -> None:
    """The release address travels with the data for the credits."""
    assert (
        sources.release_name(sources.fetch_colorado().attrs["source_url"])
        == "2026_rel2"
    )


def test_fetch_station_handles_both_providers() -> None:
    """A UK gauge and a US gauge both come back as date and millimetres."""
    for key in ("psmsl-202", "noaa-8518750"):
        frame = sources.fetch_station(sources.STATION_BY_KEY[key])
        assert list(frame.columns) == ["date", "level_mm"]
        assert len(frame) > 1000


def test_download_turns_network_errors_into_source_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers only ever need to catch SourceError."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("src.sources.requests.get", boom)
    with pytest.raises(sources.SourceError, match="request failed"):
        sources.download("https://example.invalid/")


def test_download_turns_http_errors_into_source_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 or 500 is an error, not an empty page."""

    class Bad:
        """A response whose status check always fails."""

        text = "nope"

        def raise_for_status(self) -> None:
            """Fail like a 404."""
            raise requests.HTTPError("404")

    monkeypatch.setattr("src.sources.requests.get", lambda *_a, **_k: Bad())
    with pytest.raises(sources.SourceError):
        sources.download("https://example.invalid/")


def test_credits_fill_in_today_and_the_release() -> None:
    """Citation years and the Colorado release are never written in advance."""
    now = datetime(2031, 3, 9, tzinfo=sources.UTC)
    entries = sources.credits(now=now, release="2031_rel1")
    text = " ".join(entry["credit"] + entry["name"] for entry in entries)
    assert "PSMSL), 2031" in text
    assert "09 March 2031" in text
    assert "release 2031_rel1" in text
    assert "{" not in text


def test_no_current_years_are_hard_coded_in_the_source_code() -> None:
    """Any year from 2020 on in the code would go stale; only the fallback URL may name one."""
    offenders = []
    for path in SRC.glob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if (
                re.search(r"(?<![\d-])20[2-3]\d(?![\d-])", line)
                and "sealevel.colorado.edu/files/" not in line
            ):
                offenders.append(f"{path.name}:{number}: {line.strip()[:80]}")
    assert not offenders, offenders
