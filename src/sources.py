"""Fetch and parse the sea level data sources used by the page.

Nothing here depends on Streamlit, so every parser can be tested with a small
hand-written sample and every network call goes through :func:`http_get`.

Third-party data is fetched when the page runs and is never copied into this
repository. Each source's credit and terms are in :data:`SOURCES`.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# datetime.UTC needs Python 3.11, and the repository-wide mypy target is 3.10.
UTC = timezone.utc  # noqa: UP017

TIMEOUT_SECONDS = 25
USER_AGENT = "Mozilla/5.0 (compatible; data-flakes-sea-level)"
PSMSL_MISSING = -99999
MM_PER_METRE = 1000.0

COLORADO_INDEX = "https://sealevel.colorado.edu/data/"
COLORADO_FALLBACK = (
    "https://sealevel.colorado.edu/files/2026_rel2/gmsl_2026rel2_seasons_rmvd.txt"
)
PSMSL_URL = "https://psmsl.org/data/obtaining/rlr.monthly.data/{id}.rlrdata"
NOAA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
UHSLC_URL = "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_daily_rqds.csv"
EA_STATIONS_URL = "https://environment.data.gov.uk/flood-monitoring/id/stations"
EA_MEASURE_URL = (
    "https://environment.data.gov.uk/flood-monitoring/id/measures/{m}/readings"
)


class SourceError(RuntimeError):
    """A data source could not be fetched or understood."""


@dataclass(frozen=True)
class Station:
    """A tide gauge the page can show.

    Attributes:
        key: Short unique key, for example ``"psmsl-202"``.
        name: Place name.
        provider: ``"PSMSL"`` or ``"NOAA"``.
        code: The provider's station identifier.
        region: ``"UK"`` or ``"USA"``, for grouping in the picker.
    """

    key: str
    name: str
    provider: str
    code: str
    region: str


STATIONS = [
    Station("psmsl-202", "Newlyn, Cornwall", "PSMSL", "202", "UK"),
    Station("psmsl-830", "Lerwick, Shetland", "PSMSL", "830", "UK"),
    Station("psmsl-1109", "Wick, Highland", "PSMSL", "1109", "UK"),
    Station("psmsl-755", "Millport, Firth of Clyde", "PSMSL", "755", "UK"),
    Station("psmsl-5", "Holyhead, Anglesey", "PSMSL", "5", "UK"),
    Station("psmsl-95", "North Shields, Tyne", "PSMSL", "95", "UK"),
    Station("psmsl-3", "Sheerness, Kent", "PSMSL", "3", "UK"),
    Station("psmsl-350", "Portsmouth, Hampshire", "PSMSL", "350", "UK"),
    Station("noaa-8518750", "The Battery, New York", "NOAA", "8518750", "USA"),
    Station("noaa-9414290", "San Francisco", "NOAA", "9414290", "USA"),
    Station("noaa-8724580", "Key West, Florida", "NOAA", "8724580", "USA"),
    Station("noaa-1612340", "Honolulu, Hawaii", "NOAA", "1612340", "USA"),
    Station("noaa-8771450", "Galveston, Texas", "NOAA", "8771450", "USA"),
    Station("noaa-9447130", "Seattle", "NOAA", "9447130", "USA"),
]
STATION_BY_KEY = {station.key: station for station in STATIONS}
NEWLYN_UHSLC_ID = "294"

SOURCES = [
    {
        "name": "EPA / CSIRO global sea level, 1880 to 2013",
        "role": "The long global record (the data this page used to be built on).",
        "credit": (
            "US Environmental Protection Agency, Climate Change Indicators: Sea "
            "Level, compiled from the CSIRO reconstruction of Church and White "
            "(2011), Surveys in Geophysics 32, 585-602."
        ),
        "terms": "Bundled with this app as a fixed file; it has not been updated since 2013.",
        "url": "https://www.epa.gov/climate-indicators/climate-change-indicators-sea-level",
    },
    {
        "name": "University of Colorado global mean sea level",
        "role": "Global sea level from satellite radar altimetry, from 1993 to within a few months of today.",
        "credit": (
            "CU Sea Level Research Group, University of Colorado, from NASA and "
            "CNES altimetry. Method: Nerem et al. (2018), PNAS 115, 2022-2025. The "
            "group thanks NOAA and TU Delft for the Radar Altimeter Database System."
        ),
        "terms": "No citation or redistribution terms were stated on the pages checked.",
        "url": "https://sealevel.colorado.edu/data/",
    },
    {
        "name": "PSMSL monthly mean sea level (RLR)",
        "role": "Long UK tide gauge records, up to about a year behind.",
        "credit": (
            "Permanent Service for Mean Sea Level (PSMSL), {year}, Tide Gauge Data, "
            "retrieved {today} from http://www.psmsl.org/data/obtaining/. "
            "Holgate et al. (2013), Journal of Coastal Research 29(3), 493-504, "
            "doi:10.2112/JCOASTRES-D-12-00175.1."
        ),
        "terms": "Free to use; PSMSL asks users to cite both references.",
        "url": "https://psmsl.org/data/obtaining/",
    },
    {
        "name": "NOAA CO-OPS monthly mean sea level",
        "role": "Long US tide gauge records, current to last month.",
        "credit": "NOAA National Ocean Service, Center for Operational Oceanographic Products and Services.",
        "terms": "US government data, open access.",
        "url": "https://tidesandcurrents.noaa.gov/",
    },
    {
        "name": "UHSLC research quality daily sea level, Newlyn",
        "role": "A century of daily readings at Newlyn (1915 to 2016) for the extremes analysis.",
        "credit": (
            "Caldwell, P. C., M. A. Merrifield, P. R. Thompson (2015), Sea level "
            "measured by tide gauges from global oceans, the Joint Archive for Sea "
            "Level holdings (NCEI Accession 0019568), Version 5.5, NOAA NCEI, "
            "doi:10.7289/V5V40S7W."
        ),
        "terms": "UHSLC asks users to cite the dataset.",
        "url": "https://uhslc.soest.hawaii.edu/datainfo/",
    },
    {
        "name": "Environment Agency tide gauge real-time API (Beta)",
        "role": "Live 15-minute tide readings for the 'right now' panel.",
        "credit": "This uses Environment Agency tide gauge data from the real-time data API (Beta).",
        "terms": "Open Government Licence v3.0. Readings are provisional and unvalidated.",
        "url": "https://environment.data.gov.uk/flood-monitoring/doc/reference",
    },
]


def credits(
    now: datetime | None = None, release: str | None = None
) -> list[dict[str, str]]:
    """Return the source list with dates filled in for today.

    Citations ask for the year and retrieval date, and the Colorado release
    changes, so none of that is written into the text ahead of time.

    Args:
        now: The current time (defaults to now, UTC).
        release: Colorado release name found at load time, for example
            ``YYYY_relN``.

    Returns:
        A copy of :data:`SOURCES` with placeholders resolved.
    """
    stamp = now or datetime.now(UTC)
    out = []
    for source in SOURCES:
        entry = dict(source)
        entry["credit"] = entry["credit"].format(
            year=stamp.year, today=f"{stamp:%d %B %Y}"
        )
        if release and "Colorado" in entry["name"]:
            entry["name"] = f"{entry['name']} (release {release})"
        out.append(entry)
    return out


def release_name(url: str | None) -> str | None:
    """Pull a release name such as ``YYYY_relN`` out of a Colorado file URL."""
    match = re.search(r"(\d{4}_?rel\d+)", url or "")
    return match.group(1) if match else None


def http_get(url: str, params: dict[str, str] | None = None) -> str:
    """Fetch a URL as text. Every network call goes through here (tests replace it)."""
    return download(url, params)


def download(url: str, params: dict[str, str] | None = None) -> str:
    """Download a URL as text.

    Args:
        url: The address to fetch.
        params: Optional query parameters.

    Returns:
        The response body.

    Raises:
        SourceError: If the request fails or the server returns an error.
    """
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise SourceError(f"request failed for {url}: {error}") from error
    return response.text


def decimal_year_to_timestamp(values: pd.Series) -> pd.Series:
    """Convert decimal years such as 1993.2056 to timestamps.

    Args:
        values: Decimal years.

    Returns:
        The matching datetimes, accurate to about a day.
    """
    years = np.floor(values).astype(int)
    start = pd.to_datetime(years.astype(str) + "-01-01")
    end = pd.to_datetime((years + 1).astype(str) + "-01-01")
    return start + (end - start) * (values - years)


def parse_colorado(text: str) -> pd.DataFrame:
    """Parse the Colorado global mean sea level ASCII file.

    Args:
        text: File contents: comment lines starting with ``#`` then two
            columns, decimal year and millimetres.

    Returns:
        Columns ``date`` and ``level_mm``, one row per 10-day cycle.

    Raises:
        SourceError: If no data rows are found.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.lstrip().startswith("#"):
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise SourceError("no data rows in the Colorado file")
    frame = pd.DataFrame(rows, columns=["year_frac", "level_mm"])
    frame["date"] = decimal_year_to_timestamp(frame["year_frac"])
    return frame[["date", "level_mm"]]


def parse_psmsl(text: str) -> pd.DataFrame:
    """Parse a PSMSL RLR monthly file.

    Args:
        text: Lines of ``decimal_year; height_mm; missing_days; flag``.

    Returns:
        Columns ``date`` (mid-month) and ``level_mm``; missing months removed.

    Raises:
        SourceError: If no valid months are found.
    """
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue
        try:
            year_frac, height = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if height != PSMSL_MISSING:
            rows.append((year_frac, height))
    if not rows:
        raise SourceError("no valid months in the PSMSL file")
    frame = pd.DataFrame(rows, columns=["year_frac", "level_mm"])
    frame["date"] = decimal_year_to_timestamp(frame["year_frac"])
    return frame[["date", "level_mm"]]


def parse_noaa_monthly(text: str) -> pd.DataFrame:
    """Parse NOAA CO-OPS ``monthly_mean`` JSON (metric units, MSL datum).

    Args:
        text: The JSON response.

    Returns:
        Columns ``date`` (mid-month) and ``level_mm``.

    Raises:
        SourceError: If NOAA returned an error or no data.
    """
    import json

    try:
        payload = json.loads(text)
    except ValueError as error:
        raise SourceError("NOAA response was not JSON") from error
    if "error" in payload:
        raise SourceError(str(payload["error"].get("message", "NOAA error")).strip())
    rows = [
        (int(item["year"]), int(item["month"]), float(item["MSL"]))
        for item in payload.get("data", [])
        if item.get("MSL") not in (None, "")
    ]
    if not rows:
        raise SourceError("no monthly values from NOAA")
    frame = pd.DataFrame(rows, columns=["year", "month", "metres"])
    frame["date"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=15)
    )
    frame["level_mm"] = frame["metres"] * MM_PER_METRE
    return frame[["date", "level_mm"]]


def parse_uhslc(text: str) -> pd.DataFrame:
    """Parse a UHSLC ERDDAP daily CSV (two header rows: names then units).

    Args:
        text: CSV with ``time`` and ``sea_level`` columns, levels in millimetres.

    Returns:
        Columns ``date`` and ``level_mm``; missing days removed.

    Raises:
        SourceError: If no valid days are found.
    """
    frame = pd.read_csv(io.StringIO(text), skiprows=[1])
    if "sea_level" not in frame or "time" not in frame:
        raise SourceError("UHSLC response lacks time or sea_level columns")
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    frame["level_mm"] = pd.to_numeric(frame["sea_level"], errors="coerce")
    frame = frame.dropna(subset=["level_mm"])
    if frame.empty:
        raise SourceError("no valid days in the UHSLC response")
    frame["date"] = frame["date"].dt.normalize()
    return frame[["date", "level_mm"]].reset_index(drop=True)


def parse_ea_readings(items: list[dict]) -> pd.DataFrame:
    """Turn Environment Agency reading items into a frame.

    Args:
        items: The ``items`` list from the readings endpoint.

    Returns:
        Columns ``time`` (UTC) and ``level_m`` in metres above ordnance datum,
        oldest first, with exact duplicates removed.
    """
    frame = pd.DataFrame(
        [
            (item["dateTime"], item["value"])
            for item in items
            if isinstance(item.get("value"), int | float)
        ],
        columns=["time", "level_m"],
    )
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def discover_colorado_url() -> str:
    """Find the newest Colorado seasons-removed file from their data page.

    Releases are renamed (``YYYY_relN`` and so on), so the page is read to find
    the latest; a known address is the fallback.

    Returns:
        URL of the ASCII file.
    """
    try:
        index = http_get(COLORADO_INDEX)
        releases = sorted(
            set(re.findall(r'href="[^"]*/data/(\d{4})_?rel(\d+)"', index)),
            key=lambda pair: (int(pair[0]), int(pair[1])),
        )
        if releases:
            year, rel = releases[-1]
            page = http_get(f"{COLORADO_INDEX}{year}rel{rel}")
            match = re.search(r'https?://[^"\']+/gmsl_[^"\']*seasons_rmvd\.txt', page)
            if match:
                return match.group(0)
    except SourceError:
        logger.warning("Colorado release discovery failed; using the known address")
    return COLORADO_FALLBACK


def fetch_colorado() -> pd.DataFrame:
    """Download the global satellite series.

    The address of the release that was used is kept in ``frame.attrs``.
    """
    url = discover_colorado_url()
    frame = parse_colorado(http_get(url))
    frame.attrs["source_url"] = url
    return frame


def fetch_station(station: Station) -> pd.DataFrame:
    """Download one tide gauge's monthly mean series.

    Args:
        station: A station from :data:`STATIONS`.

    Returns:
        Columns ``date`` and ``level_mm``.
    """
    if station.provider == "PSMSL":
        return parse_psmsl(http_get(PSMSL_URL.format(id=station.code)))
    today = datetime.now(UTC).strftime("%Y%m%d")
    text = http_get(
        NOAA_URL,
        {
            "product": "monthly_mean",
            "station": station.code,
            "datum": "MSL",
            "units": "metric",
            "time_zone": "gmt",
            "begin_date": "18500101",
            "end_date": today,
            "format": "json",
        },
    )
    return parse_noaa_monthly(text)


def fetch_newlyn_daily() -> pd.DataFrame:
    """Download Newlyn's research-quality daily series from UHSLC."""
    query = f"?time,sea_level&uhslc_id={NEWLYN_UHSLC_ID}&orderBy(%22time%22)"
    return parse_uhslc(http_get(UHSLC_URL + query))


def fetch_ea_tide_stations() -> list[dict[str, str]]:
    """List Environment Agency tide gauges that report ordnance-datum levels.

    Returns:
        Dicts with ``label`` and ``measure`` (the id used to fetch readings),
        sorted by label, one per place.
    """
    import json

    payload = json.loads(
        http_get(EA_STATIONS_URL, {"type": "TideGauge", "_limit": "500"})
    )
    found: dict[str, str] = {}
    for item in payload.get("items", []):
        measures = item.get("measures") or []
        if isinstance(measures, dict):
            measures = [measures]
        for measure in measures:
            ident = str(measure.get("@id", ""))
            if ident.endswith("mAOD") and item.get("label"):
                found.setdefault(item["label"], ident.rsplit("/", 1)[1])
    return [{"label": k, "measure": v} for k, v in sorted(found.items())]


def fetch_ea_readings(measure: str, since: datetime) -> pd.DataFrame:
    """Download recent 15-minute readings for one Environment Agency measure.

    Args:
        measure: Measure id from :func:`fetch_ea_tide_stations`.
        since: Earliest time wanted (UTC).

    Returns:
        Columns ``time`` and ``level_m``.
    """
    import json

    text = http_get(
        EA_MEASURE_URL.format(m=measure),
        {"since": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "_limit": "600", "_sorted": ""},
    )
    return parse_ea_readings(json.loads(text).get("items", []))
