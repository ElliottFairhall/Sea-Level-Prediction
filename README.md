# Sea level, from 1880 to now

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://elliottfairhall-sea-level-prediction-main-f0riix.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project started as a fixed dataset that stopped in 2013. It now joins that long record to live and regularly
updated sources, and uses them to teach how sea level data is handled: how fast the sea is rising, how sure we can be,
and what it means on a particular coast. Built with Streamlit, pandas, SciPy and Plotly.

## What it does

- Shows how current every source is, using what each one returned when the page loaded
- Joins a tide gauge reconstruction (1880 to 2013) to satellite altimetry (1993 to within months of today) and checks
  how well they agree
- Asks whether the rate is changing: rolling rates over a window you choose, rates by era, and a straight line compared
  with a curve, all with honest uncertainty
- Compares fourteen tide gauges (eight UK, six US) with the global ocean and estimates land movement
- Counts how often a fixed high level was reached at Newlyn over a century
- Extends two simple models forward, clearly labelled as extrapolations and not forecasts
- Shows the last two days of a live Environment Agency tide gauge, to put the tide in context
- Says plainly what it can and cannot tell you, and credits every source

## Data

| Source                                       | Used for                | Freshness                 |
| -------------------------------------------- | ----------------------- | ------------------------- |
| EPA / CSIRO (bundled file)                   | Long global record      | Ends 2013, fixed          |
| University of Colorado global mean sea level | Satellite global series | About two months behind   |
| PSMSL monthly means                          | UK tide gauges          | Up to about a year behind |
| NOAA CO-OPS monthly means                    | US tide gauges          | About one month behind    |
| UHSLC research-quality daily data            | Newlyn extremes         | Ends 2016                 |
| Environment Agency tide gauge API            | Live tide panel         | Minutes                   |

Third-party data is fetched when the page runs and is **not** copied into this repository. Each source's credit and
terms are shown on the page and defined in `src/sources.py`. If one source is unavailable the section that needs it says
so and the rest of the page keeps working. Dates, years and citations are computed from the data and the clock, never
typed in; a test fails if a current year is hard-coded.

## Techniques

Each is explained on the page where it is used and implemented in `src/analysis.py`.

- Joining two records with different baselines using their overlap, then checking the join
- Trends with a moving-block bootstrap, because neighbouring years are not independent
- Rolling-window trends and a quadratic fit for acceleration
- Removing the seasonal cycle from monthly data and expressing gauges against a common baseline
- Estimating vertical land movement from the gap between a gauge and the global ocean
- Counting extremes against a fixed early threshold
- Statistical bands around simple projections

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs Python 3.11 or newer and an internet connection. The theme follows your system's light or dark setting and is
defined in `.streamlit/config.toml`.

## Test it

```bash
pip install -e ".[dev]"
pytest
```

The tests never touch the network. They serve offline copies of every source in the real formats and cover the parsers,
each technique (against data with known answers), the whole page, outages of each source, and newer data arriving.
`tests/test_pillow_security_*.py` fails if Pillow is capped below 13 or Python 3.10 is allowed, because older Pillow
releases carry known security advisories.

## Limits

This is a teaching and exploration tool, not a forecast or a flood-risk service. The scenarios extend past trends and
know nothing about ice sheets or emissions; for physically based projections see the
[IPCC Sixth Assessment Report](https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-9/). Monthly and daily averages hide
tides and storm surges. UK gauge records lag by about a year.

Part of [Data Flakes](https://www.data-flakes.dev/).
