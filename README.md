# Airport Investment Intelligence Agent

A locally runnable proof of concept for a U.S. airport-modernization investment firm. One bounded conversational
analytics agent helps analysts find airports that merit further review because they show **sustained demand growth**,
**high scheduled-capacity utilization**, or **operational pressure**.

It is a **screening tool**. It does not calculate investment ROI, forecast profitability, measure unmet demand, or
prove that terminal renovation causes an operational problem.

## Architecture

```text
Browser
  -> Streamlit application (app.py, src/ui_components.py, src/charts.py)
      -> one LLM analytics agent (src/agent.py; Anthropic default, Gemini alternative, deterministic Demo mode)
          -> allow-listed typed analytical tools (src/tools.py, Pydantic contracts in src/models.py)
              -> deterministic Python and DuckDB calculations (src/analytics.py, src/scoring.py)
                  -> local DuckDB historical store data/airport_intelligence.duckdb (src/db.py, src/data_ingestion.py)
                  -> optional live aviation REST adapter (src/live_provider.py, Aviationstack)
```

Historical data is the only input to rankings and scores. Live data answers current-status questions only and is
always shown in its own `Current Operational Context` section. The language model never computes a number: it
chooses a tool, passes validated parameters, and explains the tool result; a grounding guard checks that every
figure in an answer appears in a tool result. See `docs/architecture.md`.

## Setup

Requires Python 3.11+ (on this machine: `"C:\Program Files\Python311\python.exe"`).

```bash
"C:\Program Files\Python311\python.exe" -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
copy .env.example .env
```

Edit `.env` and add any keys you have (all optional - the app starts without them):

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `auto` (default), `anthropic`, `gemini` or `demo` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Anthropic runtime (default model `claude-opus-5-5`) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini runtime used when only a Gemini key is present (default `gemini-3.8-flash`) |
| `AVIATIONSTACK_API_KEY`, `AVIATIONSTACK_BASE_URL` | live operational context (optional) |
| `DUCKDB_PATH` | local store (default `data/airport_intelligence.duckdb`) |
| `LONG_HAUL_THRESHOLD_MILES`, `MIN_REPORTED_FLIGHTS` | analysis defaults (2000 miles, 500 flights) |
| `DEMO_SCOPE_START`, `NATIONWIDE_SCOPE_START` | first month of each refresh scope (2023-01, 2019-01) |

Without an LLM key the sidebar shows **Demo / deterministic mode**: questions are routed by rules to the same
tools, so the suggested questions and the supported question patterns work fully.

## Refresh the historical data (required once)

```bash
.venv\Scripts\python.exe scripts/refresh_data.py --scope demo
```

```bash
.venv\Scripts\python.exe scripts/refresh_data.py --scope nationwide
```

- `demo`: all U.S. airports, months from 2023-01 to the latest available (about 1.3 GB of downloads, ~15-25 minutes).
- `nationwide`: the same pipeline from 2019-01 (about 2.8 GB, 45-60 minutes).
- Options: `--start YYYY-MM`, `--end YYYY-MM`, `--keep-raw` (cache zips under `data/raw/` for re-runs), `--workers 4`,
  `--show-metadata`, `--scores-only`.
- The runtime application reads only the aggregated DuckDB tables; raw files are never queried during a chat.
- If a source fails, the metadata table records `failed` honestly and the command exits with code 2 and prints the
  retry command. Nothing is faked.

## Run

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Headless start check: `python scripts/smoke_streamlit.py`.

## Test

```bash
.venv\Scripts\python.exe -m pytest -q
```

Smoke tests that need credentials and a refreshed database:

```bash
.venv\Scripts\python.exe scripts/smoke_llm.py
```

```bash
.venv\Scripts\python.exe scripts/smoke_live.py LAX
```

## Data sources

| Source | Used for | Notes |
|---|---|---|
| BTS TranStats **T-100 Segment (All Carriers)** | passengers, available seats, departures performed, route distance | one file per calendar year via the TranStats download form; U.S. origins, scheduled service (class F) |
| BTS TranStats **Reporting Carrier On-Time Performance (1987-present)** | scheduled flights, departures delayed 15+ minutes, cancellations, departure delay minutes | static monthly zips; reporting carriers only |
| **OurAirports** `airports.csv` (public domain) | airport names, locations, airport type | fallback: names derived from BTS city/state fields |
| **Aviationstack** `/v1/flights` (optional, live) | current departure status sample | never used in scores |

## KPI formulas

```text
Demand Growth            = (passengers latest 12 months - passengers prior 12 months) / passengers prior 12 months
Capacity Utilization     = total passengers / total available seats (latest 12 months)
Operational Pressure     = 70% x delayed-departure rate + 30% x cancellation rate
Expansion Opportunity    = 40% demand growth + 35% capacity utilization + 25% operational pressure (0-100 components)
Capacity Pressure Proxy  = 50% utilization percentile + 30% demand-growth percentile + 20% operational-pressure percentile
```

- Delayed departure: delayed by 15 minutes or more. Delay rate = delayed / operated departures; cancellation rate =
  cancelled / scheduled flights.
- Long-haul flight: route segment of at least 2,000 statute miles (configurable in the sidebar).
- New England: CT, ME, MA, NH, RI, VT.
- Peer group: airports in the active data scope matching the optional state/region and size-band filters that pass
  the coverage rule (at least 500 reported flights in **both** 12-month windows, for On-Time flights and T-100
  departures). Excluded airports are listed with the reason.
- Normalization: each component is winsorized at the 5th/95th percentile within the peer group, then min-max scaled
  to 0-100. Size band is a peer filter only and never adds points.
- Confidence: high (12/12 months, peer group >= 10, no quality flags, period not clamped), medium, or low (< 10 months,
  peer group < 5, or any non-ok status).

## Limitations

- Screening signals only: **not ROI, not profitability, not true unmet-demand modelling**. The capacity-pressure
  proxy is a proxy; direct unmet demand would require search, booking, fare, leakage, or alternative-airport data.
- Delay and cancellation statistics cover BTS reporting carriers only; small airports may be excluded for coverage.
- T-100 traffic uses scheduled passenger/combination service (class F); all-cargo and charter operations are excluded.
- Seasonal airports can pass the flight-count rule with fewer than 12 reporting months; their rows carry lower confidence.
- The live sample is one page of the provider's feed (up to 100 departures) and is not comparable to historical rates.

## Sample questions

- Which airports in New England are strong candidates for terminal expansion?
- Compare LAX and Santa Ana Airport, SNA, congestion levels.
- What percentage of departures from Anchorage, ANC, are long-haul?
- What is the capacity-pressure signal at SFO and why?
- Which medium-sized airports in Texas have the strongest demand growth?
- Compare Denver, Phoenix, and Las Vegas.
- What changed at Boston Logan in 2024?
- What is the current operational status at LAX?
- Which airports have growing passengers and worsening cancellations?
- Follow-up: "Compare LAX and SNA congestion." then "What about 2024?"

## Known live-data constraints

- Without `AVIATIONSTACK_API_KEY` the live tool returns an explicit `unavailable` state; historical analysis is unaffected.
- Free Aviationstack plans have small monthly quotas; `usage_limit_reached` / `rate_limit_reached` are shown as
  unavailable with the provider's `Retry-After` when present.
- Live context is only requested when the sidebar toggle is on.

## Troubleshooting (Windows / corporate networks)

- TLS inspection: `requests` uses certifi's CA bundle, which may not include your organization's root. The refresh
  uses `truststore` (Windows certificate store) automatically; alternatively set `REQUESTS_CA_BUNDLE` to a bundle that
  includes the root. `curl` may need `--ssl-no-revoke` on such networks.
- `python` on PATH older than 3.11: create the venv with `py -3.11 -m venv .venv` or the full interpreter path.
- Logs: `logs/app.log` (application), `logs/refresh_demo.log` (if you redirect the refresh output there).
