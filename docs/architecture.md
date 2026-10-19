# Architecture

## Goal and boundaries

One bounded conversational analytics agent for a U.S. airport-modernization investment firm. It screens airports for
sustained demand growth, high scheduled-capacity utilization and operational pressure. It is deliberately **not** a
distributed system: no FastAPI, Docker, cloud infrastructure, authentication, RAG, vector database, MCP server,
multi-agent orchestration or external database.

```text
Browser
  -> Streamlit application
      -> one LLM analytics agent (Anthropic | Gemini | Demo deterministic mode)
          -> allow-listed typed analytical tools
              -> deterministic Python and DuckDB calculations
                  -> local DuckDB historical data store
                  -> optional live aviation REST API adapter
```

## Modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit entry: sidebar state, chat loop, error boundary (no stack traces reach the user) |
| `src/ui_components.py`, `src/charts.py` | tables, one Plotly chart per answer, methodology panel, limitations, evidence block |
| `src/agent.py` | backend-neutral tool-calling loop, system prompt, grounding guard, session-context update |
| `src/llm_backends.py` | `AnthropicBackend` (default), `GeminiBackend` (alternative), provider selection |
| `src/demo_router.py` | rule-based parse -> route -> tool -> templated conclusion (no-key mode and LLM-failure fallback) |
| `src/tools.py` | the eight typed tools, registry, `execute_tool`, JSON-schema export (strict Anthropic, Gemini variant) |
| `src/models.py` | Pydantic contracts: tool inputs, `ToolResult` envelope, payloads, `SessionContext`, `RunDefaults`, `AgentTurn` |
| `src/analytics.py` | coverage detection, periods, raw KPIs, route mix, monthly series, score snapshot |
| `src/scoring.py` | pure math: normalization, scores, exclusions, confidence, methodology text |
| `src/airport_resolver.py` | names/aliases/cities -> IATA codes with explicit ambiguity |
| `src/data_ingestion.py`, `scripts/refresh_data.py` | BTS adapters, validators, aggregation into DuckDB, metadata |
| `src/db.py` | schema, connections, metadata, atomic replacement |
| `src/live_provider.py` | `LiveFlightProvider` protocol, Aviationstack adapter, null provider |
| `src/config.py` | settings from environment/.env, logging |

## Historical and live data are separate

- **Historical** (source of truth for rankings, growth, utilization, capacity pressure): BTS T-100 Segment (passengers,
  seats, departures, distance) and BTS On-Time Performance (flights, 15-minute delays, cancellations), aggregated to
  monthly airport and route tables at refresh time. The runtime queries only these aggregates.
- **Live** (optional, current status only): Aviationstack `/v1/flights` for one airport. It is never written to the
  historical tables, never influences a score, and is rendered under its own `Current Operational Context` heading.
  Missing key, rate limits and provider errors map to explicit `unavailable` / `error` statuses.

## DuckDB schema (`data/airport_intelligence.duckdb`)

| Table | Grain | Key columns |
|---|---|---|
| `airports` | airport | code, name, city, state, region, size_band, aliases, airport_type, has_* flags, latest_12m_passengers |
| `airport_monthly_traffic` | month x airport | passengers, available_seats, departures_performed/scheduled, load_factor |
| `airport_monthly_operations` | month x airport | total_flights, operated_flights, delayed_departures_15m, delay_rate, cancelled_flights, cancellation_rate, dep_delay_minutes_sum, avg_departure_delay_minutes |
| `route_monthly_metrics` | month x origin x destination | distance_miles, passengers, available_seats, departures_performed, is_long_haul |
| `airport_scores` | airport x period x peer group | raw KPIs, component scores, expansion score, data_quality_status (snapshot written at refresh) |
| `data_refresh_metadata` | source x refresh | scope, coverage, refreshed_at, record_count, status, notes |

The refresh builds a temporary database and atomically replaces the live file; the app opens read-only cursors.

## Ingestion pipeline

1. Determine the month range (`--start`, `--end` or the latest On-Time month found on the BTS server).
2. T-100 Segment (All Carriers): one TranStats form postback per calendar year; keep U.S. origins and scheduled
   service (class F); aggregate by month x origin (traffic) and month x origin x destination (routes).
3. On-Time Performance: static monthly zips downloaded with four workers, aggregated by month x origin; validated for
   columns, period, airport codes, rate ranges and national volume.
4. OurAirports reference joined on IATA code; FAA-style size bands from the latest 12 months of origin passengers;
   aliases generated for the resolver.
5. Metadata row per source (`ok` / `partial` / `failed` with the error); default score snapshot; atomic replace.

## Analytics and scoring

- Period = two adjacent rolling 12-month windows ending at a chosen month (calendar years are the `-12` case). The
  default end month is the latest month that both traffic and operations cover; requests beyond coverage are clamped
  and noted; requests before coverage return `insufficient_data`.
- Raw KPIs are window sums (conditional aggregation in DuckDB) with ratios computed in Python; zero denominators
  yield `None` with a reason code.
- Peer group = scope x filters x coverage rule. Components are winsorized (p5/p95) and min-max normalized to 0-100;
  the Expansion Opportunity Score is the 40/35/25 weighted sum; the Capacity Pressure Proxy uses Hazen percentile
  ranks with 50/30/20 weights. Ties break on airport code; results are byte-deterministic.
- Confidence is derived from coverage, months present, peer-group size, data-quality flags and period clamping.

## Tools and contracts

Every tool returns a `ToolResult` with `status`, `data_scope`, `data_period`, `sources`, `refresh_timestamp`,
`definitions`, `metrics`, `limitations`, `confidence`, plus a typed payload for the UI. `execute_tool` validates
inputs with Pydantic and never raises. Tools: `resolve_airport`, `get_airport_profile`, `rank_airports`,
`compare_airports`, `analyze_route_mix`, `analyze_capacity_pressure`, `get_live_operational_context`, `get_methodology`.

## Agent behaviour

- With an LLM key: the model classifies the question, calls approved tools (at most 4 iterations / 6 calls), and
  explains only tool output. The system prompt forbids new arithmetic, causal or ROI claims, and scores for non-ok
  statuses; follow-ups reuse the `SessionContext` block (airport codes, comparison set, filters, period, last tool).
- Grounding guard: numbers in the answer must appear in tool results; an answer with numbers but no tool call becomes
  a clarification; unverified figures are flagged.
- Without a key: the deterministic router parses period, threshold, size words, geography, metric hints, screening
  conditions and airports, maps the intent to a tool, merges follow-ups onto the last tool input, and renders a
  templated conclusion. The same renderer is the fallback when an LLM call fails after tools ran.

## Failure and integrity behaviour

| Condition | Behaviour |
|---|---|
| Unknown airport | suggestions or a clarification question |
| Ambiguous airport | candidate list, no guess |
| Missing historical coverage | coverage gap explained, no score |
| Insufficient data | `insufficient_data`, excluded from ranking with the reason |
| Missing live key | historical works; live says unavailable |
| Live API failure | separate error state; historical unchanged |
| Invalid follow-up | one concise clarification |
| Internal error | safe message in the UI; details in `logs/app.log` |
