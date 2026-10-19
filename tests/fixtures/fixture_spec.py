"""FIXTURE DATA ONLY - synthetic constants used to generate tests/fixtures/fixture_*.csv.

Nothing here is real BTS data. Values are chosen so that expected KPIs are hand-computable:
  * monthly passengers are constant within a calendar year and grow by G24 / G25 between years,
    so calendar-2024 demand growth == G24 exactly;
  * seats = round(passengers / LF), so utilization ~= LF;
  * flights per month F, cancelled = F * C (integer), operated = F - cancelled,
    delayed = operated * D (integer), so delay_rate == D and cancellation_rate == C exactly;
  * dep_delay_minutes_sum = delayed * 30 + (operated - delayed) * 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AirportSpec:
    code: str
    name: str
    city: str
    state: str
    size_band: str
    pax_2023: int          # monthly passengers in 2023
    g24: float             # growth 2024 vs 2023
    g25: float             # growth 2025 vs 2024
    lf: float              # load factor (utilization)
    flights: int           # monthly scheduled flights (OTP)
    delay_rate: float      # D (constant unless overridden per year)
    cancel_rate: float     # C
    per_year_ops: dict = field(default_factory=dict)   # year -> (delay_rate, cancel_rate)
    aliases: tuple = ()
    airport_type: str = "large_airport"


AIRPORTS: list[AirportSpec] = [
    AirportSpec("LAX", "Los Angeles International Airport", "Los Angeles", "CA", "large", 2_500_000, 0.04, 0.02, 0.85, 20_000, 0.20, 0.02,
                aliases=("los angeles", "los angeles airport", "los angeles international")),
    AirportSpec("SNA", "John Wayne Airport", "Santa Ana", "CA", "medium", 450_000, 0.06, 0.03, 0.80, 4_000, 0.25, 0.015,
                aliases=("santa ana", "santa ana airport", "john wayne", "orange county"), airport_type="medium_airport"),
    AirportSpec("SFO", "San Francisco International Airport", "San Francisco", "CA", "large", 2_000_000, 0.08, 0.05, 0.88, 16_000, 0.22, 0.025,
                aliases=("san francisco", "san francisco airport")),
    AirportSpec("ANC", "Ted Stevens Anchorage International Airport", "Anchorage", "AK", "medium", 250_000, 0.02, 0.01, 0.75, 3_000, 0.15, 0.01,
                aliases=("anchorage", "anchorage airport", "ted stevens"), airport_type="medium_airport"),
    AirportSpec("BOS", "Boston Logan International Airport", "Boston", "MA", "large", 1_500_000, 0.05, 0.02, 0.84, 12_000, 0.20, 0.0125,
                per_year_ops={2023: (0.1875, 0.016), 2024: (0.20, 0.0125), 2025: (0.20, 0.0125), 2026: (0.20, 0.0125)},
                aliases=("boston", "boston logan", "logan", "logan airport")),
    AirportSpec("PVD", "Rhode Island T. F. Green International Airport", "Providence", "RI", "small", 150_000, 0.03, 0.01, 0.80, 1_500, 0.17, 0.012,
                aliases=("providence", "t.f. green", "tf green"), airport_type="medium_airport"),
    AirportSpec("BDL", "Bradley International Airport", "Hartford", "CT", "medium", 280_000, 0.07, 0.04, 0.82, 2_800, 0.19, 0.014,
                aliases=("hartford", "bradley"), airport_type="medium_airport"),
    AirportSpec("MHT", "Manchester-Boston Regional Airport", "Manchester", "NH", "small", 80_000, -0.02, 0.00, 0.78, 800, 0.16, 0.010,
                aliases=("manchester", "manchester boston"), airport_type="medium_airport"),
    AirportSpec("PWM", "Portland International Jetport", "Portland", "ME", "small", 90_000, 0.09, 0.03, 0.86, 900, 0.14, 0.009,
                aliases=("portland jetport",), airport_type="medium_airport"),
    AirportSpec("BTV", "Burlington International Airport", "Burlington", "VT", "small", 60_000, 0.04, 0.02, 0.79, 600, 0.13, 0.008,
                aliases=("burlington",), airport_type="medium_airport"),
    AirportSpec("AUS", "Austin-Bergstrom International Airport", "Austin", "TX", "medium", 900_000, 0.10, 0.06, 0.83, 7_000, 0.21, 0.018,
                aliases=("austin", "austin bergstrom")),
    AirportSpec("DFW", "Dallas/Fort Worth International Airport", "Dallas", "TX", "large", 3_500_000, 0.03, 0.02, 0.82, 28_000, 0.19, 0.02,
                aliases=("dallas fort worth", "dfw airport")),
    AirportSpec("DAL", "Dallas Love Field", "Dallas", "TX", "medium", 700_000, 0.05, 0.03, 0.81, 5_000, 0.23, 0.02,
                aliases=("love field",), airport_type="medium_airport"),
    AirportSpec("HOU", "William P. Hobby Airport", "Houston", "TX", "medium", 600_000, 0.05, 0.03, 0.81, 5_000, 0.23, 0.02,
                aliases=("hobby",), airport_type="medium_airport"),
    AirportSpec("IAH", "George Bush Intercontinental Airport", "Houston", "TX", "large", 1_900_000, 0.02, 0.01, 0.80, 15_000, 0.20, 0.019,
                aliases=("bush intercontinental",)),
    AirportSpec("DCA", "Ronald Reagan Washington National Airport", "Washington", "DC", "large", 1_100_000, 0.03, 0.02, 0.85, 10_000, 0.17, 0.013,
                aliases=("reagan", "reagan national")),
    AirportSpec("IAD", "Washington Dulles International Airport", "Washington", "VA", "large", 1_000_000, 0.06, 0.04, 0.78, 9_000, 0.18, 0.015,
                aliases=("dulles",)),
    AirportSpec("DEN", "Denver International Airport", "Denver", "CO", "large", 3_200_000, 0.04, 0.03, 0.84, 26_000, 0.21, 0.017,
                aliases=("denver",)),
    AirportSpec("PHX", "Phoenix Sky Harbor International Airport", "Phoenix", "AZ", "large", 2_100_000, 0.05, 0.02, 0.86, 18_000, 0.16, 0.012,
                aliases=("phoenix", "sky harbor")),
    AirportSpec("LAS", "Harry Reid International Airport", "Las Vegas", "NV", "large", 2_300_000, 0.02, 0.01, 0.87, 17_000, 0.19, 0.014,
                aliases=("las vegas", "harry reid", "mccarran")),
    # Synthetic tiny airport that fails the 500-flight coverage rule (30 flights/month = 360 per 12 months).
    AirportSpec("TNY", "Tinytown Municipal Airport (fixture)", "Tinytown", "MA", "non_hub", 2_000, 0.00, 0.00, 0.60, 30, 0.10, 0.0,
                aliases=("tinytown",), airport_type="small_airport"),
]

# Traffic months: 2023-01 .. 2025-12 (36). Operations months: 2023-01 .. 2026-02 (38).
TRAFFIC_YEARS = (2023, 2024, 2025)
OPS_MONTHS_END = "2026-02"
TRAFFIC_MONTHS_END = "2025-12"
# AUS has no traffic row for this month (exercises months-present counting).
MISSING_TRAFFIC = {("AUS", "2025-11")}

# Routes per origin: (destination, distance_miles, monthly departures, monthly passengers). Same every month.
ROUTES: dict[str, list[tuple[str, float, int, int]]] = {
    "ANC": [
        ("SEA", 1450.0, 140, 16_800),
        ("FAI", 260.0, 100, 7_000),
        ("JNU", 570.0, 60, 4_200),
        ("ORD", 2850.0, 30, 4_500),
        ("HNL", 2780.0, 20, 3_000),
        ("DFW", 3040.0, 10, 1_500),
        ("SLC", 2000.0, 40, 5_000),   # exactly at the default threshold -> counts as long-haul
    ],
    "LAX": [("JFK", 2475.0, 300, 45_000), ("SFO", 337.0, 700, 70_000)],
    "BOS": [("LAX", 2611.0, 100, 15_000), ("DCA", 399.0, 400, 40_000)],
    "AUS": [("SEA", 1770.0, 50, 7_000), ("DFW", 190.0, 450, 40_000)],
}
# Expected: ANC total 400 departures/month, 100 long-haul at >= 2000 mi -> 25.0 %; at >= 1400 mi -> 240/400 = 60 %;
#           at >= 2001 mi -> 60/400 = 15 %.

TRAFFIC_SOURCE = "FIXTURE synthetic T-100-shaped traffic"
OPS_SOURCE = "FIXTURE synthetic OTP-shaped operations"
ROUTE_SOURCE = "FIXTURE synthetic T-100-shaped routes"
REFRESHED_AT = "2026-10-01T00:00:00"


def pax_for_year(spec: AirportSpec, year: int) -> int:
    pax = spec.pax_2023
    if year >= 2024:
        pax = round(pax * (1 + spec.g24))
    if year >= 2025:
        pax = round(pax * (1 + spec.g25))
    return int(pax)


def ops_for_year(spec: AirportSpec, year: int) -> tuple[int, int, int, int, float]:
    """Return (total_flights, cancelled, operated, delayed, dep_delay_minutes_sum) for one month."""
    d, c = spec.per_year_ops.get(year, (spec.delay_rate, spec.cancel_rate))
    total = spec.flights
    cancelled = int(round(total * c))
    operated = total - cancelled
    delayed = int(round(operated * d))
    minutes = delayed * 30.0 + (operated - delayed) * 2.0
    return total, cancelled, operated, delayed, minutes


def spec_by_code(code: str) -> AirportSpec:
    for s in AIRPORTS:
        if s.code == code:
            return s
    raise KeyError(code)
