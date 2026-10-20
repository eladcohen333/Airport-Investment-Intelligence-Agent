"""Phase 4: airport name resolution, alias generation and phrase parsing (fixture airports)."""
from __future__ import annotations

import pytest

from src import airport_resolver as ar
from src.analytics import airport_rows
from src.models import AirportRef


@pytest.fixture(scope="module")
def resolver(fixture_conn):
    return ar.AirportResolver.from_rows(airport_rows(fixture_conn))


@pytest.mark.parametrize(
    "query,code,match",
    [
        ("Los Angeles Airport", "LAX", "alias"),
        ("Santa Ana Airport", "SNA", "alias"),
        ("Anchorage Airport", "ANC", "alias"),
        ("San Francisco Airport", "SFO", "alias"),
        ("John Wayne", "SNA", "alias"),
        ("Boston Logan", "BOS", "alias"),
        ("Logan", "BOS", "alias"),
        ("LAX", "LAX", "code"),
        ("sfo", "SFO", "code"),
        ("Santa Ana Airport, SNA", "SNA", "code"),
        ("Portland, ME", "PWM", "alias"),
        ("Portland Maine", "PWM", "alias"),
        ("Denver", "DEN", "alias"),
        ("Austin-Bergstrom International Airport", "AUS", "alias"),
    ],
)
def test_resolves_brief_examples(resolver, query, code, match):
    res = resolver.resolve(query)
    assert res.status == "resolved", res
    assert res.airport.airport_code == code
    assert res.match_type == match


def test_code_wins_over_conflicting_name(resolver):
    res = resolver.resolve("Boston Logan, LAX")
    assert res.status == "resolved" and res.airport.airport_code == "LAX" and "code wins" in res.note


def test_ambiguous_cities_return_candidates(resolver):
    res = resolver.resolve("Washington")
    assert res.status == "ambiguous"
    assert {c.airport_code for c in res.candidates} == {"DCA", "IAD"}
    assert res.airport is None and "matches 2 airports" in res.note
    houston = resolver.resolve("Houston")
    assert houston.status == "ambiguous" and {c.airport_code for c in houston.candidates} == {"IAH", "HOU"}
    dallas = resolver.resolve("Dallas")
    assert dallas.status == "ambiguous" and {c.airport_code for c in dallas.candidates} == {"DFW", "DAL"}


def test_city_dominance_rule():
    big = AirportRef(airport_code="BIG", airport_name="Big City International Airport", city="Twin Town", state="ZZ", latest_12m_passengers=9_500_000)
    small = AirportRef(airport_code="SML", airport_name="Twin Town Municipal Airport", city="Twin Town", state="ZZ", latest_12m_passengers=400_000)
    r = ar.AirportResolver([big, small])
    res = r.resolve("Twin Town")
    assert res.status == "resolved" and res.airport.airport_code == "BIG" and res.match_type == "city_dominant"
    assert "SML" in res.note
    even = ar.AirportResolver([big.model_copy(update={"latest_12m_passengers": 5_000_000}), small.model_copy(update={"latest_12m_passengers": 4_000_000})])
    assert even.resolve("Twin Town").status == "ambiguous"


def test_fuzzy_and_unknown(resolver):
    res = resolver.resolve("Los Angelos")
    assert res.status == "resolved" and res.airport.airport_code == "LAX" and res.match_type == "fuzzy" and res.score >= 0.9
    unknown = resolver.resolve("Atlantis Spaceport")
    assert unknown.status == "unknown" and unknown.airport is None
    assert "No airport matching" in unknown.note
    assert resolver.resolve("").status == "unknown"


def test_extract_airports_from_sentences(resolver):
    found = resolver.extract_airports("Compare Denver, Phoenix, and Las Vegas.")
    assert [r.airport.airport_code for r in found] == ["DEN", "PHX", "LAS"]
    found = resolver.extract_airports("Compare LAX and Santa Ana Airport, SNA, congestion levels.")
    assert [r.airport.airport_code for r in found] == ["LAX", "SNA"]
    found = resolver.extract_airports("What percentage of departures from Anchorage, ANC, are long-haul?")
    assert [r.airport.airport_code for r in found] == ["ANC"]
    assert resolver.extract_airports("Which airports in New England are strong candidates for terminal expansion?") == []
    found = resolver.extract_airports("What changed at Boston Logan in 2024?")
    assert [r.airport.airport_code for r in found] == ["BOS"]
    washington = resolver.extract_airports("How congested is Washington?")
    assert len(washington) == 1 and washington[0].status == "ambiguous"


def test_extract_airports_respects_excluded_spans(resolver):
    text = "Which medium-sized airports in Texas have the strongest demand growth?"
    geo = ar.parse_geo(text)
    assert resolver.extract_airports(text, geo.spans) == []


def test_parse_geo():
    g = ar.parse_geo("Which airports in New England are strong candidates?")
    assert g.region == "new_england" and g.states is None and g.spans
    assert ar.parse_geo("medium-sized airports in Texas").states == ["TX"]
    assert ar.parse_geo("airports in MA and Rhode Island").states == ["RI", "MA"] or ar.parse_geo("airports in MA and Rhode Island").states == ["MA", "RI"]
    assert ar.parse_geo("delays in Oregon or Maine").states == ["OR", "ME"]
    assert ar.parse_geo("Compare LAX and SNA congestion").states is None
    assert ar.parse_geo("What about Washington state?").states == ["WA"]
    assert ar.parse_geo("airports in Washington").states == ["WA"] or ar.parse_geo("airports in Washington").states is None


def test_parse_size_words():
    assert ar.parse_size_words("Which medium-sized airports in Texas")[0] == ["medium"]
    assert ar.parse_size_words("small airports with growth")[0] == ["small"]
    assert ar.parse_size_words("large hubs")[0] == ["large"]
    assert ar.parse_size_words("regional airports")[0] == ["small", "non_hub"]
    assert ar.parse_size_words("the largest demand growth")[0] is None
    assert ar.parse_size_words("Compare LAX and SNA")[0] is None


def test_generic_name_parts_never_become_aliases():
    # "Montgomery Regional (Dannelly Field) Airport" splits on the parentheses into a bare " Airport" part
    mgm = ar.generate_aliases("MGM", "Montgomery Regional (Dannelly Field) Airport", "Montgomery", "AL")
    assert "airport" not in mgm and "dannelly field" in mgm and "montgomery" in mgm
    assert "international airport" not in ar.generate_aliases("ZZZ", "Zed City (International Airport)", "Zed City", "ZZ")


def test_stored_generic_aliases_are_ignored_at_load_time():
    # stores built by earlier refreshes already hold the bare "airport" alias; the resolver must not index it
    mgm = AirportRef(airport_code="MGM", airport_name="Montgomery Regional (Dannelly Field) Airport", city="Montgomery", state="AL", latest_12m_passengers=900_000)
    sfo = AirportRef(airport_code="SFO", airport_name="San Francisco International Airport", city="San Francisco", state="CA", latest_12m_passengers=25_000_000)
    r = ar.AirportResolver([mgm, sfo], {"MGM": ["airport", "montgomery", "mgm"]})
    assert r.resolve("airport").status == "unknown"
    found = r.extract_airports("What is the unmet flight demand in SFO airport and why?")
    assert [x.airport.airport_code for x in found] == ["SFO"]


def test_brief_wording_la_and_santa_ana(resolver):
    found = resolver.extract_airports("Compare LA and Santa Ana airport congestion levels.")
    assert [r.airport.airport_code for r in found] == ["LAX", "SNA"]
    found = resolver.extract_airports("Compare L.A. and SNA.")
    assert [r.airport.airport_code for r in found] == ["LAX", "SNA"]
    assert resolver.resolve("LA").airport.airport_code == "LAX"
    # lower-case "la" is an ordinary word, and "in LA" is the state of Louisiana (excluded span), not an airport
    assert resolver.extract_airports("the la brea area") == []
    text = "Which airports in LA have the strongest growth?"
    geo = ar.parse_geo(text)
    assert geo.states == ["LA"]
    assert resolver.extract_airports(text, geo.spans) == []


def test_generate_aliases_and_normalize():
    aliases = ar.generate_aliases("STL", "St. Louis Lambert International Airport", "St. Louis", "MO")
    assert "saint louis" in aliases and "saint louis lambert" in aliases and "stl airport" in aliases and "lambert" in aliases
    assert ar.normalize("Dallas/Fort Worth Intl.") == "dallas fort worth international"
    assert ar.normalize("  Sea-Tac  ") == "sea tac"
    sna = ar.generate_aliases("SNA", "John Wayne Airport", "Santa Ana", "CA")
    assert "john wayne" in sna and "santa ana airport" in sna and "orange county" in sna      # curated alias merged
