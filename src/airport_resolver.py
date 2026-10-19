"""Airport name resolution, alias generation and lightweight phrase parsing.

Resolution order: explicit code -> alias table -> city (+state) -> city dominance rule -> fuzzy.
Ambiguity is reported, never guessed away.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

from src.models import NEW_ENGLAND_STATES, REGIONS, AirportRef, ResolveResult

# --------------------------------------------------------------------------------------
# Curated aliases (lower-case phrase -> IATA code). Extend freely; auto aliases cover the rest.
# --------------------------------------------------------------------------------------
CURATED_ALIASES: dict[str, str] = {
    "los angeles airport": "LAX", "los angeles": "LAX", "los angeles international": "LAX", "lax airport": "LAX",
    "santa ana airport": "SNA", "santa ana": "SNA", "john wayne": "SNA", "john wayne airport": "SNA", "orange county": "SNA",
    "orange county airport": "SNA",
    "anchorage airport": "ANC", "anchorage": "ANC", "ted stevens": "ANC", "ted stevens anchorage": "ANC",
    "san francisco airport": "SFO", "san francisco": "SFO", "san francisco international": "SFO",
    "boston logan": "BOS", "logan": "BOS", "logan airport": "BOS", "boston": "BOS", "boston logan international": "BOS",
    "sea-tac": "SEA", "seatac": "SEA", "sea tac": "SEA", "seattle": "SEA", "seattle tacoma": "SEA",
    "o'hare": "ORD", "ohare": "ORD", "chicago o'hare": "ORD", "chicago ohare": "ORD", "midway": "MDW", "chicago midway": "MDW",
    "dulles": "IAD", "washington dulles": "IAD", "reagan": "DCA", "reagan national": "DCA", "washington national": "DCA",
    "laguardia": "LGA", "la guardia": "LGA", "kennedy": "JFK", "jfk airport": "JFK", "newark": "EWR", "newark liberty": "EWR",
    "hobby": "HOU", "houston hobby": "HOU", "bush intercontinental": "IAH", "houston intercontinental": "IAH",
    "love field": "DAL", "dallas love field": "DAL", "dallas fort worth": "DFW", "dallas/fort worth": "DFW", "dfw airport": "DFW",
    "sky harbor": "PHX", "phoenix sky harbor": "PHX", "phoenix": "PHX",
    "harry reid": "LAS", "mccarran": "LAS", "las vegas": "LAS", "las vegas airport": "LAS",
    "denver": "DEN", "denver airport": "DEN", "denver international": "DEN",
    "bradley": "BDL", "bradley international": "BDL", "hartford": "BDL",
    "t.f. green": "PVD", "tf green": "PVD", "t f green": "PVD", "providence": "PVD",
    "portland jetport": "PWM", "portland maine": "PWM", "manchester boston": "MHT", "manchester-boston": "MHT",
    "burlington vermont": "BTV", "austin bergstrom": "AUS", "austin": "AUS", "bergstrom": "AUS",
    "hartsfield": "ATL", "hartsfield jackson": "ATL", "atlanta": "ATL",
    "minneapolis": "MSP", "minneapolis st paul": "MSP", "msp airport": "MSP",
    "salt lake": "SLC", "salt lake city": "SLC", "san diego": "SAN", "lindbergh": "SAN",
    "fort lauderdale": "FLL", "miami": "MIA", "orlando": "MCO", "tampa": "TPA", "charlotte": "CLT", "philadelphia": "PHL",
    "detroit": "DTW", "detroit metro": "DTW", "baltimore": "BWI", "bwi airport": "BWI", "nashville": "BNA", "raleigh durham": "RDU",
    "raleigh": "RDU", "st louis": "STL", "saint louis": "STL", "lambert": "STL", "kansas city": "MCI", "new orleans": "MSY",
    "san jose": "SJC", "oakland": "OAK", "sacramento": "SMF", "honolulu": "HNL", "san antonio": "SAT", "el paso": "ELP",
    "albuquerque": "ABQ", "tucson": "TUS", "pittsburgh": "PIT", "cleveland": "CLE", "columbus": "CMH", "indianapolis": "IND",
    "cincinnati": "CVG", "milwaukee": "MKE", "omaha": "OMA", "memphis": "MEM", "louisville": "SDF", "jacksonville": "JAX",
    "richmond": "RIC", "norfolk": "ORF", "buffalo": "BUF", "rochester": "ROC", "albany": "ALB", "syracuse": "SYR",
    "fairbanks": "FAI", "juneau": "JNU", "boise": "BOI", "spokane": "GEG", "reno": "RNO", "ontario": "ONT", "burbank": "BUR",
    "long beach": "LGB", "palm springs": "PSP", "fresno": "FAT", "santa barbara": "SBA",
}

STOPWORDS = {"international", "intl", "airport", "regional", "municipal", "field", "metropolitan", "national", "airfield",
             "airpark", "county", "memorial", "the", "of", "at"}

US_STATES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington state": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", "puerto rico": "PR",
}
STATE_NAMES: dict[str, str] = {}
for _name, _code in US_STATES.items():
    STATE_NAMES.setdefault(_code, _name.title())
STATE_NAMES["DC"] = "District of Columbia"

SIZE_WORDS: list[tuple[str, list[str]]] = [
    (r"\b(medium[- ]sized?|mid[- ]sized?|medium[- ]hub|medium)\b", ["medium"]),
    (r"\b(large[- ]hub|large|major|big|biggest|largest)\b", ["large"]),
    (r"\b(small[- ]hub|small|smaller)\b", ["small"]),
    (r"\b(non[- ]?hub)\b", ["non_hub"]),
    (r"\b(regional)\b", ["small", "non_hub"]),
]

CITY_DOMINANCE_SHARE = 0.90
FUZZY_RESOLVE = 0.88
FUZZY_CANDIDATE = 0.80
FUZZY_SUGGEST = 0.60


# --------------------------------------------------------------------------------------
# Normalization and alias generation
# --------------------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\bst\.", "saint", text)
    text = re.sub(r"\bft\.", "fort", text)
    text = re.sub(r"\bintl\b", "international", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_stopwords(text: str) -> str:
    return " ".join(t for t in text.split() if t not in STOPWORDS)


def generate_aliases(code: str, name: str, city: str, state: str, curated: Optional[dict[str, str]] = None) -> list[str]:
    """Deterministic alias list for one airport (lower-case, normalized)."""
    curated = CURATED_ALIASES if curated is None else curated
    out: set[str] = set()
    code_l = code.lower()
    out.add(code_l)
    out.add(f"{code_l} airport")
    nname = normalize(name)
    if nname:
        out.add(nname)
        stripped = _strip_stopwords(nname)
        if stripped:
            out.add(stripped)
        for part in re.split(r"[-/()]", name):
            p = normalize(part)
            if p:
                out.add(p)
                ps = _strip_stopwords(p)
                if ps and len(ps) >= 4:
                    out.add(ps)
    ncity = normalize(city)
    if ncity:
        out.add(ncity)
        out.add(f"{ncity} airport")
        out.add(f"{ncity} international")
        out.add(f"{ncity} international airport")
        if state:
            out.add(f"{ncity} {state.lower()}")
            out.add(f"{ncity} {normalize(STATE_NAMES.get(state.upper(), ''))}".strip())
    for phrase, target in curated.items():
        if target == code:
            out.add(normalize(phrase))
    out.discard("")
    return sorted(out)


# --------------------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------------------

@dataclass
class _Entry:
    ref: AirportRef
    aliases: set[str]


class AirportResolver:
    def __init__(self, airports: Iterable[AirportRef], aliases_by_code: Optional[dict[str, list[str]]] = None):
        self.by_code: dict[str, AirportRef] = {}
        self.alias_index: dict[str, set[str]] = {}
        self.city_index: dict[str, set[str]] = {}
        for ref in airports:
            code = ref.airport_code.upper()
            self.by_code[code] = ref
            aliases = set(aliases_by_code.get(code, []) if aliases_by_code else [])
            aliases |= set(generate_aliases(code, ref.airport_name, ref.city, ref.state))
            for alias in aliases:
                self.alias_index.setdefault(normalize(alias), set()).add(code)
            ncity = normalize(ref.city)
            if ncity:
                self.city_index.setdefault(ncity, set()).add(code)
        # Curated aliases pointing at airports that exist in this universe
        for phrase, code in CURATED_ALIASES.items():
            if code in self.by_code:
                self.alias_index.setdefault(normalize(phrase), set()).add(code)
        self._alias_keys = sorted(self.alias_index)

    # ---- public API ------------------------------------------------------------------
    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> "AirportResolver":
        refs, aliases = [], {}
        for r in rows:
            ref = AirportRef(
                airport_code=str(r["airport_code"]).upper(),
                airport_name=r.get("airport_name") or str(r["airport_code"]),
                city=r.get("city") or "",
                state=r.get("state") or "",
                size_band=r.get("size_band") or None,
                has_traffic_data=bool(r.get("has_traffic_data")),
                has_operations_data=bool(r.get("has_operations_data")),
                latest_12m_passengers=int(r.get("latest_12m_passengers") or 0),
            )
            refs.append(ref)
            raw_aliases = r.get("aliases")
            if raw_aliases:
                try:
                    aliases[ref.airport_code] = list(json.loads(raw_aliases)) if isinstance(raw_aliases, str) else list(raw_aliases)
                except (TypeError, ValueError):
                    pass
        return cls(refs, aliases)

    def get(self, code: str) -> Optional[AirportRef]:
        return self.by_code.get(code.upper())

    def resolve(self, name: str) -> ResolveResult:
        query = str(name or "").strip()
        if not query:
            return ResolveResult(status="unknown", query=query, note="Empty airport name.")

        # 1. explicit code anywhere in the phrase (e.g. "Santa Ana Airport, SNA")
        codes = [t for t in re.findall(r"\b[A-Za-z]{3}\b", query) if t.upper() in self.by_code and (t.isupper() or len(query) == 3)]
        if codes:
            code = codes[0].upper()
            note = None
            name_part = re.sub(r"\b[A-Za-z]{3}\b", " ", query) if len(query.strip()) > 3 else ""
            alias_hit = self._alias_lookup(name_part) if normalize(name_part) else set()
            if alias_hit and code not in alias_hit:
                note = f"Name resolved to {','.join(sorted(alias_hit))} but code {code} was given; the code wins."
            return ResolveResult(status="resolved", query=query, airport=self.by_code[code], match_type="code", note=note)

        nq = normalize(query)
        # 2. alias exact
        hits = self._alias_lookup(query)
        if hits:
            return self._decide(query, hits, "alias")

        # 3. city + state ("Portland, ME" / "Portland Maine")
        m = re.match(r"^(.*?)[\s,]+([a-z]{2}|[a-z ]+)$", nq)
        if m:
            city_part, state_part = m.group(1).strip(), m.group(2).strip()
            state_code = state_part.upper() if len(state_part) == 2 else US_STATES.get(state_part)
            if state_code and city_part in self.city_index:
                in_state = {c for c in self.city_index[city_part] if self.by_code[c].state.upper() == state_code}
                if in_state:
                    return self._decide(query, in_state, "city")

        # 4. city alone
        if nq in self.city_index:
            return self._decide(query, set(self.city_index[nq]), "city")

        # 5. fuzzy
        close = difflib.get_close_matches(nq, self._alias_keys, n=6, cutoff=FUZZY_CANDIDATE)
        if close:
            scored: list[tuple[float, str, str]] = []
            for key in close:
                ratio = difflib.SequenceMatcher(None, nq, key).ratio()
                for code in self.alias_index[key]:
                    scored.append((ratio, key, code))
            scored.sort(key=lambda t: (-t[0], t[2]))
            best_ratio, _, best_code = scored[0]
            other = [s for s in scored if s[2] != best_code]
            margin_ok = not other or (best_ratio - other[0][0]) >= 0.05
            if best_ratio >= FUZZY_RESOLVE and margin_ok:
                return ResolveResult(status="resolved", query=query, airport=self.by_code[best_code], match_type="fuzzy", score=round(best_ratio, 3))
            cands = self._sorted_refs({s[2] for s in scored if best_ratio - s[0] < 0.05})
            if len(cands) > 1:
                return ResolveResult(status="ambiguous", query=query, candidates=cands, score=round(best_ratio, 3),
                                     note=f"'{query}' is close to several airports.")
            return ResolveResult(status="unknown", query=query, candidates=cands, score=round(best_ratio, 3),
                                 note=f"No exact match for '{query}'. Did you mean {cands[0].display}?" if cands else None)

        suggestions = difflib.get_close_matches(nq, self._alias_keys, n=3, cutoff=FUZZY_SUGGEST)
        cands = self._sorted_refs({c for k in suggestions for c in self.alias_index[k]})[:3]
        note = f"No airport matching '{query}'."
        if cands:
            note += " Did you mean " + " or ".join(c.display for c in cands) + "?"
        return ResolveResult(status="unknown", query=query, candidates=cands, note=note)

    def extract_airports(self, text: str, exclude_spans: Iterable[tuple[int, int]] = ()) -> list[ResolveResult]:
        """Find airport mentions in free text (codes first, then longest alias n-grams)."""
        spans = list(exclude_spans)
        found: list[tuple[int, ResolveResult]] = []
        seen_codes: set[str] = set()

        def covered(a: int, b: int) -> bool:
            return any(not (b <= s or a >= e) for s, e in spans)

        for m in re.finditer(r"\b([A-Z]{3})\b", text):
            code = m.group(1)
            if code in self.by_code and not covered(m.start(), m.end()):
                spans.append((m.start(), m.end()))
                if code not in seen_codes:
                    seen_codes.add(code)
                    found.append((m.start(), ResolveResult(status="resolved", query=code, airport=self.by_code[code], match_type="code")))

        tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z.'&-]*", text)]
        i = 0
        while i < len(tokens):
            matched = False
            for n in range(4, 0, -1):
                if i + n > len(tokens):
                    continue
                start, end = tokens[i][1], tokens[i + n - 1][2]
                if covered(start, end):
                    continue
                phrase = normalize(text[start:end])
                if not phrase or (n == 1 and len(phrase) < 4):
                    continue
                hits = self._alias_lookup(phrase) or (set(self.city_index[phrase]) if phrase in self.city_index else set())
                if hits:
                    res = self._decide(text[start:end], hits, "alias")
                    code = res.airport.airport_code if res.airport else None
                    if code is None or code not in seen_codes:
                        if code:
                            seen_codes.add(code)
                        found.append((start, res))
                    spans.append((start, end))
                    i += n
                    matched = True
                    break
            if not matched:
                i += 1
        found.sort(key=lambda t: t[0])
        return [r for _, r in found]

    # ---- internals -------------------------------------------------------------------
    def _alias_lookup(self, phrase: str) -> set[str]:
        return set(self.alias_index.get(normalize(phrase), set()))

    def _sorted_refs(self, codes: Iterable[str]) -> list[AirportRef]:
        return sorted((self.by_code[c] for c in codes), key=lambda r: (-r.latest_12m_passengers, r.airport_code))

    def _decide(self, query: str, codes: set[str], match_type: str) -> ResolveResult:
        refs = self._sorted_refs(codes)
        if len(refs) == 1:
            return ResolveResult(status="resolved", query=query, airport=refs[0], match_type=match_type)  # type: ignore[arg-type]
        total = sum(r.latest_12m_passengers for r in refs)
        if total > 0 and refs[0].latest_12m_passengers / total >= CITY_DOMINANCE_SHARE:
            others = ", ".join(r.airport_code for r in refs[1:])
            return ResolveResult(status="resolved", query=query, airport=refs[0], candidates=refs, match_type="city_dominant",
                                 note=f"Resolved to the dominant airport {refs[0].airport_code}; other matches: {others}.")
        return ResolveResult(status="ambiguous", query=query, candidates=refs,
                             note=f"'{query}' matches {len(refs)} airports: " + ", ".join(r.display for r in refs) + ".")


# --------------------------------------------------------------------------------------
# Phrase parsing shared by the router
# --------------------------------------------------------------------------------------

@dataclass
class GeoParse:
    region: Optional[str]
    states: Optional[list[str]]
    spans: list[tuple[int, int]]


def parse_geo(text: str) -> GeoParse:
    spans: list[tuple[int, int]] = []
    region = None
    states: list[str] = []
    for m in re.finditer(r"\bnew[\s-]+england\b", text, re.I):
        region = "new_england"
        spans.append((m.start(), m.end()))
    lowered = text.lower()
    for name in sorted(US_STATES, key=len, reverse=True):
        if name == "washington state":
            pattern = r"\bwashington\s+state\b"
        else:
            pattern = r"\b" + re.escape(name) + r"\b"
        for m in re.finditer(pattern, lowered):
            if any(not (m.end() <= s or m.start() >= e) for s, e in spans):
                continue
            code = US_STATES[name]
            if code not in states:
                states.append(code)
            spans.append((m.start(), m.end()))
    for m in re.finditer(r"\b(?:in|across|within|for|of)\s+([A-Z]{2})\b", text):
        code = m.group(1)
        if code in STATE_NAMES and code not in states:
            states.append(code)
            spans.append((m.start(1), m.end(1)))
    if region and not states:
        return GeoParse(region=region, states=None, spans=spans)
    return GeoParse(region=region, states=states or None, spans=spans)


def parse_size_words(text: str) -> tuple[Optional[list[str]], list[tuple[int, int]]]:
    bands: list[str] = []
    spans: list[tuple[int, int]] = []
    for pattern, mapped in SIZE_WORDS:
        for m in re.finditer(pattern, text, re.I):
            # 'large' inside 'largest demand' etc. is still a size cue only when it precedes 'airport(s)' or 'hub(s)'
            tail = text[m.end(): m.end() + 40].lower()
            if mapped == ["large"] and not re.match(r"[\s-]*(hub|airport|airports|hubs|and|,|-sized)", tail):
                continue
            for b in mapped:
                if b not in bands:
                    bands.append(b)
            spans.append((m.start(), m.end()))
    return (bands or None), spans


def region_states(region: Optional[str]) -> Optional[list[str]]:
    if not region:
        return None
    return list(REGIONS.get(region, ()))


def describe_states(states: Optional[Iterable[str]]) -> str:
    if not states:
        return ""
    return ", ".join(STATE_NAMES.get(s, s) for s in states)


__all__ = [
    "AirportResolver", "CURATED_ALIASES", "US_STATES", "STATE_NAMES", "NEW_ENGLAND_STATES", "generate_aliases", "normalize",
    "parse_geo", "parse_size_words", "region_states", "describe_states", "GeoParse",
]
