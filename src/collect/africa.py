"""Africa-aware helpers for the collection layer.

Real job feeds are global and messy. For a new graduate in Africa, three things matter
that generic tooling ignores:

  * **Location canonicalisation** for African cities/countries written many ways
    ("Lagos, NG" / "lagos" / "Lagos, Nigeria" -> Lagos, Nigeria).
  * **Remote relevance** — a global-remote role IS reachable from Nairobi, so it should
    not be filtered out as "not local".
  * **Currency awareness** — salaries come in NGN/KES/GHS/ZAR/EGP/USD; a bare number is
    meaningless without the unit.

Keeping this in one place makes the "who is missing from the data?" fairness beat
concrete: we can show which regions the feed under-represents.
"""
from __future__ import annotations

import re

# Canonical "City, Country" for the African hubs this demo cares about, plus the messy
# spellings that map to each. Extend freely — students are invited to add their city.
AFRICA_CITIES = {
    "Nairobi, Kenya": ["nairobi", "nairobi, ke", "nairobi, kenya"],
    "Lagos, Nigeria": ["lagos", "lagos, ng", "lagos, nigeria"],
    "Abuja, Nigeria": ["abuja", "abuja, ng", "abuja, nigeria"],
    "Accra, Ghana": ["accra", "accra, gh", "accra, ghana"],
    "Kampala, Uganda": ["kampala", "kampala, ug", "kampala, uganda"],
    "Kigali, Rwanda": ["kigali", "kigali, rw", "kigali, rwanda"],
    "Dar es Salaam, Tanzania": ["dar es salaam", "dar", "dar es salaam, tz"],
    "Addis Ababa, Ethiopia": ["addis ababa", "addis", "addis ababa, et"],
    "Cairo, Egypt": ["cairo", "cairo, eg", "cairo, egypt"],
    "Casablanca, Morocco": ["casablanca", "casablanca, ma"],
    "Dakar, Senegal": ["dakar", "dakar, sn"],
    "Johannesburg, South Africa": ["johannesburg", "joburg", "jhb",
                                   "johannesburg, za"],
    "Cape Town, South Africa": ["cape town", "cape town, za"],
    "Nakuru, Kenya": ["nakuru"],
    "Mombasa, Kenya": ["mombasa"],
}

# Country names/codes that mark a post as Africa-relevant even without a known city.
AFRICAN_COUNTRIES = {
    "kenya", "ke", "nigeria", "ng", "ghana", "gh", "uganda", "ug", "rwanda", "rw",
    "tanzania", "tz", "ethiopia", "et", "egypt", "eg", "morocco", "ma", "senegal",
    "sn", "south africa", "za", "zambia", "zm", "zimbabwe", "zw", "cameroon", "cm",
    "ivory coast", "côte d'ivoire", "ci", "tunisia", "tn", "angola", "ao",
}

# Region grouping for the fairness / representation readout.
CITY_REGION = {
    "Nairobi, Kenya": "East Africa", "Nakuru, Kenya": "East Africa",
    "Mombasa, Kenya": "East Africa", "Kampala, Uganda": "East Africa",
    "Kigali, Rwanda": "East Africa", "Dar es Salaam, Tanzania": "East Africa",
    "Addis Ababa, Ethiopia": "East Africa",
    "Lagos, Nigeria": "West Africa", "Abuja, Nigeria": "West Africa",
    "Accra, Ghana": "West Africa", "Dakar, Senegal": "West Africa",
    "Cairo, Egypt": "North Africa", "Casablanca, Morocco": "North Africa",
    "Johannesburg, South Africa": "Southern Africa",
    "Cape Town, South Africa": "Southern Africa",
    "Remote": "Remote",
}

# Currency symbols / codes -> ISO code, so a salary number carries its unit.
# Tokens must be specific: no bare single letters (a greedy "r " once tagged
# "junioR Backend" as ZAR — exactly the kind of silent data bug this talk is about).
CURRENCY_TOKENS = {
    "ngn": "NGN", "₦": "NGN", "naira": "NGN",
    "kes": "KES", "ksh": "KES", "shilling": "KES",
    "ghs": "GHS", "gh₵": "GHS", "cedi": "GHS",
    "zar": "ZAR", "rand": "ZAR",
    "egp": "EGP",
    "us$": "USD", "usd": "USD", "dollar": "USD",
    "eur": "EUR", "€": "EUR",
    "$": "USD",
}

_REMOTE_RE = re.compile(r"\b(remote|work from home|wfh|anywhere|distributed team)\b",
                        re.IGNORECASE)
# Phrases that NEGATE remote, so "no remote" / "on-site only" isn't read as remote-friendly.
_NOT_REMOTE_RE = re.compile(r"\b(no remote|not remote|on[- ]?site|in[- ]?office|"
                            r"relocation required)\b", re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def canonical_location(raw: str) -> str:
    """Map a messy location string to a canonical 'City, Country' or 'Remote'/'Unknown'."""
    key = _norm(raw)
    if key == "":
        return "Unknown"
    if _REMOTE_RE.search(key):
        return "Remote"
    for canon, variants in AFRICA_CITIES.items():
        if key in variants or any(key == v or v in key for v in variants):
            return canon
    # Country-only mention -> keep the country, Title-cased, so it still groups.
    for country in AFRICAN_COUNTRIES:
        if country in key and len(country) > 2:
            return country.title()
    return raw.strip().title()


def region_of(canonical_loc: str) -> str:
    """Region bucket for a canonical location (for the representation readout)."""
    if canonical_loc == "Remote":
        return "Remote"
    return CITY_REGION.get(canonical_loc, "Other / non-Africa")


def is_remote(text: str) -> bool:
    """True for remote-friendly text, but False when it's explicitly negated
    ('no remote', 'on-site only') — a common phrasing that fools naive keyword checks."""
    t = str(text or "")
    if _NOT_REMOTE_RE.search(t):
        return False
    return bool(_REMOTE_RE.search(t))


def detect_currency(text: str) -> str | None:
    """Best-effort currency ISO code from free text (or None if not stated)."""
    t = _norm(text)
    # Longer, more specific tokens first so 'us$' wins over '$'.
    for token in sorted(CURRENCY_TOKENS, key=len, reverse=True):
        if token in t:
            return CURRENCY_TOKENS[token]
    return None


def is_africa_relevant(location: str, description: str = "",
                       include_remote: bool = True) -> bool:
    """True if a post is reachable for an Africa-based candidate.

    That means: a known African city/country, OR (when include_remote) a remote role,
    which a Nairobi graduate can genuinely take.
    """
    canon = canonical_location(location)
    if canon == "Remote":
        return include_remote
    if region_of(canon) not in ("Other / non-Africa", "Remote"):
        return True
    # Location field was unhelpful — check the description for a country mention.
    blob = _norm(f"{location} {description}")
    if any(re.search(rf"\b{re.escape(c)}\b", blob) for c in AFRICAN_COUNTRIES
           if len(c) > 2):
        return True
    return include_remote and is_remote(description)
