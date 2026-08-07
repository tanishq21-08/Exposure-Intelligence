"""
grounding.py — External grounding of ONE field (address) against a real map.

This is Paper 2's RAG / external-evidence idea at the honest scale: instead of
trusting the model's extracted address, check it against an external source of
truth (OpenStreetMap via Nominatim). Retrieve evidence from the world, not just
from the source document. Like the other layers, it ANNOTATES — it never fixes.

Scope guardrail: ONE field (address), done well. Not every field, not a
retrieval framework. One clean grounding example that demonstrates the pattern.

What it checks per property:
  1. Does the address resolve to a real place at all? (the strongest signal)
  2. Does the resolved postcode agree with the postcode in the extracted address?
  3. What is the canonical address + coordinates the map returns?

Honest limitations (say these out loud):
  - Geocoders GUESS. A no-match is NOT proof the address is fake (OSM may simply
    lack a messy UK commercial address); a match is NOT proof it is correct.
    This is evidence to WEIGH — a signal for triage, not an oracle. Same posture
    as the verification critic.
  - On this dataset the addresses already contain postcodes, so the postcode
    check is partly circular; the real value is the resolve / no-resolve binary
    (it flags "Office & showroom, see broker note", which validation and
    verification also flag — three independent layers converging).

Design notes:
  - Uses Nominatim (free, no API key). Nominatim REQUIRES a descriptive
    User-Agent and asks for <= 1 request/second — both handled below.
  - Fails GRACEFULLY: if the network/service is unavailable, each property gets
    a note and the pipeline still completes; nothing crashes.
  - The network call (geocode) is separated from the pure comparison logic
    (_build_result), so the flagging logic is offline-testable.
"""

import re
import time
import logging
from dataclasses import dataclass

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim bans requests with no/deceptive User-Agent. Put a real contact here.
USER_AGENT = "ExposureIntelligence/1.0 (grounding demo; contact: tanishqgharat1729@gmail.com)"

_POSTCODE_RE = re.compile(r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

@dataclass
class GroundingResult:
    ref: str
    resolved: bool
    extracted_postcode: str | None
    resolved_postcode: str | None
    postcode_match: bool | None    # None when not comparable (no resolve / missing pc)
    canonical: str | None          # Nominatim's cleaned display_name
    lat: str | None
    lon: str | None
    flagged: bool                  # True == worth a human's attention
    note: str


# ---------------------------------------------------------------------------
# Postcode helpers
# ---------------------------------------------------------------------------

def _norm_pc(pc):
    """Normalise a postcode for comparison: upper-case, spaces removed."""
    if not pc:
        return None
    return str(pc).upper().replace(" ", "")


def _extract_postcode(text):
    """Pull a UK postcode out of free text and normalise it, or None."""
    if not text:
        return None
    m = _POSTCODE_RE.search(str(text))
    return _norm_pc(m.group(0)) if m else None


def _ref_for(loc, index):
    addr = loc["address"]["value"]
    if addr and str(addr).strip().upper() != "UNKNOWN":
        return str(addr)
    return f"property #{index + 1}"


# ---------------------------------------------------------------------------
# The network call (geocode) — isolated so the rest stays offline-testable
# ---------------------------------------------------------------------------

def _spaced_postcode(pc_norm):
    """Reformat a normalised postcode ('LS16HW') back to spaced form ('LS1 6HW')
    for the Nominatim query."""
    if not pc_norm or len(pc_norm) < 5:
        return pc_norm
    return f"{pc_norm[:-3]} {pc_norm[-3:]}"


def geocode(address, timeout=10):
    """Ground the address against Nominatim.

    Messy broker descriptions ("Ground floor retail, 145 Briggate, ...") do NOT
    resolve as a full free-text query — the descriptive prefix isn't an
    addressable component, so Nominatim returns nothing. Empirically, even a
    cleaned street query returns a plausible-but-different postcode (145 Briggate
    resolved to LS1 6BR vs the stated LS1 6HW), so full-address grounding of
    commercial SOV addresses is unreliable.

    We therefore ground on the POSTCODE alone: check whether the extracted
    postcode corresponds to a real, locatable UK postcode. This is a deliberately
    MODEST check — it confirms the postcode is real, not that the whole address
    is correct — but it is robust, unlike full-address geocoding on this data.
    Returns the top result dict, or None if there is no postcode or it doesn't
    resolve. Raises on network/service errors (caught by the caller)."""
    postcode = _extract_postcode(address)
    if not postcode:
        return None
    params = {
        "postalcode": _spaced_postcode(postcode),
        "format": "json",
        "addressdetails": 1,
        "limit": 1,
        "countrycodes": "gb",
    }
    resp = requests.get(NOMINATIM_URL, params=params,
                        headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


# ---------------------------------------------------------------------------
# Pure comparison logic (no network) — offline-testable
# ---------------------------------------------------------------------------

def _build_result(ref, extracted_address, geo):
    """Turn a (possibly None) geocode response into a GroundingResult.
    geo is None       -> did not resolve (flagged).
    geo is a dict     -> resolved; compare postcodes where possible."""
    extracted_pc = _extract_postcode(extracted_address)

    if geo is None:
        return GroundingResult(
            ref=ref, resolved=False,
            extracted_postcode=extracted_pc, resolved_postcode=None,
            postcode_match=None, canonical=None, lat=None, lon=None,
            flagged=True,
            note="address did not resolve to a known location",
        )

    resolved_pc = _norm_pc(geo.get("address", {}).get("postcode")) \
        or _extract_postcode(geo.get("display_name", ""))
    canonical = geo.get("display_name")
    lat, lon = geo.get("lat"), geo.get("lon")

    if extracted_pc and resolved_pc:
        match = (extracted_pc == resolved_pc)
    else:
        match = None

    if match is True:
        note = "resolved; postcode matches"
        flagged = False
    elif match is False:
        note = f"resolved but postcode differs (extracted {extracted_pc} vs resolved {resolved_pc})"
        flagged = True
    else:
        note = "resolved; postcode not comparable"
        flagged = False

    return GroundingResult(
        ref=ref, resolved=True,
        extracted_postcode=extracted_pc, resolved_postcode=resolved_pc,
        postcode_match=match, canonical=canonical, lat=lat, lon=lon,
        flagged=flagged, note=note,
    )


# ---------------------------------------------------------------------------
# Per-property + portfolio drivers
# ---------------------------------------------------------------------------

def ground_property(loc, index=0):
    """Ground one property's address. Never raises — on any service error it
    returns a resolved=False result with an explanatory note, so the pipeline
    keeps going."""
    ref = _ref_for(loc, index)
    address = loc["address"]["value"]

    try:
        geo = geocode(address)
    except Exception as e:
        logger.warning(f"geocoding unavailable for {ref!r}: {e}")
        return GroundingResult(
            ref=ref, resolved=False,
            extracted_postcode=_extract_postcode(address), resolved_postcode=None,
            postcode_match=None, canonical=None, lat=None, lon=None,
            flagged=False,  # a service outage is not the DATA's fault — don't flag the address
            note=f"geocoding unavailable: {e}",
        )

    return _build_result(ref, address, geo)


def ground_portfolio(consolidated, delay=1.0):
    """Ground every property's address. Sleeps `delay` seconds between calls to
    respect Nominatim's ~1 req/sec policy. Returns a list of GroundingResults."""
    results = []
    for i, loc in enumerate(consolidated):
        results.append(ground_property(loc, i))
        if i < len(consolidated) - 1:
            time.sleep(delay)
    return results


def summarize(results):
    resolved   = sum(1 for r in results if r.resolved)
    unresolved = sum(1 for r in results if not r.resolved)
    mismatches = sum(1 for r in results if r.postcode_match is False)
    flagged    = sum(1 for r in results if r.flagged)
    return {"resolved": resolved, "unresolved": unresolved,
            "postcode_mismatches": mismatches, "flagged": flagged,
            "total": len(results)}


# ---------------------------------------------------------------------------
# Offline self-test — python src/grounding.py   (NO network call)
#
# Exercises the pure comparison logic with fake Nominatim responses, covering
# the three cases: postcode match, postcode mismatch, and no-resolve. The real
# run happens through main.py and does hit the network.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Case 1: resolves, postcode matches -> not flagged
    geo_match = {
        "display_name": "145, Briggate, Leeds, LS1 6HW, United Kingdom",
        "lat": "53.7965", "lon": "-1.5440",
        "address": {"postcode": "LS1 6HW"},
    }
    r1 = _build_result("Leeds retail", "Ground floor retail, 145 Briggate, Leeds, LS1 6HW", geo_match)

    # Case 2: resolves, postcode DIFFERS -> flagged
    geo_mismatch = {
        "display_name": "Somewhere else, Leeds, LS1 9ZZ, United Kingdom",
        "lat": "53.80", "lon": "-1.55",
        "address": {"postcode": "LS1 9ZZ"},
    }
    r2 = _build_result("Leeds retail", "Ground floor retail, 145 Briggate, Leeds, LS1 6HW", geo_mismatch)

    # Case 3: did not resolve -> flagged
    r3 = _build_result("Office & showroom", "Office & showroom, see broker note", None)

    for r in (r1, r2, r3):
        star = "  <-- FLAGGED" if r.flagged else ""
        print(f"[resolved={r.resolved}] {r.ref} :: {r.note}{star}")

    print()
    print(summarize([r1, r2, r3]))