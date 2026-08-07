"""
validator.py — Deterministic validation rules for extracted portfolios.

This is a PURE-CODE layer: no LLM, no API calls. It inspects an already-extracted
Portfolio and returns a list of ValidationIssue objects. It never mutates or
"fixes" the data — it only annotates, exactly like the confidence layer does.

Why this layer exists (the point to be able to defend out loud):
A field can be HIGH-CONFIDENCE and STILL be wrong. If all 5 self-consistency
samples agree that year_built = 2050, confidence is 1.0 — but the value is
obviously wrong. The confidence layer can't catch that, because it only measures
agreement between samples, not correctness. These deterministic rules catch a
slice of those "coherent-but-wrong" errors that confidence is blind to.

Severity has two levels, and the split is a real design decision:
  - "error"   = definitely wrong, a human must look (future year, negative TIV,
                a value outside the controlled vocabulary).
  - "warning" = suspicious but possibly fine (incomplete-looking address, an
                unusually high storey count). Flag it, don't fail it.
This mirrors how an underwriter triages a submission: hard-stop problems vs.
things worth a second glance.
"""

import re
from dataclasses import dataclass
from datetime import datetime

from config import config


# ---------------------------------------------------------------------------
# Issue structure
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    ref: str        # which property the issue belongs to (address, or index fallback)
    field: str      # which field failed
    severity: str   # "error" or "warning"
    message: str    # human-readable description
    value: object   # the offending value, kept for the audit trail


# ---------------------------------------------------------------------------
# Config-driven thresholds
# (.get() with defaults so this module runs even before you add the keys to
#  config.yaml — but you SHOULD add them, so the assumptions live in config
#  and not buried in code. See note at the bottom of this file.)
# ---------------------------------------------------------------------------

YEAR_MIN     = config.get("year_min", 1600)      # older than this ~= extraction error
STOREYS_MAX  = config.get("storeys_max", 200)    # taller than this ~= extraction error

CONSTRUCTION_VOCAB = config.get(
    "construction_vocab",
    ["Steel Frame", "Reinforced Concrete", "Masonry", "Timber Frame", "UNKNOWN"],
)
OCCUPANCY_VOCAB = config.get(
    "occupancy_vocab",
    ["Warehouse/Storage", "Office", "Retail", "Restaurant",
     "Manufacturing", "Mixed/Other", "UNKNOWN"],
)
SPRINKLERED_VOCAB = config.get("sprinklered_vocab", ["Y", "N", "UNKNOWN"])

# Loose UK postcode pattern — good enough to detect "there is a postcode here",
# not a strict validator. Matches e.g. SW1A 1AA, M1 1AE, B33 8TH, CR2 6XH.
_POSTCODE_RE = re.compile(r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_unknown(value) -> bool:
    """UNKNOWN and None are legitimate 'we don't know' answers, not errors.
    Numeric/vocab rules must skip these rather than flag them."""
    return value is None or (isinstance(value, str) and value.strip().upper() == "UNKNOWN")


def _to_num(value):
    """Coerce a value to a number. Returns None if it can't be parsed.
    Handles stray commas / £ signs defensively, though structured extraction
    should already hand us clean numbers."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("£", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _ref_for(loc, index: int) -> str:
    """Identify a property for the issue report. Prefer its address; fall back
    to a positional label if the address is missing/unknown."""
    addr = getattr(loc.address, "value", None)
    if addr and not _is_unknown(addr):
        return str(addr)
    return f"property #{index + 1}"


# ---------------------------------------------------------------------------
# Rule groups
# ---------------------------------------------------------------------------

def _check_vocab(loc, ref):
    """Each controlled-vocabulary field must hold an allowed value.

    These rules SHOULD rarely fire, because extraction already constrains the
    model to these vocabularies. That's precisely why they belong here: a rule
    that rarely fires is still a cheap guarantee for the day the model drifts
    off-schema (e.g. returns 'Concrete' instead of 'Reinforced Concrete')."""
    issues = []
    checks = [
        ("construction", loc.construction.value, CONSTRUCTION_VOCAB),
        ("occupancy",    loc.occupancy.value,    OCCUPANCY_VOCAB),
        ("sprinklered",  loc.sprinklered.value,  SPRINKLERED_VOCAB),
    ]
    for field, value, vocab in checks:
        # UNKNOWN is a member of every vocab, so it passes naturally.
        if value not in vocab:
            issues.append(ValidationIssue(
                ref=ref, field=field, severity="error",
                message=f"value {value!r} is not in the allowed vocabulary {vocab}",
                value=value,
            ))
    return issues


def _check_numeric(loc, ref):
    """Numeric fields must fall in a physically possible range. UNKNOWN values
    are skipped (an unknown year is not an invalid year)."""
    issues = []
    current_year = datetime.now().year

    # --- year_built ---
    yb = loc.year_built.value
    if not _is_unknown(yb):
        n = _to_num(yb)
        if n is None:
            issues.append(ValidationIssue(
                ref, "year_built", "error",
                f"expected a number, got un-parseable value {yb!r}", yb))
        elif n > current_year:
            issues.append(ValidationIssue(
                ref, "year_built", "error",
                f"year {int(n)} is in the future (current year {current_year})", yb))
        elif n < YEAR_MIN:
            issues.append(ValidationIssue(
                ref, "year_built", "error",
                f"year {int(n)} is before {YEAR_MIN}; almost certainly an extraction error", yb))

    # --- tiv_gbp ---
    tiv = loc.tiv_gbp.value
    if not _is_unknown(tiv):
        n = _to_num(tiv)
        if n is None:
            issues.append(ValidationIssue(
                ref, "tiv_gbp", "error",
                f"expected a number, got un-parseable value {tiv!r}", tiv))
        elif n <= 0:
            issues.append(ValidationIssue(
                ref, "tiv_gbp", "error",
                f"TIV must be positive, got {n}", tiv))

    # --- storeys ---
    st = loc.storeys.value
    if not _is_unknown(st):
        n = _to_num(st)
        if n is None:
            issues.append(ValidationIssue(
                ref, "storeys", "error",
                f"expected a number, got un-parseable value {st!r}", st))
        elif n <= 0:
            issues.append(ValidationIssue(
                ref, "storeys", "error",
                f"storeys must be positive, got {n}", st))
        elif n > STOREYS_MAX:
            # Positive but implausible → warning, not a hard error.
            issues.append(ValidationIssue(
                ref, "storeys", "warning",
                f"{int(n)} storeys is unusually high (> {STOREYS_MAX}); worth a check", st))

    # --- floor_area_sqft ---
    fa = loc.floor_area_sqft.value
    if not _is_unknown(fa):
        n = _to_num(fa)
        if n is None:
            issues.append(ValidationIssue(
                ref, "floor_area_sqft", "error",
                f"expected a number, got un-parseable value {fa!r}", fa))
        elif n <= 0:
            issues.append(ValidationIssue(
                ref, "floor_area_sqft", "error",
                f"floor area must be positive, got {n}", fa))

    return issues


def _check_completeness(loc, ref):
    """Softer heuristics on the address. These are WARNINGS by design — an
    address with no detectable postcode might still be usable, but it's worth
    a human glance."""
    issues = []
    addr = loc.address.value

    if _is_unknown(addr):
        issues.append(ValidationIssue(
            ref, "address", "warning", "address is UNKNOWN", addr))
        return issues  # nothing more to check on an unknown address

    text = str(addr).strip()
    if len(text) < 8:
        issues.append(ValidationIssue(
            ref, "address", "warning",
            f"address {text!r} is suspiciously short; may be incomplete", addr))
    if not _POSTCODE_RE.search(text):
        issues.append(ValidationIssue(
            ref, "address", "warning",
            "no UK postcode detected in address; may be incomplete", addr))

    return issues


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def validate(portfolio):
    """Run every deterministic rule over every property and return a flat list
    of ValidationIssue objects. Empty list == everything passed.

    NOTE ON SCHEMA ASSUMPTIONS (the one thing to confirm against schema.py):
      - portfolio.locations  is the list of properties
      - each property exposes .address, .tiv_gbp, .construction, .occupancy,
        .year_built, .storeys, .floor_area_sqft, .sprinklered
      - each of those is an object with a .value attribute
    If your container attribute is named differently (e.g. .properties), change
    it in this one function."""
    issues = []
    for i, loc in enumerate(portfolio.locations):
        ref = _ref_for(loc, i)
        issues.extend(_check_vocab(loc, ref))
        issues.extend(_check_numeric(loc, ref))
        issues.extend(_check_completeness(loc, ref))
    return issues


def summarize(issues):
    """Convenience: counts by severity, for a quick line in main.py output."""
    errors   = sum(1 for x in issues if x.severity == "error")
    warnings = sum(1 for x in issues if x.severity == "warning")
    return {"errors": errors, "warnings": warnings, "total": len(issues)}


# ---------------------------------------------------------------------------
# Self-test — runs with:  python src/validator.py   (from the project root)
#
# It uses SimpleNamespace stand-ins instead of your real Pydantic objects.
# This works BECAUSE the validator only depends on the *shape* (a .value on
# each field), not on the exact classes — so a lightweight fake is a valid
# stand-in, and you can test every rule with no schema import and no API call.
# Each field below is deliberately broken to trip a specific rule.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from types import SimpleNamespace as N

    def field(v):
        return N(value=v, confidence=1.0, source="self-test", type="verbatim")

    broken = N(
        address=field("12"),                 # too short AND no postcode -> 2 warnings
        tiv_gbp=field(-5),                    # negative -> error
        construction=field("Concrete"),      # not in vocab (should be "Reinforced Concrete") -> error
        occupancy=field("Office"),           # valid
        year_built=field(2050),              # future -> error
        storeys=field(0),                    # non-positive -> error
        floor_area_sqft=field(1000),         # valid
        sprinklered=field("Maybe"),          # not in vocab -> error
    )

    clean = N(
        address=field("12 King Street, Bristol BS1 4EF"),
        tiv_gbp=field(2_500_000),
        construction=field("Steel Frame"),
        occupancy=field("Warehouse/Storage"),
        year_built=field(1998),
        storeys=field(3),
        floor_area_sqft=field(45_000),
        sprinklered=field("Y"),
    )

    portfolio = N(locations=[broken, clean])

    found = validate(portfolio)
    print(f"\n{summarize(found)}\n")
    for issue in found:
        print(f"[{issue.severity:7}] {issue.ref} :: {issue.field} — {issue.message}")