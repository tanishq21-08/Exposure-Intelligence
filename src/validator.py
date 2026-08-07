"""
validator.py — Deterministic validation rules for a consolidated portfolio.

This is a PURE-CODE layer: no LLM, no API calls. It inspects the already-
consolidated output of extract_with_consistency() and returns a list of
ValidationIssue objects. It never mutates or "fixes" the data — it only
annotates, exactly like the confidence layer does.

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

INPUT SHAPE (what this module consumes):
  A list of consolidated location dicts, as produced by extract_with_consistency.
  Each location is a dict keyed by field name; each field is itself a dict:
      loc["year_built"] == {"value": ..., "confidence": ..., "agreement": ...,
                            "all_values": [...]}
  So a field's value is reached as  loc["year_built"]["value"].
  Numeric values arrive as STRINGS (e.g. "2050", "2,500,000") because the
  voting step stringifies them; a genuinely-missing value arrives as None.
  The helpers below (_get / _to_num / _is_unknown) absorb all of that.
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
#  config.yaml — but you SHOULD add them, so the domain assumptions live in
#  config and not buried in code. See note at the bottom of this file.)
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
# Shape accessor + small helpers
# ---------------------------------------------------------------------------

def _get(loc, field):
    """Read a field's value out of a consolidated location dict.

    This is the SINGLE place that knows the consolidated shape
    ({"value": ..., "confidence": ..., ...}). If that shape ever changes,
    only this function changes — every rule below stays untouched."""
    return loc[field]["value"]


def _is_unknown(value) -> bool:
    """UNKNOWN and None are legitimate 'we don't know' answers, not errors.
    Numeric/vocab rules must skip these rather than flag them."""
    return value is None or (isinstance(value, str) and value.strip().upper() == "UNKNOWN")


def _to_num(value):
    """Coerce a value to a number. Returns None if it can't be parsed.
    Consolidated numeric values arrive as strings (possibly with commas / £),
    so this does the cleaning the voting step's str() made necessary."""
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
    addr = _get(loc, "address")
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
        ("construction", _get(loc, "construction"), CONSTRUCTION_VOCAB),
        ("occupancy",    _get(loc, "occupancy"),    OCCUPANCY_VOCAB),
        ("sprinklered",  _get(loc, "sprinklered"),  SPRINKLERED_VOCAB),
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
    yb = _get(loc, "year_built")
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
    tiv = _get(loc, "tiv_gbp")
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
    st = _get(loc, "storeys")
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
            # Positive but implausible -> warning, not a hard error.
            issues.append(ValidationIssue(
                ref, "storeys", "warning",
                f"{int(n)} storeys is unusually high (> {STOREYS_MAX}); worth a check", st))

    # --- floor_area_sqft ---
    fa = _get(loc, "floor_area_sqft")
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
    addr = _get(loc, "address")

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

def validate(locations):
    """Run every deterministic rule over every property and return a flat list
    of ValidationIssue objects. Empty list == everything passed.

    `locations` is the list returned by extract_with_consistency — a list of
    consolidated location dicts (see INPUT SHAPE note at the top of this file).
    All shape knowledge is confined to _get(); the rules never touch the layout
    directly."""
    issues = []
    for i, loc in enumerate(locations):
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
# The fake locations below are plain dicts in the SAME shape that
# extract_with_consistency produces (each field a {"value": ...} dict), so the
# validator runs against realistic input with no schema import and no API call.
# Numeric values are strings on purpose — that's how the voting step stores them.
# Each field in `broken` is deliberately wrong to trip a specific rule.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def field(v):
        # mirrors the consolidated field shape
        return {"value": v, "confidence": 1.0, "agreement": "5/5", "all_values": []}

    broken = {
        "address":         field("12"),            # too short AND no postcode -> 2 warnings
        "tiv_gbp":         field("-5"),            # negative -> error
        "construction":    field("Concrete"),      # not in vocab -> error
        "occupancy":       field("Office"),        # valid
        "year_built":      field("2050"),          # future -> error
        "storeys":         field("0"),             # non-positive -> error
        "floor_area_sqft": field("1000"),          # valid
        "sprinklered":     field("Maybe"),         # not in vocab -> error
    }

    clean = {
        "address":         field("12 King Street, Bristol BS1 4EF"),
        "tiv_gbp":         field("2,500,000"),     # comma string -> _to_num handles it
        "construction":    field("Steel Frame"),
        "occupancy":       field("Warehouse/Storage"),
        "year_built":      field("1998"),
        "storeys":         field("3"),
        "floor_area_sqft": field("45000"),
        "sprinklered":     field("Y"),
    }

    found = validate([broken, clean])
    print(f"\n{summarize(found)}\n")
    for issue in found:
        print(f"[{issue.severity:7}] {issue.ref} :: {issue.field} — {issue.message}")