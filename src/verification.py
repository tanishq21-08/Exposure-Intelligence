"""
verification.py — Critic / verification pass (Paper 2's "verify before you trust").

This is the agentic step, at the honest scale: ONE scoped LLM call per property,
with a narrow adversarial job — re-read the source SOV and judge, per field,
whether the extracted value is actually supported by the source. It does NOT
re-extract and does NOT fix anything; like the validator, it only annotates.

Why this layer exists (the point to defend):
Your confidence layer measures agreement between samples, not correctness — so a
value where all 5 samples agreed but were WRONG gets confidence 1.0 and sails
through. Your deterministic rules catch impossible values, but a wrong-but-
plausible value (in-vocab, in-range) passes them too. An INDEPENDENT second
reader, checking the value against the source, is the only layer that can catch
those "coherent-but-wrong" errors. On Broker B you saw ~3 such fields in the
1.0-confidence bucket. This pass is aimed exactly at them.

Two design decisions that make or break it:
  1. The critic is kept BLIND to the extractor's confidence. If you told it "the
     extractor was 100% sure," it would anchor and rubber-stamp. It judges from
     source + value alone. The cross-reference with confidence happens AFTER,
     in pure code — that's where the "unsupported AND high-confidence" money
     square is computed.
  2. Temperature 0. Unlike the confidence layer (0.7, where you WANT disagreement
     to surface), here you want the critic's single most consistent judgment.

Honest limitation (say this out loud in interviews):
The critic is an LLM too. An "unsupported" verdict is not truth — it is a second
opinion that disagrees. What it buys you is not correctness but a DISAGREEMENT
SIGNAL for triage: it routes human attention to the fields where two independent
passes conflict. This is exactly Paper 2's framing — make the process inspectable
and route uncertain cases to review, not give the model authority.
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

from config import config

load_dotenv()
client = OpenAI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIELD_NAMES = ["address", "tiv_gbp", "construction", "occupancy",
               "year_built", "storeys", "floor_area_sqft", "sprinklered"]


# ---------------------------------------------------------------------------
# Output schema for the critic (structured output, same mechanism as extraction)
# Defined here rather than in schema.py because it's specific to this pass —
# move it to schema.py if you'd rather keep all Pydantic models together.
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    supported = "supported"
    unsupported = "unsupported"
    unclear = "unclear"


class FieldVerdict(BaseModel):
    judgment: Verdict
    reason: str          # one short sentence citing what in the source drove the call


class PropertyVerification(BaseModel):
    address: FieldVerdict
    tiv_gbp: FieldVerdict
    construction: FieldVerdict
    occupancy: FieldVerdict
    year_built: FieldVerdict
    storeys: FieldVerdict
    floor_area_sqft: FieldVerdict
    sprinklered: FieldVerdict


# ---------------------------------------------------------------------------
# Critic prompt
# ---------------------------------------------------------------------------

_CONSTRUCTION = ", ".join(config["construction_vocab"])
_OCCUPANCY    = ", ".join(config["occupancy_vocab"])
_SPRINKLERED  = ", ".join(config["sprinklered_vocab"])

SYSTEM = f"""You are a careful auditor checking data that was extracted from a broker's Statement of Values (SOV). You did NOT perform the extraction. Your job is to independently check it against the source text.

You will be given:
1. The full SOV source text.
2. The extracted values for ONE property.

IMPORTANT — the extraction is NORMALISED, so it will deliberately NOT match the source word-for-word. This is correct behaviour, not an error:
- construction is mapped to one of: {_CONSTRUCTION}
- occupancy is mapped to one of: {_OCCUPANCY}
- sprinklered is mapped to one of: {_SPRINKLERED}
- monetary values are converted to plain numbers (e.g. "£2.4m" -> 2400000)
- floor area is converted to square feet
Mapping a source description onto the closest allowed vocabulary term is the extractor's JOB. For example "Cash & carry", "Storage & distribution", and "trade counter" all correctly map to Warehouse/Storage; "Steel portal frame, brick infill" correctly maps to Steel Frame; "light manufacturing" correctly maps to Manufacturing. A correct mapping like these is SUPPORTED — do NOT flag a value merely because its wording differs from the source.

For EACH of the 8 fields, first locate the matching property in the source, then judge:
- "supported": the value is stated in the source, OR is a correct normalisation / vocabulary-mapping / unit-conversion of what the source states. UNKNOWN is SUPPORTED when the source genuinely does not state that fact. A value of None (for numeric fields like tiv_gbp, year_built, storeys, floor_area_sqft) is the numeric equivalent of UNKNOWN: it means "not stated", and is SUPPORTED whenever the source genuinely does not provide that number. Do NOT flag None as "not a valid value" — declining to invent a number the source lacks is correct.
- "unsupported": the value CONTRADICTS the source, is FABRICATED (appears nowhere), or is an INFERENCE THE SOURCE DOES NOT JUSTIFY. The key distinction: mapping a synonym or description to the vocabulary is fine, but concluding a SPECIFIC fact the source never states is not. Examples of unsupported:
    * Source says only "fire protection: yes" and sprinklered was extracted as Y — general fire protection does not specifically mean sprinklers.
    * The source clearly states a fact but the extractor returned UNKNOWN (a miss).
    * The source describes a genuinely MIXED use (e.g. "office / showroom") but the extractor collapsed it to a single narrower term when a broader term such as Mixed/Other is available.
- "unclear": you genuinely cannot tell from the source (the source is ambiguous, or you cannot confidently locate this property or this field's value within the source). IMPORTANT: if you simply cannot FIND the relevant figure or detail in the source text, that is "unclear", NOT "unsupported". Reserve "unsupported" for cases where the source actively says something DIFFERENT from the extracted value, or where a specific value was clearly invented. Absence-of-evidence is "unclear"; contradiction or fabrication is "unsupported".

For each field give a SHORT reason (one sentence) citing the source text you relied on.

Judge from the source text and the extracted value, ACCOUNTING FOR expected normalisation. Do not assume the extractor was right, but also do not flag correct vocabulary-mapping or unit-conversion as unsupported."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_property(loc):
    """Render ONLY the extracted values for the critic — deliberately excluding
    confidence, so the critic judges independently and can't anchor on it."""
    lines = []
    for field in FIELD_NAMES:
        lines.append(f"- {field}: {loc[field]['value']}")
    return "\n".join(lines)


def _ref_for(loc, index):
    """Identify a property for the flag report. Prefer address; fall back to a
    positional label. (Kept local so this module stays independent of validator.)"""
    addr = loc["address"]["value"]
    if addr and str(addr).strip().upper() != "UNKNOWN":
        return str(addr)
    return f"property #{index + 1}"


# ---------------------------------------------------------------------------
# The critic call (API) — mirrors extraction.py's retry/backoff exactly
# ---------------------------------------------------------------------------

def verify_property(source_text, loc, temperature=0):
    """One critic call for one property. Returns a PropertyVerification."""
    user_content = (
        f"SOV source text:\n\n{source_text}\n\n"
        f"---\n\n"
        f"Extracted values for ONE property:\n\n{_format_property(loc)}"
    )

    max_retries = 3
    base_delay = 1

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.beta.chat.completions.parse(
                model=config["model"],
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                response_format=PropertyVerification,
            )
            return completion.choices[0].message.parsed

        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Verification failed after {max_retries} attempts: {e}")
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"Attempt {attempt} failed ({e}); retrying in {delay}s...")
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Flag aggregation — PURE (no API), so it's offline-testable
# ---------------------------------------------------------------------------

@dataclass
class VerificationFlag:
    ref: str
    field: str
    judgment: str            # "unsupported" or "unclear"
    reason: str
    confidence: float        # the extractor's confidence for this field (cross-referenced)
    high_conf_conflict: bool # True when unsupported AND high confidence — the money square


def _flags_for_property(ref, loc, verdict, flag_threshold):
    """Turn one property's critic verdict into flags, cross-referencing the
    extractor's confidence. 'supported' fields produce nothing. This is where
    the two independent signals meet: critic says unsupported + extractor was
    highly confident == the coherent-but-wrong case worth a human's eyes."""
    flags = []
    for field in FIELD_NAMES:
        fv = getattr(verdict, field)
        if fv.judgment == Verdict.supported:
            continue
        conf = loc[field]["confidence"]
        flags.append(VerificationFlag(
            ref=ref,
            field=field,
            judgment=fv.judgment.value,
            reason=fv.reason,
            confidence=conf,
            high_conf_conflict=(fv.judgment == Verdict.unsupported and conf >= flag_threshold),
        ))
    return flags


def verify_portfolio(source_text, consolidated, flag_threshold=0.8):
    """Run the critic over every property and return a flat list of
    VerificationFlags. One API call per property. Empty list == the critic
    found everything supported."""
    flags = []
    for i, loc in enumerate(consolidated):
        ref = _ref_for(loc, i)
        verdict = verify_property(source_text, loc)
        flags.extend(_flags_for_property(ref, loc, verdict, flag_threshold))
    return flags


def summarize(flags):
    """Counts for a quick line in main.py output."""
    unsupported = sum(1 for f in flags if f.judgment == "unsupported")
    unclear     = sum(1 for f in flags if f.judgment == "unclear")
    money       = sum(1 for f in flags if f.high_conf_conflict)
    return {"unsupported": unsupported, "unclear": unclear,
            "high_conf_conflicts": money, "total": len(flags)}


# ---------------------------------------------------------------------------
# Offline self-test — python src/verification.py  (NO API call)
#
# Exercises the pure flag-aggregation logic with a hand-built verdict, so you can
# confirm the plumbing — especially the high_conf_conflict "money square" — for
# free. The real critic run happens through main.py and does hit the API.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def field(v, c):
        return {"value": v, "confidence": c, "agreement": "", "all_values": []}

    # One property. Note tiv is high-confidence but the critic will call it
    # unsupported -> that's the money square. year_built is unclear.
    loc = {
        "address":         field("12 King Street, Bristol BS1 4EF", 1.0),
        "tiv_gbp":         field("4200000", 1.0),   # extractor sure, critic disagrees
        "construction":    field("Steel Frame", 0.8),
        "occupancy":       field("Warehouse/Storage", 1.0),
        "year_built":      field("1998", 0.6),
        "storeys":         field("3", 1.0),
        "floor_area_sqft": field("45000", 1.0),
        "sprinklered":     field("Y", 0.4),
    }

    verdict = PropertyVerification(
        address=FieldVerdict(judgment=Verdict.supported,   reason="address matches source row"),
        tiv_gbp=FieldVerdict(judgment=Verdict.unsupported, reason="source states £2.4m (=2400000), not 4200000"),
        construction=FieldVerdict(judgment=Verdict.supported, reason="'steel frame' stated in source"),
        occupancy=FieldVerdict(judgment=Verdict.supported, reason="'warehouse' stated in source"),
        year_built=FieldVerdict(judgment=Verdict.unclear,  reason="no build year visible in source for this row"),
        storeys=FieldVerdict(judgment=Verdict.supported,   reason="3 storeys stated"),
        floor_area_sqft=FieldVerdict(judgment=Verdict.supported, reason="45,000 sq ft stated"),
        sprinklered=FieldVerdict(judgment=Verdict.supported, reason="UNKNOWN correct; source silent on sprinklers"),
    )

    flags = _flags_for_property(_ref_for(loc, 0), loc, verdict, flag_threshold=0.8)
    print(f"\n{summarize(flags)}\n")
    for f in flags:
        star = "  <-- HIGH-CONF CONFLICT" if f.high_conf_conflict else ""
        print(f"[{f.judgment:11}] {f.ref} :: {f.field} (conf {f.confidence}) — {f.reason}{star}")