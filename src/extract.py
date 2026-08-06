import pandas as pd
import os, json
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter, defaultdict

path = "data/Exposure_SOV_practice.xlsx"

load_dotenv()
client = OpenAI()

# ============ SCHEMA ============
class FieldType(str, Enum):
    verbatim = "verbatim"
    derived = "derived"

class TextField(BaseModel):
    value: str
    confidence: float
    source: str
    type: FieldType

class NumField(BaseModel):
    value: Optional[float]
    confidence: float
    source: str
    type: FieldType

class Location(BaseModel):
    address: TextField
    tiv_gbp: NumField
    construction: TextField
    occupancy: TextField
    year_built: NumField
    storeys: NumField
    floor_area_sqft: NumField
    sprinklered: TextField

class Portfolio(BaseModel):
    locations: list[Location]

# ============ STAGE 1: read sheet to text ============
def sheet_to_text(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    return df.to_string(index=False, na_rep="")

# ============ STAGE 3: prompt + extraction ============
SYSTEM = """You extract commercial-property exposure data from messy broker Statements of Value.You are expert in this field and you realise that even a small error and cause a magnificent loss to the insurance companies so you are super careful while you fill in the values.

Return one entry per property location. For EVERY field, provide four things:
- value: the normalized answer
- confidence: your certainty from 0.0 to 1.0
- source: the exact text from the document you used
- type: "verbatim" if copied directly from the text, "derived" if you inferred, converted, or guessed it

Rules:
- construction must be one of: Steel Frame, Reinforced Concrete, Masonry, Timber Frame, UNKNOWN
- occupancy must be one of: Warehouse/Storage, Office, Retail, Restaurant, Manufacturing, Mixed/Other, UNKNOWN
- sprinklered must be one of: Y, N, UNKNOWN
- Convert values to standard form: '£2.4m' -> 2400000 (pure numbers, currency already defined). Area must end up in square feet: read the ORIGINAL unit from the header/context; if already sq ft keep it (verbatim); if in another unit (sqm, etc.) convert to sq ft and mark derived, keeping the original in source.
- type: "verbatim" ONLY if character-for-character identical to source. Any typo fix, vocab mapping, or conversion is "derived".
- For the address: if clearly incomplete or references elsewhere ("see broker note", "TBC"), return what's there but set confidence low (~0.3) and type derived.
- If a value is genuinely missing, set value to null (numbers) or "UNKNOWN" (construction/occupancy/sprinklered), and lower confidence. NEVER invent a value."""

def extract(text, temperature=0):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Statement of Values:\n\n{text}"},
        ],
        response_format=Portfolio,
    )
    return completion.choices[0].message.parsed

# ============ CONFIDENCE LAYER (self-consistency) ============
def extract_with_consistency(text, n=5, temperature=0.7):
    runs = [extract(text, temperature=temperature) for _ in range(n)]
    num_locations = len(runs[0].locations)
    field_names = ["address", "tiv_gbp", "construction", "occupancy",
                   "year_built", "storeys", "floor_area_sqft", "sprinklered"]
    consolidated = []
    for i in range(num_locations):
        location_result = {}
        for field in field_names:
            values = [str(getattr(run.locations[i], field).value) for run in runs]
            most_common, count = Counter(values).most_common(1)[0]
            final_value = None if most_common == "None" else most_common
            location_result[field] = {
                "value": final_value,
                "confidence": round(count / n, 2),
                "agreement": f"{count}/{n}",
                "all_values": values,
            }
        consolidated.append(location_result)
    return consolidated

# ============ CALIBRATION ============
def _clean_num(x):
    if x is None: return None
    s = str(x).strip().replace(",", "").replace("£", "")
    if s.upper() in ("UNKNOWN", "", "NAN", "NONE"): return None
    try: return float(s)
    except ValueError: return None

def load_ground_truth(path):
    gt = pd.read_excel(path, sheet_name="GROUND TRUTH (do not peek)", header=2)
    gt = gt[gt["Source"].isin(["A", "B"])]
    truth = {}
    for _, row in gt.iterrows():
        key = (str(row["Source"]).strip(), str(row["Ref"]).strip())
        truth[key] = {
            "address":         row["Normalized Address"],
            "tiv_gbp":         row["TIV (£)"],
            "construction":    row["Construction (normalized)"],
            "occupancy":       row["Occupancy (normalized)"],
            "year_built":      row["Year Built"],
            "storeys":         row["Storeys"],
            "floor_area_sqft": row["Floor Area (sq ft)"],
            "sprinklered":     row["Sprinklered"],
        }
    return truth

def is_correct(field, predicted, truth):
    truth_str = str(truth).strip().upper() if truth is not None else ""
    pred_str  = str(predicted).strip().upper() if predicted is not None else ""
    truth_is_unknown = truth_str in ("UNKNOWN", "", "NAN", "NONE")
    pred_is_unknown  = pred_str  in ("UNKNOWN", "", "NAN", "NONE")
    if truth_is_unknown:
        return pred_is_unknown
    if pred_is_unknown:
        return False
    numeric_fields = {"tiv_gbp", "year_built", "floor_area_sqft", "storeys"}
    if field in numeric_fields:
        p, t = _clean_num(predicted), _clean_num(truth)
        if p is None or t is None: return False
        if t == 0: return p == 0
        tol = 0.02 if field == "floor_area_sqft" else 0.0
        return abs(p - t) <= abs(t) * tol or p == t
    return pred_str == truth_str

def calibrate(consolidated, truth_subset, source_label):
    fields = ["tiv_gbp", "construction", "occupancy", "year_built", "floor_area_sqft", "sprinklered"]
    records = []
    truth_rows = list(truth_subset.values())
    for i, loc in enumerate(consolidated):
        if i >= len(truth_rows): break
        gt = truth_rows[i]
        for field in fields:
            pred = loc[field]["value"]
            conf = loc[field]["confidence"]
            records.append((conf, is_correct(field, pred, gt.get(field))))
    buckets = defaultdict(list)
    for conf, correct in records:
        buckets[round(conf, 1)].append(correct)
    print(f"\n=== Calibration: {source_label} ===")
    print(f"{'Confidence':>10} | {'Accuracy':>8} | {'Count':>5}")
    ece, total = 0.0, len(records)
    for conf in sorted(buckets):
        outcomes = buckets[conf]
        acc = sum(outcomes) / len(outcomes)
        print(f"{conf:>10.1f} | {acc:>8.2f} | {len(outcomes):>5}")
        ece += (len(outcomes) / total) * abs(conf - acc)
    print(f"\nECE: {ece:.3f}   (lower = more honest)")
    return ece

# ============ RUN ============
if __name__ == "__main__":
    # ---- STEP 1: EXTRACTION (makes API calls) — run ONCE to create the JSON, then COMMENT OUT ----
    # text = sheet_to_text(path, "Broker A - Meridian")
    # consolidated = extract_with_consistency(text, n=5)
    # with open("outputs/Broker_A_consistency.json", "w") as f:
    #     json.dump(consolidated, f, indent=2)
    # print("extraction saved")

    # ---- STEP 2: CALIBRATION (no API calls) — reads the saved JSON ----
    truth = load_ground_truth(path)
    truth_A = {k: v for k, v in truth.items() if k[0] == "A"}
    with open("outputs/Broker_A_consistency.json") as f:
        consolidated = json.load(f)

    print(f"Loaded {len(consolidated)} vs truth {len(truth_A)}")
    for i, gt in enumerate(truth_A.values()):
        if i < len(consolidated):
            print(f"[{i}] pred: {str(consolidated[i]['address']['value'])[:35]!r}  truth: {str(gt['address'])[:35]!r}")

    calibrate(consolidated, truth_A, "Broker A")