import pandas as pd
path="data/Exposure_SOV_practice.xlsx"

from enum import Enum
from typing import Optional
from pydantic import BaseModel

class FieldType(str, Enum):
    verbatim = "verbatim"   # copied straight from the document
    derived = "derived"     # inferred, converted, or guessed

class TextField(BaseModel):
    value: str
    confidence: float
    source: str
    type: FieldType

class NumField(BaseModel):
    value: Optional[float]   # None when genuinely missing
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


import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()                 # reads .env into the environment
client = OpenAI()             # picks up OPENAI_API_KEY automatically

# ---- Stage 1: read a sheet to faithful text ----
def sheet_to_text(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    return df.to_string(index=False, na_rep="")

# ---- Stage 3: the prompt (your 3 design decisions live here) ----
SYSTEM = """You extract commercial-property exposure data from messy broker Statements of Value.You are expert in this field and you realise that even a small error and cause a magnificent loss to the insurance companies so you are super careful while you fill in the values.

Return one entry per property location. For EVERY field, provide four things:
- value: the normalized answer
- confidence: your certainty from 0.0 to 1.0
- source: the exact text from the document you used
- type: "verbatim" if copied directly from the text, "derived" if you inferred, converted, or guessed it

Rules:
- construction must be one of: Steel Frame, Reinforced Concrete, Masonry, Timber Frame, UNKNOWN
- occupancy must be one of: Warehouse/Storage, Office, Retail, Restaurant, Manufacturing, Mixed/Other, UNKNOWN
- Convert values to standard form: '£2.4m' -> 2400000 (everything must be in pure numbers, becasue we have already defined the currency unit); Area should be in sqft, convert any other unit to sqft based on the conversion rates.
-You are a perfect person who doesn't neglect spelling mistakes and look for end to end consistency.- type: "verbatim" ONLY if the value is character-for-character identical to the source text. If you corrected a typo, mapped to a controlled-vocabulary value, converted a unit, or inferred anything at all, it is "derived". Examples: source "Masonary" -> value "Masonry" is DERIVED (typo corrected). Source "Warehouse" -> "Warehouse/Storage" is DERIVED (mapped to vocabulary). Source "3200000" -> 3200000 is VERBATIM (identical).
- If a value is genuinely missing, set value to null (for numbers) or "UNKNOWN" (for construction/occupancy), and lower the confidence. NEVER invent a value to fill a gap.
-for the sprinklered part, be consistent, it's either Yes or No or Unknown
-- For the address field: assess whether it is a COMPLETE, usable address (has a street and locality/postcode). If it is clearly incomplete or refers elsewhere (e.g. "see broker note", "TBC", "as above", partial fragments), still return what's there, but set confidence LOW (around 0.3) and type "derived". A complete, verbatim address gets high confidence; an incomplete one must be flagged with low confidence."""

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

from collections import Counter

def extract_with_consistency(text, n=5, temperature=0.7):
    runs = [extract(text, temperature=temperature) for _ in range(n)]   # n samples at higher temp

    num_locations = len(runs[0].locations)
    field_names = ["address", "tiv_gbp", "construction", "occupancy",
                   "year_built", "storeys", "floor_area_sqft", "sprinklered"]

    consolidated = []
    for i in range(num_locations):
        location_result = {}
        for field in field_names:
            values = [str(getattr(run.locations[i], field).value) for run in runs]  # this field across all runs
            most_common, count = Counter(values).most_common(1)[0]
            final_value=None if most_common =="None" else most_common                # majority vote
            location_result[field] = {
                "value": final_value,
                "confidence": round(count / n, 2),   # agreement fraction = REAL confidence
                "agreement": f"{count}/{n}",
                "all_values": values,                # keep so you can SEE the disagreement
            }
        consolidated.append(location_result)
    return consolidated



if __name__ == "__main__":
    path = "data/Exposure_SOV_practice.xlsx"
    text = sheet_to_text(path, "Broker A - Meridian")
    result = extract_with_consistency(text, n=5)
    with open("outputs/Broker_A_consistency.json", "w") as f:
        json.dump(result, f, indent=2)
    print("done -> outputs/Broker_A_consistency.json")

# ---- Stage 4: run on both brokers and save ----
# if __name__ == "__main__":
#     path = "data/Exposure_SOV_practice.xlsx"
#     for sheet in ["Broker A - Meridian", "Broker B - Castlegate"]:
#         text = sheet_to_text(path, sheet)
#         result = extract(text)
#         out = f"outputs/{sheet.split(' - ')[0].replace(' ', '_')}.json"
#         with open(out, "w") as f:
#             json.dump(result.model_dump(), f, indent=2, default=str)
#         print(f"{sheet}: {len(result.locations)} locations -> {out}")



