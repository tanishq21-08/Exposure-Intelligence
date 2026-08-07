from ingestion import sheet_to_text
from confidence import extract_with_consistency
from config import config

path = config["data_path"]
text = sheet_to_text(path, "Broker B - Castlegate")
consolidated = extract_with_consistency(text, n=5)   # cache hit → no API calls

for loc in consolidated:
    if "Glasgow" in str(loc["address"]["value"]):
        print("tiv:", loc["tiv_gbp"]["value"], "| floor:", loc["floor_area_sqft"]["value"])