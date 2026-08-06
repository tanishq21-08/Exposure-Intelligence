import json
from ingestion import sheet_to_text
from confidence import extract_with_consistency
from calibration import load_ground_truth, calibrate

path = "data/Exposure_SOV_practice.xlsx"

if __name__ == "__main__":
    # ---- EXTRACTION (API calls) — uncomment to regenerate the JSON ----
    # text = sheet_to_text(path, "Broker A - Meridian")
    # consolidated = extract_with_consistency(text, n=5)
    # with open("outputs/Broker_A_consistency.json", "w") as f:
    #     json.dump(consolidated, f, indent=2)

    # ---- CALIBRATION (no API calls) — reads saved JSON ----
    truth = load_ground_truth(path)
    truth_A = {k: v for k, v in truth.items() if k[0] == "A"}
    with open("outputs/Broker_A_consistency.json") as f:
        consolidated = json.load(f)

    calibrate(consolidated, truth_A, "Broker A")