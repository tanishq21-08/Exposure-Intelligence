import json
from ingestion import sheet_to_text
from confidence import extract_with_consistency
from calibration import load_ground_truth, calibrate
from config import config

path = config["data_path"]

if __name__ == "__main__":
    text = sheet_to_text(path, "Broker A - Meridian")
    consolidated = extract_with_consistency(text, n=5)   # cached → free/instant on repeat runs

    truth = load_ground_truth(path)
    truth_A = {k: v for k, v in truth.items() if k[0] == "A"}
    calibrate(consolidated, truth_A, "Broker A")