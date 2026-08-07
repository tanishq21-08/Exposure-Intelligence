import json
from ingestion import sheet_to_text
from confidence import extract_with_consistency
from calibration import load_ground_truth, calibrate
from config import config
from validator import validate, summarize

path = config["data_path"]

if __name__ == "__main__":
    text = sheet_to_text(path, "Broker B - Castlegate")
    consolidated = extract_with_consistency(text, n=5)   # cached → free/instant on repeat runs

    # --- deterministic validation pass ---
    issues = validate(consolidated)
    print(summarize(issues))
    for issue in issues:
        print(f"[{issue.severity}] {issue.ref} :: {issue.field} — {issue.message}")

    truth = load_ground_truth(path)
    truth_B = {k: v for k, v in truth.items() if k[0] == "B"}
    calibrate(consolidated, truth_B, "Broker B")

print("truth keys:", list(truth.keys()))
print("truth_B size:", len(truth_B))