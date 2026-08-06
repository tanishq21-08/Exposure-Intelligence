import pandas as pd
from collections import defaultdict

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
            records.append((loc[field]["confidence"], is_correct(field, loc[field]["value"], gt.get(field))))
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