from ingestion import sheet_to_text
from confidence import extract_with_consistency
from calibration import load_ground_truth, calibrate
from config import config
from validator import validate, summarize as validation_summary
from verification import verify_portfolio, summarize as verification_summary
from grounding import ground_portfolio, summarize as grounding_summary

path = config["data_path"]

if __name__ == "__main__":
    text = sheet_to_text(path, "Broker B - Castlegate")
    consolidated = extract_with_consistency(text, n=5)   # cached -> free/instant on repeat runs

    # --- deterministic validation pass (pure code, no API) ---
    # "is this value possible?" — impossible/out-of-vocab values
    issues = validate(consolidated)
    print("\n=== Validation ===")
    print(validation_summary(issues))
    for issue in issues:
        print(f"[{issue.severity}] {issue.ref} :: {issue.field} — {issue.message}")

    # --- verification / critic pass (one API call per property) ---
    # "does the source actually support this value?" — catches coherent-but-wrong.
    # Needs the SOURCE TEXT (the same dump extraction saw), not `consolidated`.
    flags = verify_portfolio(text, consolidated)
    print("\n=== Verification (critic) ===")
    print(verification_summary(flags))
    for f in flags:
        star = "  <-- HIGH-CONF CONFLICT" if f.high_conf_conflict else ""
        print(f"[{f.judgment}] {f.ref} :: {f.field} (conf {f.confidence}) — {f.reason}{star}")

    # --- grounding pass (external geocoding, one field: address) ---
    # "does this address resolve to a real place in the world?" — checks the
    # extracted address against OpenStreetMap, not just against the source.
    grounded = ground_portfolio(consolidated)
    print("\n=== Grounding (geocoding) ===")
    print(grounding_summary(grounded))
    for r in grounded:
        star = "  <-- FLAGGED" if r.flagged else ""
        print(f"[resolved={r.resolved}] {r.ref} — {r.note}{star}")

    # --- calibration (compares to ground truth) ---
    # "is this value correct?" — the measurement harness
    truth = load_ground_truth(path)
    truth_B = {k: v for k, v in truth.items() if k[0] == "B"}
    calibrate(consolidated, truth_B, "Broker B")