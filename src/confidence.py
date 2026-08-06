from collections import Counter
from extraction import extract        # <-- importing from your own module

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