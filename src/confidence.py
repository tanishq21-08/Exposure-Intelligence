from collections import Counter
from extraction import extract       
from config import config
from cache import get_cached, save_cache 

def extract_with_consistency(text, n=None, temperature=None, use_cache=True):
    if n is None: n=config["n_samples"]
    if temperature is None: temperature = config["confidence_temperature"]

    if use_cache:
        cached = get_cached(text, n, temperature)
        if cached is not None:
            print("[cache hit] returning stored result — no API calls")
            return cached

    
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

    if use_cache:
        save_cache(text, n, temperature, consolidated) 
    return consolidated