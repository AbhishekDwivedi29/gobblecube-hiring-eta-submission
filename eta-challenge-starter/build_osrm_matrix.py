import pandas as pd
import requests
import itertools
from tqdm import tqdm

# Load the coordinates you already generated
zones = pd.read_csv("data/zone_coords.csv")

# Create every possible combination of Pickup -> Dropoff
zone_ids = zones["LocationID"].tolist()
combinations = list(itertools.product(zone_ids, zone_ids))

results = []

print(f"Querying OSRM for {len(combinations)} zone combinations...")
for pu, do in tqdm(combinations):
    # Get coordinates
    pu_lon = zones.loc[zones["LocationID"] == pu, "lon"].values[0]
    pu_lat = zones.loc[zones["LocationID"] == pu, "lat"].values[0]
    do_lon = zones.loc[zones["LocationID"] == do, "lon"].values[0]
    do_lat = zones.loc[zones["LocationID"] == do, "lat"].values[0]
    
    # OSRM expects coordinates as lon,lat
    url = f"http://localhost:5000/route/v1/driving/{pu_lon},{pu_lat};{do_lon},{do_lat}?overview=false"
    
    try:
        response = requests.get(url).json()
        if response["code"] == "Ok":
            route = response["routes"][0]
            results.append({
                "pickup_zone": pu,
                "dropoff_zone": do,
                "osrm_distance_meters": route["distance"],
                "osrm_duration_seconds": route["duration"]
            })
        else:
            # Fallback if route fails (e.g., islands with no bridges)
            results.append({"pickup_zone": pu, "dropoff_zone": do, "osrm_distance_meters": 0, "osrm_duration_seconds": 0})
    except Exception as e:
        results.append({"pickup_zone": pu, "dropoff_zone": do, "osrm_distance_meters": 0, "osrm_duration_seconds": 0})

# Save the final lookup table!
matrix_df = pd.DataFrame(results)
matrix_df.to_csv("data/osrm_matrix.csv", index=False)
print("Saved OSRM matrix to data/osrm_matrix.csv!")