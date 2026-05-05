import pandas as pd
import geopandas as gpd

# 1. Load the shapefile
gdf = gpd.read_file("data/taxi_zones/taxi_zones.shp")

# 2. Project to a flat local CRS for accurate geometry math 
# (EPSG:2263 is the standard projected CRS for NYC data)
gdf_projected = gdf.to_crs("EPSG:2263")

# 3. Calculate the centroids accurately in the flat projection
centroids = gdf_projected.geometry.centroid

# 4. Convert those center points to standard GPS coordinates (WGS84 / EPSG:4326)
centroids_gps = centroids.to_crs("EPSG:4326")

# 5. Extract the lat/lon values
gdf["lon"] = centroids_gps.x
gdf["lat"] = centroids_gps.y

# 6. Save this as a simple CSV
zone_coords = gdf[["LocationID", "lat", "lon"]].copy()
zone_coords.to_csv("data/zone_coords.csv", index=False)
print("Saved zone coordinates!")