
import pandas as pd
import numpy as np

# 1. Load the data directly from your file
df = pd.read_csv("data/osrm_matrix.csv")

# 2. Calculate the speed in km/h
# Formula: (Distance in meters / Duration in seconds) * 3.6
# We use np.where to prevent DivisionByZero errors if duration is 0
df['speed_kmph'] = np.where(
    df['osrm_duration_seconds'] > 0,
    (df['osrm_distance_meters'] / df['osrm_duration_seconds']) * 3.6,
    0.0  # Default to 0.0 km/h if duration is 0
)

# Optional: Round the speed to 2 decimal places for cleaner data
df['speed_kmph'] = df['speed_kmph'].round(2)

# 3. Save the updated dataframe to a new file
output_filename = "data/osrm_matrix_with_speed.csv"
df.to_csv(output_filename, index=False)

print(f"Successfully calculated speeds and saved to {output_filename}!\n")

# --- NEW: Calculate and print Max and Min (non-zero) speeds ---
max_speed = df['speed_kmph'].max()
min_nonzero_speed = df[df['speed_kmph'] > 0]['speed_kmph'].min()

print(f"Maximum Speed: {max_speed} km/h")
print(f"Minimum Non-Zero Speed: {min_nonzero_speed} km/h\n")

print("Here is a preview of the new data:")
print(df.head(10).to_string(index=False))