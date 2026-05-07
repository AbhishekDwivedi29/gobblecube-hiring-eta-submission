
import polars as pl

def create_nyc_weather_features(weather_path, zones_path, output_path):
    # 1. Load the data 
    # infer_schema_length is increased to safely handle the messy columns in the weather dataset
    weather_df = pl.read_csv(weather_path, infer_schema_length=10000, null_values=["", "NA"])
    zones_df = pl.read_csv(zones_path)

    print("Parsing NOAA ISD string formats...")
    
    # 2. Parse the messy NOAA ISD strings into clean numeric columns
    parsed_weather = weather_df.with_columns([
        # Floor the datetime to the nearest hour
        pl.col("DATE").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).dt.truncate("1h").alias("Hour"),

        # Extract Temperature (TMP): "+0100,5" -> Extract first part, cast, handle missing (9999)
        pl.col("TMP").str.split(",").list.get(0).cast(pl.Float32).alias("raw_temp"),
        
        # Extract Wind Speed (WND): "030,5,N,0015,5" -> Speed is index 3
        pl.col("WND").str.split(",").list.get(3).cast(pl.Float32).alias("raw_wind"),
        
        # Extract Visibility (VIS): "002414,5,N,5" -> index 0
        pl.col("VIS").str.split(",").list.get(0).cast(pl.Float32).alias("raw_vis"),
        
        # Boolean flags for Rain and Snow using the REM (METAR notes) column
        # Using Regex to find specific weather codes like 'RA' (Rain), 'SN' (Snow), 'DZ' (Drizzle)
        pl.col("REM").str.contains(r"(?i)\b(RA|-RA|\+RA|DZ)\b").fill_null(False).alias("Is_Raining"),
        pl.col("REM").str.contains(r"(?i)\b(SN|-SN|\+SN)\b").fill_null(False).alias("Is_Snowing")
    ])

    # Apply scaling factors for continuous variables
    parsed_weather = parsed_weather.with_columns([
        pl.when(pl.col("raw_temp") == 9999).then(None).otherwise(pl.col("raw_temp") / 10.0).alias("Temp_C"),
        pl.when(pl.col("raw_wind") == 9999).then(None).otherwise(pl.col("raw_wind") / 10.0).alias("WindSpeed_mps"),
        pl.when(pl.col("raw_vis") == 999999).then(None).otherwise(pl.col("raw_vis")).alias("Visibility_m"),
    ])

    print("Aggregating weather to hourly frequency...")
    
    # 3. Aggregate multiple intra-hour readings into a single hourly row
    hourly_weather = parsed_weather.group_by("Hour").agg([
        pl.col("Temp_C").mean(),
        pl.col("WindSpeed_mps").mean(),
        pl.col("Visibility_m").mean(),
        pl.col("Is_Raining").any(),
        pl.col("Is_Snowing").any()
    ]).sort("Hour")

    # Forward-fill and backward-fill any gaps so no hour has missing continuous features
    hourly_weather = hourly_weather.with_columns([
        pl.col("Temp_C").fill_null(strategy="forward").fill_null(strategy="backward"),
        pl.col("WindSpeed_mps").fill_null(strategy="forward").fill_null(strategy="backward"),
        pl.col("Visibility_m").fill_null(strategy="forward").fill_null(strategy="backward")
    ])

    print("Executing cross-join with taxi zones...")
    
    # 4. Cross Join: Broadcast this hourly weather to EVERY taxi zone
    # We create a dummy column in both dataframes to perform the Cartesian product
    hourly_weather = hourly_weather.with_columns(pl.lit(1).alias("cross_key"))
    zones_df = zones_df.with_columns(pl.lit(1).alias("cross_key"))

    nyc_weather_zones = hourly_weather.join(zones_df, on="cross_key", how="inner").drop("cross_key")

    # 5. Export
    nyc_weather_zones.write_csv(output_path)
    print(f"Success! Generated continuous and boolean weather features saved to {output_path}")

# Run the pipeline
create_nyc_weather_features(
    weather_path="weather_data.csv", 
    zones_path="taxi_zone_lookup.csv", 
    output_path="nyc_weather.csv"
)