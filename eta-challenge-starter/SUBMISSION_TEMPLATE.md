# Your Submission: Writeup Template

### Your final score

**Dev MAE:** 215 s

### Your approach, in one paragraph

I built a **Hybrid Baseline-Residual architecture** that combines historical lookups with gradient boosting. Instead of predicting trip duration directly, I used a **3-tier vectorized median lookup** (Zone Pair + Hour + Day) as a base margin and trained a **LightGBM model** to optimize the residuals. The system relies heavily on "Ground Truth Physics" by integrating **OSRM routing data** to filter out-of-bounds speed anomalies. Key features include NOAA hourly weather data, K-Means neighborhood clusters, and specific flags for airports and same-zone "intra-trips." The pipeline is engineered for high-performance data processing using **Polars** for memory-efficient feature engineering.

### What you tried that didn't work

* **Naive XGBoost:** My first attempt at a standard GBT model resulted in a high MAE (~350s) because it couldn't learn the high-cardinality zone relationships effectively from raw data.
* **One-Hot Encoding:** Categorical encoding for NYC's 260+ zones led to massive, sparse matrices that slowed down training and actually hurt performance compared to LightGBM’s native categorical handling.
* **Straight-line Distance:** Using Haversine/Euclidean distance was a dead end. NYC’s grid system and water barriers (East River/Hudson) make "as the crow flies" metrics functionally useless compared to OSRM road-network distance.

### Where AI tooling sped you up most

**Gemini and Cursor** were instrumental in the **Data Engineering and Refactoring** stages. Specifically:

* **Vectorization:** AI helped refactor slow Python loops into vectorized Polars operations, which was critical for the 3-tier fallback lookup strategy.
* **Debugging:** It quickly identified the "Payment Type 4" mirror-record issue by helping me analyze the negative fare distribution in the raw TLC logs.
* **Boilerplate & Refactoring:** AI was excellent at generating the repetitive data-cleaning logic and refactoring my training script to handle Optuna trials efficiently.
* **Shortcomings:** The tools occasionally struggled with the specific memory constraints of the large taxi dataset, sometimes suggesting "standard" Pandas code that would crash the kernel, requiring me to manually pivot back to Polars lazy evaluation.

### Next experiments

* **Graph Neural Networks (GNNs):** I would model the NYC taxi zones as nodes and OSRM distances as edges to better capture the spatial dependencies and "spillover" traffic effects between neighboring zones.
* **Transformer-based Time Series:** Experiment with a temporal-attention mechanism to better weigh the impact of recent traffic trends (e.g., the last 2 hours of traffic) versus long-term historical medians.

### How to reproduce

```bash
# 1. Initialize Git LFS and clone
git lfs install
git clone https://github.com/AbhishekDwivedi29/gobblecube-hiring-eta-submission
cd eta-challenge-starter

# 2. Build the Docker image
docker build -t submission .

# 3. Run the inference pipeline
# Ensure your dev.parquet is in the local folder
docker run --rm -v "$(pwd):/work" submission /work/dev.parquet /work/preds.csv

```

**Total time spent on this challenge:** 24 hours.