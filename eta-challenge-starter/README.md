# 🚖 NYC Taxi Trip Duration Prediction: Journey to a 215s MAE

## 📌 Introduction

When I was handed this task, the initial approach was a standard Gradient Boosted Tree (XGBoost) model. It was a classic "throw data at an algorithm" strategy, and it resulted in a Mean Absolute Error (MAE) of ~350s.

Through aggressive data cleaning, domain-specific feature engineering, and a shift in architecture, I managed to bring the Dev MAE down to **215s**. This README documents my step-by-step process, the lightbulb moments I had along the way, and why I made each technical pivot.

---

## 📈 The Incremental Journey

### Phase 1: The Baseline Epiphany

**Baseline MAE: ~350s ➡️ 274s**

* **The Problem:** The naive XGBoost model was struggling to learn the complex, high-cardinality relationships of NYC traffic purely from raw features.
* **The Fix:** I realized that in a highly periodic environment like Manhattan, historical medians are incredibly hard to beat. I threw out the XGBoost model and built a **3-Tier Fallback Lookup Strategy**:
1. Look up exact `(Zone + Hour)` median.
2. Fallback to `(Zone)` route median.
3. Fallback to `Global Median`.


* **The Result:** A massive drop to a 274s MAE. Sometimes a well-calculated average is better than an untuned ML model.

### Phase 2: The Hybrid Architecture

**Dev MAE: 274s ➡️ 272.4s**

* **The Problem:** The lookup table was fast, but it was "blind" to broader contextual features (like weather or specific days of the week).
* **The Fix:** I created a **Hybrid Lookup + LightGBM Residual Model**. Instead of predicting the duration directly, the ML model now predicts the *error* (residual) of my median baseline. I switched to LightGBM to utilize its native categorical support, skipping heavy One-Hot Encoding, and vectorized the lookup to keep inference lightning fast.

### Phase 3: Error Analysis & "Ghost Trips"

**Dev MAE: 272.4s ➡️ 267s**

* **The Problem:** I built a deep-dive error analysis script (`nlargest(20)`) and noticed the model was being heavily penalized by massive outliers (e.g., 3-hour trips that started and ended in the exact same zone). These were drivers leaving the meter running, not actual traffic.
* **The Fix:** I capped the duration at the **99th percentile** for each same pickup/dropoff zone pair. I also added an `is_weekend` feature to capture the obvious difference between a Tuesday morning commute and a Saturday night out.

### Phase 4: Ground Truth Physics & The "Payment Type 4" Breakthrough

**Dev MAE: 267s ➡️ 231s**

* **The Problem:** The model didn't understand physical distance, only categorical zone IDs. Furthermore, the dataset was littered with duplicates and accounting noise that skewed historical medians.
* **The Fix:**
* **Accounting Artifacts & Strict Duplicate Removal:** This was a massive lightbulb moment. I realized that a huge chunk of our "duplicate" and noisy records were actually ledger corrections. Specifically, **Payment Type 4 indicates a "Dispute."** When a fare is disputed or entered incorrectly, the TLC system logs an exact mirror row with a *negative fare* to void it, followed by a new duplicate row with the corrected amount. By strictly filtering out these negative fares and their associated duplicates, the training signal became instantly cleaner.
* **OSRM Integration:** I built `build_osrm_matrix.py` to extract actual routing distances and times between zones.
* **Speed Ratio Filter:** I implemented an OSRM-based speed ratio filter. If a trip's actual speed was outside the $0.2\times$ to $1.5\times$ range of the OSRM estimate, it was dropped as a glitch.
* **Time Features:** Added `week_of_year` and `week_of_month` while dropping useless features like `passenger_count`.



### Phase 5: Spatial Clusters & Environmental Context

**Dev MAE: 231s ➡️ 228s**

* **The Problem:** The TLC dataset has "Unknown" zones (264 and 265) that add pure noise. Also, baseline medians lacked awareness of weather-induced traffic jams.
* **The Fix:**
* Dropped all trips involving zones 264/265.
* Used K-Means to create `pickup_cluster` and `dropoff_cluster` features, allowing the model to learn broader neighborhood routing patterns.
* Merged **NOAA hourly weather data** (`temp_c`, `windspeed_mps`, `visibility_m`, `is_raining`, `is_snowing`).
* Added a `route_day_hour_freq` feature to dynamically measure traffic density on specific routes at specific times.



### Phase 6: Domain Polish & Final Tuning

**Dev MAE: 228s ➡️ 215s**

* **The Problem:** Trips to/from airports (JFK/LaGuardia) have entirely different pricing and traffic dynamics than standard city trips. Furthermore, the speed filter needed tightening.
* **The Fix:**
* Tightened the lower bound of the OSRM speed ratio threshold to **0.25**.
* Engineered specific categorical flags: `is_airport`, `is_intra_zone`, and `airport_intra`.
* Added `expected_zone_speed` to give the model better intuition on same-zone trips where OSRM returns 0 distance.
* Applied `log_route_freq` to smooth out high-variance traffic density.
* **Final stroke:** Lowered the same-zone outlier cap from the 99th percentile down to the **95th percentile**. Same-zone trips are notoriously noisy, and aggressively filtering the top 5% of these long-duration anomalies (often tied to the meter left running) stabilized the model significantly, bringing the MAE to our final 215s mark.



---

## 🏗️ Final Architecture

1. **Data Pipeline:** Strict duplicate & "Payment Type 4" dispute removal $\rightarrow$ OSRM speed bounds (0.25-1.5x) $\rightarrow$ 95th Percentile Same-Zone Capping (Train only) $\rightarrow$ Drop zones 264/265.
2. **Baseline Generation:** Vectorized dictionary lookups mapping `(Zone Pair + Hour + Day)` to historical medians, falling back to OSRM estimates.
3. **Residual Optimization:** A LightGBM model utilizing `init_score` (Base Margin) to predict how much the baseline will be off by, relying heavily on NOAA weather, temporal flags, and spatial K-Means clusters.

---

## 🛠️ How to Run

This project is fully Dockerized for easy reproducibility. Please ensure you have Docker and Git LFS installed before starting.

```bash
# 1. Initialize Git LFS (Required for downloading large datasets/models)
git lfs install

# 2. Clone the repository
git clone [https://github.com/AbhishekDwivedi29/gobblecube-hiring-eta-submission](https://github.com/AbhishekDwivedi29/gobblecube-hiring-eta-submission)
cd eta-challenge-starter

# 3. Build the Docker image
docker build -t submission .

# 4. Run the inference pipeline
# Replace <local_path> with the absolute path to the directory containing your dev.parquet file.
docker run --rm -v "<local_path>:/work" submission /work/dev.parquet /work/preds.csv

