# 🌫️ Air Quality Analysis and Prediction for Indian Cities

An end-to-end data analytics + machine learning project: clean 5+ years of city-level air quality data, uncover seasonal and COVID-19 patterns, **predict tomorrow's AQI**, and explore everything in an interactive Streamlit dashboard.

**Live demo:** _add your Streamlit Cloud link here_

## What it does
- **Cleans** messy sensor data: continuous daily series per city, time-based interpolation for short gaps, consistent CPCB AQI categories.
- **Explores** it: city rankings, yearly trends, a city × month seasonality heatmap, pollutant correlations, and the COVID-19 lockdown effect.
- **Predicts** next-day AQI using lag features, rolling averages and seasonality, comparing a naive baseline against Ridge, Random Forest, HistGradientBoosting (and XGBoost if installed).
- **Serves** results in a dashboard with a what-if predictor and health advisory.

## Dataset
[Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) on Kaggle. Download `city_day.csv` and place it in `data/`.
Cities include Delhi, Chandigarh, Amritsar, Lucknow, Mumbai, Chennai, Bengaluru, Kolkata and more.

> No dataset yet? `python make_sample_data.py` creates a **synthetic** file so you can test the pipeline. Never report results from it.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put city_day.csv in data/, then:
python -m src.eda        # figures -> outputs/figures, findings -> outputs/insights.md
python -m src.train      # compares models, saves models/best_model.joblib + metrics
streamlit run app.py     # dashboard
```

## Project structure
```
app.py                  Streamlit dashboard (4 tabs)
make_sample_data.py     synthetic data generator (testing only)
src/data_prep.py        loading, cleaning, feature engineering, time split
src/eda.py              charts + auto-generated key findings
src/train.py            model comparison, evaluation, feature importance
data/                   put city_day.csv here
models/                 trained model (created by train.py)
outputs/                figures, metrics.json, insights.md
```

## Methodology
1. **Cleaning:** reindex each city to every calendar day, interpolate gaps of up to 5 days (longer gaps stay missing rather than being invented), rebuild AQI categories.
2. **Features:** today's AQI and core pollutants (PM2.5, PM10, NO2, SO2, CO, O3), AQI lags (1–3 days), 7-day rolling mean, 3-day PM2.5 mean, cyclical month encoding, day of week, city (one-hot).
3. **Target:** AQI of the next day.
4. **Validation:** chronological 80/20 split, not random, so the model never sees the future. Includes a *persistence baseline* ("tomorrow = today") that any real model must beat.
5. **Metrics:** MAE, RMSE, R², and category accuracy (how often the predicted CPCB band is right).
6. **Interpretation:** permutation importance and per-city error table.

## Results
_Run the pipeline on the real dataset, then paste your numbers and 2–3 screenshots here._

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence baseline | | | |
| Ridge | | | |
| Random Forest | | | |
| HistGradientBoosting | | | |

Key findings are generated in `outputs/insights.md`.

## Deploy for free
1. Push to GitHub (commit `models/best_model.joblib` and a small data file, or remove them from `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set the main file to `app.py`.

## Resume bullets (edit with your real numbers)
- Built an end-to-end AQI analytics and forecasting pipeline on 5+ years of data across 26 Indian cities using Python, Pandas and Scikit-learn.
- Engineered lag, rolling and seasonal features; compared 4+ models with time-based validation, beating a persistence baseline by X% RMSE.
- Deployed an interactive Streamlit dashboard with trend analysis, COVID-19 lockdown impact and next-day AQI prediction with health advisories.

## Ideas to extend
- Multi-day forecast (7 days) with Prophet or an LSTM.
- Add weather data (temperature, humidity, wind) for better winter predictions.
- Use the Gemini API to generate plain-language city air quality summaries.
- Live data via the OpenAQ or CPCB API.
