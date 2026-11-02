"""India Air Quality Explorer: Streamlit dashboard.

Run:  streamlit run app.py
"""
import json

import joblib
import pandas as pd
import streamlit as st

from src.data_prep import (CAT_FEATURES, CORE_POLLUTANTS, NUM_FEATURES, ROOT,
                           bucket_of, build_features, clean, find_data_path, load_raw)
from src.eda import covid_impact, monthly_matrix

st.set_page_config(page_title="India AQI Explorer", page_icon="🌫️", layout="wide")

COLORS = {"Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#e6a700",
          "Poor": "#ff7f27", "Very Poor": "#e53935", "Severe": "#7b1fa2"}
ADVICE = {
    "Good": "Air quality is good. Enjoy outdoor activities.",
    "Satisfactory": "Acceptable. Unusually sensitive people may notice mild discomfort.",
    "Moderate": "People with asthma, heart or lung disease should limit prolonged outdoor exertion.",
    "Poor": "Everyone may feel discomfort on prolonged exposure. Reduce outdoor exercise.",
    "Very Poor": "Avoid outdoor exercise. Sensitive groups should stay indoors; consider a mask outdoors.",
    "Severe": "Health emergency conditions. Stay indoors, use air purifiers, avoid all outdoor exertion.",
}

DATA_PATH = find_data_path()
if DATA_PATH is None:
    st.error("No dataset found. Download `city_day.csv` from Kaggle into the `data/` "
             "folder (see README), then reload.")
    st.stop()


@st.cache_data
def load_data(path):
    return clean(load_raw(path))


@st.cache_data
def load_features(path):
    return build_features(load_data(path))


@st.cache_resource
def load_model():
    p = ROOT / "models" / "best_model.joblib"
    return joblib.load(p) if p.exists() else None


df = load_data(str(DATA_PATH))
feats = load_features(str(DATA_PATH))
bundle = load_model()

st.title("🌫️ India Air Quality Explorer")
if DATA_PATH.name.startswith("sample"):
    st.warning("Showing SYNTHETIC sample data. Add the real Kaggle `city_day.csv` to `data/`.")
st.caption(f"Data: {DATA_PATH.name} · {df['City'].nunique()} cities · "
           f"{df['Date'].min():%b %Y} to {df['Date'].max():%b %Y}")

cities = sorted(df["City"].unique())
city = st.sidebar.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
cdf = df[df["City"] == city].dropna(subset=["AQI"])

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Seasonality & COVID", "Predict tomorrow",
                                  "Model performance"])

# ---------------------------------------------------------------- Overview
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest AQI in data", f"{cdf['AQI'].iloc[-1]:.0f}")
    c2.metric("Average AQI", f"{cdf['AQI'].mean():.0f}")
    c3.metric("Worst day", f"{cdf['AQI'].max():.0f}", f"{cdf.loc[cdf['AQI'].idxmax(), 'Date']:%d %b %Y}",
              delta_color="off")
    c4.metric("Days in 'Poor' or worse", f"{(cdf['AQI'] > 200).mean() * 100:.0f}%")

    st.subheader(f"AQI over time: {city}")
    ts = cdf.set_index("Date")[["AQI"]].assign(**{"30-day average": cdf.set_index("Date")["AQI"].rolling(30).mean()})
    st.line_chart(ts)

    st.subheader("Share of days by AQI category")
    dist = cdf["AQI_Bucket"].value_counts(normalize=True).mul(100).round(1)
    st.bar_chart(dist.reindex(list(COLORS)).fillna(0))

    st.subheader("Cities ranked by average AQI")
    st.bar_chart(df.groupby("City")["AQI"].mean().sort_values(ascending=False))

# ---------------------------------------------------------------- Seasonality & COVID
with tab2:
    st.subheader(f"Average AQI by month: {city}")
    m = monthly_matrix(df).loc[city]
    m.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    st.bar_chart(m)

    st.subheader("All cities: mean AQI by month")
    st.dataframe(monthly_matrix(df).style.background_gradient(cmap="RdYlGn_r", axis=None),
                 use_container_width=True)

    st.subheader("COVID-19 lockdown effect (25 Mar to 31 May, 2020 vs 2019)")
    cv = covid_impact(df)
    if cv.empty:
        st.info("Not enough 2019/2020 data to compare.")
    else:
        st.bar_chart(cv.set_index("City")["change_pct"])
        st.dataframe(cv, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Prediction
with tab3:
    st.subheader(f"Predict tomorrow's AQI: {city}")
    if bundle is None:
        st.info("No trained model found. Run `python -m src.train` first.")
    else:
        last = feats[(feats["City"] == city) & feats["AQI"].notna()].iloc[-1]
        st.caption(f"Starting from the latest record in the data ({last['Date']:%d %b %Y}). "
                   "Edit the values to try what-if scenarios.")
        cols = st.columns(4)
        values = {}
        for i, f in enumerate(["AQI"] + CORE_POLLUTANTS):
            default = float(last[f]) if pd.notna(last[f]) else 0.0
            values[f] = cols[i % 4].number_input(f, min_value=0.0, value=round(default, 1),
                                                 key=f"{city}_{f}")
        row = last.copy()
        for k, v in values.items():
            row[k] = v
        X = pd.DataFrame([row])[NUM_FEATURES + CAT_FEATURES]
        X[NUM_FEATURES] = X[NUM_FEATURES].astype(float)
        pred = max(0.0, float(bundle["model"].predict(X)[0]))
        bucket = bucket_of(pred)

        st.markdown(
            f"<div style='padding:18px;border-radius:12px;background:{COLORS[bucket]}22;"
            f"border-left:8px solid {COLORS[bucket]}'>"
            f"<div style='font-size:14px;opacity:.7'>Predicted AQI tomorrow</div>"
            f"<div style='font-size:42px;font-weight:700'>{pred:.0f} "
            f"<span style='font-size:22px;color:{COLORS[bucket]}'>{bucket}</span></div>"
            f"<div>{ADVICE[bucket]}</div></div>", unsafe_allow_html=True)
        st.caption(f"Model: {bundle['name']}")

# ---------------------------------------------------------------- Performance
with tab4:
    mpath = ROOT / "outputs" / "metrics.json"
    if not mpath.exists():
        st.info("Run `python -m src.train` to generate model metrics.")
    else:
        meta = json.loads(mpath.read_text())
        st.write(f"Chronological split: trained before **{meta['split_date']}**, tested after. "
                 f"Best model: **{meta['best_model']}**.")
        table = pd.DataFrame(meta["models"]).T.round(3).sort_values("RMSE")
        st.dataframe(table, use_container_width=True)
        for img, cap in [("model_actual_vs_pred.png", "Actual vs predicted"),
                         ("model_timeseries.png", "Recent test period"),
                         ("model_feature_importance.png", "Feature importance")]:
            p = ROOT / "outputs" / "figures" / img
            if p.exists():
                st.image(str(p), caption=cap)
