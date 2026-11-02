"""Exploratory analysis: saves figures to outputs/figures and findings to outputs/insights.md.

Run:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_prep import (ROOT, POLLUTANTS, SEASONS, clean, find_data_path,
                           load_raw)

FIG_DIR = ROOT / "outputs" / "figures"


def covid_impact(df):
    """Mean AQI in the lockdown window (25 Mar - 31 May): 2020 vs same window in 2019."""
    d = df[["City", "Date", "AQI"]].dropna()

    def window(year):
        return d[(d.Date >= f"{year}-03-25") & (d.Date <= f"{year}-05-31")]

    a = window(2019).groupby("City")["AQI"].mean().rename("AQI_2019")
    b = window(2020).groupby("City")["AQI"].mean().rename("AQI_2020")
    out = pd.concat([a, b], axis=1).dropna()
    out["change_pct"] = (out["AQI_2020"] / out["AQI_2019"] - 1) * 100
    return out.sort_values("change_pct").round(1).reset_index()


def monthly_matrix(df):
    d = df.dropna(subset=["AQI"]).copy()
    d["month"] = d["Date"].dt.month
    return d.pivot_table(index="City", columns="month", values="AQI",
                         aggfunc="mean").round(0)


def _save(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=140)
    plt.close()


def run(df):
    d = df.dropna(subset=["AQI"]).copy()
    d["year"] = d["Date"].dt.year
    city_mean = d.groupby("City")["AQI"].mean().sort_values()

    # 1. most / least polluted cities
    plt.figure(figsize=(8, 7))
    plt.barh(city_mean.index, city_mean.values, color="#e57373")
    plt.xlabel("Mean AQI")
    plt.title("Average AQI by city")
    _save("eda_city_mean_aqi.png")

    # 2. yearly trend: 3 most and 2 least polluted cities
    pick = list(city_mean.index[-3:]) + list(city_mean.index[:2])
    yearly = d[d.City.isin(pick)].pivot_table(index="year", columns="City", values="AQI")
    yearly.plot(marker="o", figsize=(8, 5))
    plt.ylabel("Mean AQI")
    plt.title("Yearly AQI trend")
    _save("eda_yearly_trend.png")

    # 3. city x month heatmap
    mm = monthly_matrix(d).loc[city_mean.index[::-1]]
    plt.figure(figsize=(9, 7))
    plt.imshow(mm.values, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(label="Mean AQI")
    plt.xticks(range(12), list("JFMAMJJASOND"))
    plt.yticks(range(len(mm)), mm.index)
    plt.title("Seasonality: mean AQI by city and month")
    _save("eda_seasonality_heatmap.png")

    # 4. COVID-19 lockdown effect
    cv = covid_impact(d)
    if not cv.empty:
        x = np.arange(len(cv))
        w = 0.4
        plt.figure(figsize=(10, 5))
        plt.bar(x - w / 2, cv["AQI_2019"], w, label="2019")
        plt.bar(x + w / 2, cv["AQI_2020"], w, label="2020 (lockdown)")
        plt.xticks(x, cv["City"], rotation=60, ha="right")
        plt.ylabel("Mean AQI (25 Mar - 31 May)")
        plt.legend()
        plt.title("Air quality during the COVID-19 lockdown")
        _save("eda_covid_impact.png")

    # 5. correlation heatmap
    cols = [c for c in POLLUTANTS + ["AQI"] if c in d.columns]
    corr = d[cols].corr()
    plt.figure(figsize=(8, 7))
    plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=60, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Pollutant correlations")
    _save("eda_correlation.png")

    # key findings as text
    season_mean = (d.assign(season=d["Date"].dt.month.map(SEASONS))
                   .groupby("season")["AQI"].mean())
    lines = [
        "# Key findings (auto-generated)\n",
        f"- Most polluted city (mean AQI): **{city_mean.index[-1]}** "
        f"({city_mean.iloc[-1]:.0f}); cleanest: **{city_mean.index[0]}** "
        f"({city_mean.iloc[0]:.0f}).",
        f"- Winter average AQI is {season_mean['Winter']:.0f} vs "
        f"{season_mean['Monsoon']:.0f} in the monsoon "
        f"({(season_mean['Winter'] / season_mean['Monsoon'] - 1) * 100:.0f}% higher).",
    ]
    corr_aqi = corr["AQI"].drop("AQI").sort_values(ascending=False)
    lines.append(f"- Pollutants most correlated with AQI: {', '.join(corr_aqi.index[:3])}.")
    if not cv.empty:
        lines.append(
            f"- Lockdown effect: mean AQI changed by {cv['change_pct'].mean():.0f}% on "
            f"average vs the same period in 2019 (largest drop: {cv.iloc[0]['City']}, "
            f"{cv.iloc[0]['change_pct']:.0f}%).")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "insights.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    path = find_data_path()
    if path is None:
        raise SystemExit("No data found. Put city_day.csv in data/ (see README).")
    print(f"Using {path.name}")
    run(clean(load_raw(path)))
