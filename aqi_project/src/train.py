"""Train and compare models that predict TOMORROW's AQI.

Run:  python -m src.train            (auto-detects data/city_day.csv)
      python -m src.train --data path/to/city_day.csv
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_prep import (CAT_FEATURES, NUM_FEATURES, ROOT, TARGET, bucket_series,
                           build_features, clean, find_data_path, load_raw,
                           time_split, training_frame)

OUT = ROOT / "outputs"
FIG = OUT / "figures"
MODEL_DIR = ROOT / "models"
FEATURES = NUM_FEATURES + CAT_FEATURES


def make_pipeline(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def candidate_models():
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                              n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=400, random_state=42),
    }
    try:  # optional extra
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.05,
                                         max_depth=6, subsample=0.8,
                                         colsample_bytree=0.8, random_state=42)
    except ImportError:
        pass
    return models


def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        # how often the predicted CPCB category (Good/Moderate/...) is right
        "Category_Accuracy": float((bucket_series(y_true).values ==
                                    bucket_series(y_pred).values).mean()),
    }


def main(path):
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    feats = build_features(clean(load_raw(path)))
    frame = training_frame(feats)
    train, test, cutoff = time_split(frame, test_frac=0.2)
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows | split date: {cutoff.date()}")

    results, preds = {}, {}
    # baseline: "tomorrow will be the same as today"
    preds["Persistence (baseline)"] = test["AQI"].values
    results["Persistence (baseline)"] = evaluate(y_te, preds["Persistence (baseline)"])

    fitted = {}
    for name, model in candidate_models().items():
        pipe = make_pipeline(model).fit(X_tr, y_tr)
        fitted[name] = pipe
        preds[name] = pipe.predict(X_te)
        results[name] = evaluate(y_te, preds[name])
        print(f"{name:<24} RMSE={results[name]['RMSE']:.1f}  "
              f"MAE={results[name]['MAE']:.1f}  R2={results[name]['R2']:.3f}")

    best = min(fitted, key=lambda n: results[n]["RMSE"])
    print(f"\nBest model: {best}")

    # ---- per-city error and saved predictions
    pred_df = test[["City", "Date"]].copy()
    pred_df["actual"] = y_te.values
    pred_df["predicted"] = preds[best]
    pred_df.to_csv(OUT / "test_predictions.csv", index=False)
    by_city = (pred_df.assign(err=(pred_df.actual - pred_df.predicted).abs())
               .groupby("City")["err"].mean().round(1).rename("MAE").sort_values())
    by_city.to_csv(OUT / "metrics_by_city.csv")

    # ---- plots
    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, preds[best], s=4, alpha=0.3)
    lim = [0, max(y_te.max(), preds[best].max())]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("Actual next-day AQI")
    plt.ylabel("Predicted next-day AQI")
    plt.title(f"{best}: actual vs predicted (test set)")
    plt.tight_layout()
    plt.savefig(FIG / "model_actual_vs_pred.png", dpi=140)
    plt.close()

    city = "Delhi" if "Delhi" in pred_df.City.values else pred_df.City.iloc[0]
    one = pred_df[pred_df.City == city].tail(180)
    plt.figure(figsize=(10, 4))
    plt.plot(one.Date, one.actual, label="Actual")
    plt.plot(one.Date, one.predicted, label="Predicted")
    plt.title(f"{city}: next-day AQI, last 180 test days")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "model_timeseries.png", dpi=140)
    plt.close()

    sample = test.sample(min(2000, len(test)), random_state=0)
    imp = permutation_importance(fitted[best], sample[FEATURES], sample[TARGET],
                                 scoring="neg_root_mean_squared_error",
                                 n_repeats=5, random_state=0)
    order = np.argsort(imp.importances_mean)
    plt.figure(figsize=(7, 6))
    plt.barh(np.array(FEATURES)[order], imp.importances_mean[order])
    plt.xlabel("Increase in RMSE when feature is shuffled")
    plt.title(f"Permutation importance ({best})")
    plt.tight_layout()
    plt.savefig(FIG / "model_feature_importance.png", dpi=140)
    plt.close()

    # ---- refit the winner on ALL data for the app, then save
    final = make_pipeline(clone(candidate_models()[best])).fit(frame[FEATURES], frame[TARGET])
    joblib.dump({"model": final, "name": best, "features": FEATURES},
                MODEL_DIR / "best_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps({
        "best_model": best, "split_date": str(cutoff.date()),
        "train_rows": len(train), "test_rows": len(test), "models": results,
    }, indent=2))
    print("Saved: models/best_model.joblib, outputs/metrics.json, outputs/figures/model_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    data_path = args.data or find_data_path()
    if data_path is None:
        raise SystemExit("No data found. Put city_day.csv in data/ (see README).")
    print(f"Using {data_path}")
    main(data_path)
