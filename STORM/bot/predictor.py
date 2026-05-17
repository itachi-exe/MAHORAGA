"""Use STORM_v1.joblib to predict tomorrow's weather."""
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "STORM_v1.joblib"
CSV_PATH   = Path(__file__).parent.parent / "STORM.csv"


import logging as _logging
_log = _logging.getLogger("predictor")


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"STORM model not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    return (
        bundle["xgb_ensemble_component"],
        bundle["lgbm_ensemble_component"],
        bundle["quantile_risk_models"],
        bundle["feature_columns"],
    )


def _build_features(feat_cols: list) -> pd.DataFrame:
    """Read last 25 rows of CSV, compute exactly the features the model expects."""
    df = pd.read_csv(CSV_PATH, skiprows=3, low_memory=False)
    df = df[df["time"].str.match(r"^\d{4}-\d{2}-\d{2}", na=False)]
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    df = df.set_index("time").sort_index()
    df.columns = [c.split(" (")[0].strip() for c in df.columns]

    # Convert all columns to numeric and drop junk rows
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["pressure_msl"] > 900) & (df["pressure_msl"] < 1100)]

    df = df.iloc[-25:].copy()

    # Time features
    df["hour"]       = df.index.hour
    df["month"]      = df.index.month
    df["day_of_year"]= df.index.dayofyear
    df["is_day"]     = ((df["hour"] >= 6) & (df["hour"] <= 20)).astype(int)
    df["sin_hour"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]  = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features (24h ago)
    df["temperature_2m_lag_24h"]  = df["temperature_2m"].shift(24)
    df["precipitation_lag_24h"]   = df["precipitation"].shift(24)
    df["pressure_msl_lag_24h"]    = df["pressure_msl"].shift(24)

    # Dynamics
    df["pressure_msl_velocity"]     = df["pressure_msl"].diff()
    df["pressure_msl_acceleration"] = df["pressure_msl_velocity"].diff()
    df["temperature_momentum"]      = df["temperature_2m"] * df["wind_speed_10m"]
    df["rainfall_persistence_24h"]  = df["precipitation"].rolling(24, min_periods=1).sum()
    df["instability_score"]         = np.where(
        df["pressure_msl"] != 0,
        (df["temperature_2m"] * df["relative_humidity_2m"]) / df["pressure_msl"],
        0.0,
    )

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Return just the last row with the exact feature columns the model was trained on
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feat_cols].iloc[[-1]]


def run_forecast() -> dict | None:
    try:
        xgb_model, lgbm_model, quantile_models, feat_cols = _load_model()

        X = _build_features(feat_cols)

        xgb_pred  = xgb_model.predict(X)[0]
        lgbm_pred = lgbm_model.predict(X)[0]

        # Blend 50/50 (or use ensemble_config if available)
        try:
            import json
            cfg = json.load(open(Path(__file__).parent.parent / "models" / "ensemble_config.json"))
            w = cfg["lgbm_weight"]
        except Exception:
            w = 0.5

        temp = w * (lgbm_pred if np.isscalar(lgbm_pred) else lgbm_pred[0]) + \
               (1 - w) * (xgb_pred if np.isscalar(xgb_pred) else xgb_pred[0])

        # Quantile risk bounds for temperature
        q10 = float(quantile_models[0.1].predict(X)[0])
        q50 = float(quantile_models[0.5].predict(X)[0])
        q90 = float(quantile_models[0.9].predict(X)[0])

        return {
            "forecast_for":    (datetime.now(timezone.utc) + timedelta(hours=24)).strftime("%Y-%m-%d"),
            "temperature_2m":  round(float(temp), 2),
            "temp_low":        round(q10, 2),
            "temp_mid":        round(q50, 2),
            "temp_high":       round(q90, 2),
        }
    except Exception as e:
        _log.error("run_forecast failed: %s", e)
        return None


if __name__ == "__main__":
    f = run_forecast()
    print(f"\n24h Forecast for London ({f['forecast_for']}):")
    print(f"  Temperature : {f['temperature_2m']} °C")
    print(f"  Range       : {f['temp_low']} – {f['temp_high']} °C  (10th–90th percentile)")
