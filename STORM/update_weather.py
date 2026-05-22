"""
Fetch missing hourly London weather data from Open-Meteo and append to
STORM.csv so STORM predictions always use up-to-date inputs.

Uses the archive API for data older than 7 days, and the forecast API
(past_days) for the most recent week.  Spread/ensemble columns that are
unavailable from the standard API are left blank (the predictor fills
them with 0.0 when missing from feat_cols).
"""
import logging
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

log = logging.getLogger("update_weather")

CSV_PATH = Path(__file__).parent / "STORM.csv"
LAT, LON = 51.493847, -0.1630249

_HOURLY_VARS = ",".join([
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "rain", "precipitation", "snowfall",
    "snow_depth", "weather_code", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
    "wind_direction_100m", "wind_gusts_10m",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
])

# Open-Meteo API key → CSV column header (with units)
_UNIT_MAP = {
    "temperature_2m":              "temperature_2m (°C)",
    "relative_humidity_2m":        "relative_humidity_2m (%)",
    "dew_point_2m":                "dew_point_2m (°C)",
    "apparent_temperature":        "apparent_temperature (°C)",
    "rain":                        "rain (mm)",
    "precipitation":               "precipitation (mm)",
    "snowfall":                    "snowfall (cm)",
    "snow_depth":                  "snow_depth (m)",
    "weather_code":                "weather_code (wmo code)",
    "pressure_msl":                "pressure_msl (hPa)",
    "surface_pressure":            "surface_pressure (hPa)",
    "cloud_cover":                 "cloud_cover (%)",
    "cloud_cover_low":             "cloud_cover_low (%)",
    "cloud_cover_mid":             "cloud_cover_mid (%)",
    "cloud_cover_high":            "cloud_cover_high (%)",
    "et0_fao_evapotranspiration":  "et0_fao_evapotranspiration (mm)",
    "vapour_pressure_deficit":     "vapour_pressure_deficit (kPa)",
    "wind_speed_10m":              "wind_speed_10m (km/h)",
    "wind_speed_100m":             "wind_speed_100m (km/h)",
    "wind_direction_10m":          "wind_direction_10m (°)",
    "wind_direction_100m":         "wind_direction_100m (°)",
    "wind_gusts_10m":              "wind_gusts_10m (km/h)",
    "soil_temperature_0_to_7cm":   "soil_temperature_0_to_7cm (°C)",
    "soil_temperature_7_to_28cm":  "soil_temperature_7_to_28cm (°C)",
    "soil_temperature_28_to_100cm":"soil_temperature_28_to_100cm (°C)",
    "soil_temperature_100_to_255cm":"soil_temperature_100_to_255cm (°C)",
    "soil_moisture_0_to_7cm":      "soil_moisture_0_to_7cm (m³/m³)",
    "soil_moisture_7_to_28cm":     "soil_moisture_7_to_28cm (m³/m³)",
    "soil_moisture_28_to_100cm":   "soil_moisture_28_to_100cm (m³/m³)",
    "soil_moisture_100_to_255cm":  "soil_moisture_100_to_255cm (m³/m³)",
}

_COMMON_PARAMS = {
    "latitude": LAT, "longitude": LON,
    "hourly": _HOURLY_VARS,
    "timezone": "UTC", "wind_speed_unit": "kmh",
}


def _fetch_archive(start_date: str, end_date: str) -> dict:
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={**_COMMON_PARAMS, "start_date": start_date, "end_date": end_date},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_recent(past_days: int) -> dict:
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={**_COMMON_PARAMS, "past_days": past_days, "forecast_days": 1},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _to_df(data: dict) -> pd.DataFrame:
    hourly = data["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"], utc=True)})
    for api_key, csv_col in _UNIT_MAP.items():
        df[csv_col] = hourly.get(api_key)
    return df


def _last_csv_timestamp() -> pd.Timestamp:
    df = pd.read_csv(CSV_PATH, skiprows=3, low_memory=False, usecols=["time"])
    df = df[df["time"].str.match(r"^\d{4}-\d{2}-\d{2}", na=False)]
    return pd.to_datetime(df["time"].iloc[-1], utc=True)


def _csv_columns() -> list[str]:
    return list(pd.read_csv(CSV_PATH, skiprows=3, low_memory=False, nrows=0).columns)


def update_csv() -> int:
    """Append missing rows to STORM.csv. Returns number of rows appended."""
    last_ts  = _last_csv_timestamp()
    now_utc  = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    if last_ts >= now_utc - timedelta(hours=1):
        log.info("STORM.csv is up to date (last: %s)", last_ts.isoformat())
        return 0

    gap_days = (now_utc - last_ts).days
    log.info("STORM.csv stale by %d day(s) — fetching update…", gap_days)

    frames: list[pd.DataFrame] = []

    # Data older than 7 days → archive API
    archive_end = now_utc - timedelta(days=7)
    if last_ts < archive_end:
        start_str = (last_ts + timedelta(hours=1)).strftime("%Y-%m-%d")
        end_str   = archive_end.strftime("%Y-%m-%d")
        log.info("Archive API  %s → %s", start_str, end_str)
        frames.append(_to_df(_fetch_archive(start_str, end_str)))

    # Last 7 days → forecast API (archive doesn't cover very recent hours)
    log.info("Forecast API  past_days=7")
    frames.append(_to_df(_fetch_recent(past_days=7)))

    new_df = pd.concat(frames, ignore_index=True)
    new_df = (
        new_df[new_df["time"] > last_ts]
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    if new_df.empty:
        log.info("No new rows after deduplication")
        return 0

    # Align to exact CSV column order; spread cols not available → blank
    all_cols = _csv_columns()
    new_df["time"] = new_df["time"].dt.strftime("%Y-%m-%dT%H:%M")
    for col in all_cols:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[all_cols]

    with open(CSV_PATH, "a", encoding="utf-8") as fh:
        new_df.to_csv(fh, header=False, index=False)

    log.info("Appended %d new hourly rows to STORM.csv (now up to %s)",
             len(new_df), new_df["time"].iloc[-1])
    return len(new_df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    n = update_csv()
    print(f"Done — {n} rows appended.")
