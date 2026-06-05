
import os
import sys
import time
import logging
import requests
import pandas as pd
from azure.storage.blob import BlobServiceClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import (
    NASA_POWER_BASE_URL, CITIES, WEATHER_START, WEATHER_END, NASA_PARAMS,
    LOCAL_WEATHER_CSV, LOCAL_DATA_DIR, LOCAL_LOG_DIR,
    LOCAL_ECOMMERCE_PATH, LOCAL_REVIEWS_PATH,
    AZURE_CONN_STR, AZURE_CONTAINER
)

os.makedirs(LOCAL_LOG_DIR,  exist_ok=True)
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOCAL_LOG_DIR + "upload_azure.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def fetch_nasa_weather(city_name, lat, lon):
    log.info("Fetching NASA POWER data for %s", city_name)
    params = {
        "parameters": NASA_PARAMS,
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      WEATHER_START,
        "end":        WEATHER_END,
        "format":     "JSON"
    }
    try:
        response = requests.get(NASA_POWER_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data       = response.json()
        properties = data["properties"]["parameter"]
        dates      = list(properties["T2M"].keys())
        records    = []
        for date_str in dates:
            records.append({
                "date":          date_str,
                "city":          city_name,
                "latitude":      lat,
                "longitude":     lon,
                "temperature":   properties["T2M"].get(date_str),
                "precipitation": properties["PRECTOTCORR"].get(date_str),
                "wind_speed":    properties["WS2M"].get(date_str),
                "humidity":      properties["RH2M"].get(date_str),
            })
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df.replace(-999.0, pd.NA, inplace=True)
        log.info("Got %d records for %s", len(df), city_name)
        return df
    except Exception as e:
        log.error("NASA API failed for %s: %s", city_name, e)
        return pd.DataFrame()


def fetch_all_weather():
    frames = []
    for city_name, coords in CITIES.items():
        df = fetch_nasa_weather(city_name, coords["lat"], coords["lon"])
        if not df.empty:
            frames.append(df)
        time.sleep(1)
    if not frames:
        log.error("No weather data retrieved.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    log.info("Total weather records: %d", len(combined))
    return combined


def upload_to_azure(local_path, blob_path):
    log.info("Uploading %s to Azure as %s", local_path, blob_path)
    try:
        client    = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        container = client.get_container_client(AZURE_CONTAINER)
        try:
            container.create_container()
        except Exception:
            pass
        with open(local_path, "rb") as f:
            container.upload_blob(name=blob_path, data=f, overwrite=True)
        log.info("Uploaded: %s", blob_path)
    except Exception as e:
        log.error("Azure upload failed for %s: %s", blob_path, e)


def main():
    log.info("=" * 60)
    log.info("Step 0: Upload datasets to Azure Blob Storage")
    log.info("=" * 60)

    # 1. Fetch weather
    log.info("Fetching NASA POWER weather data...")
    weather_df = fetch_all_weather()
    if not weather_df.empty:
        weather_df.to_csv(LOCAL_WEATHER_CSV, index=False)
        log.info("Weather saved locally: %s", LOCAL_WEATHER_CSV)
        upload_to_azure(LOCAL_WEATHER_CSV, "raw/weather/weather_nasa.csv")
    else:
        log.warning("Skipping weather upload - no data fetched")

    # 2. Upload eCommerce CSVs if they exist locally
    for fname in ["2019-Oct.csv", "2019-Nov.csv"]:
        fpath = os.path.join(LOCAL_DATA_DIR, fname)
        if os.path.exists(fpath):
            upload_to_azure(fpath, f"raw/ecommerce/{fname}")
        else:
            log.warning("Not found locally (skipping upload): %s", fpath)
            log.warning("Place %s in the data/ folder to upload it.", fname)

    # 3. Upload Electronics.jsonl if it exists locally
    if os.path.exists(LOCAL_REVIEWS_PATH):
        upload_to_azure(LOCAL_REVIEWS_PATH, "raw/reviews/Electronics.jsonl")
    else:
        log.warning("Not found locally (skipping upload): %s", LOCAL_REVIEWS_PATH)
        log.warning("Place Electronics.jsonl in the data/ folder to upload it.")

    log.info("Step 0 complete.")
    log.info("All available datasets uploaded to Azure Blob Storage.")
    log.info("Spark pipeline reads from local data/ folder.")


if __name__ == "__main__":
    main()
