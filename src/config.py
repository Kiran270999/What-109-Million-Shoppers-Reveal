import os

AZURE_ACCOUNT   = os.getenv("AZURE_ACCOUNT",   "ecommercedata2024")
AZURE_KEY       = os.getenv("AZURE_KEY",       "")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "ecommerce-project")
AZURE_CONN_STR  = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_ACCOUNT};"
    f"AccountKey={AZURE_KEY};"
    f"EndpointSuffix=core.windows.net"
)

MONGO_URI           = os.getenv("MONGO_URI", "")
MONGO_DB            = "ecommerce_analysis"
MONGO_COL_BEHAVIOUR = "results_behaviour"
MONGO_COL_SENTIMENT = "results_sentiment"
MONGO_COL_WEATHER   = "results_weather"

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
CITIES = {
    "New York":    {"lat": 40.7128, "lon": -74.0060},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
    "Chicago":     {"lat": 41.8781, "lon": -87.6298},
    "Houston":     {"lat": 29.7604, "lon": -95.3698},
    "Phoenix":     {"lat": 33.4484, "lon": -112.0740},
}
WEATHER_START = "20191001"
WEATHER_END   = "20191130"
NASA_PARAMS   = "T2M,PRECTOTCORR,WS2M,RH2M"

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_DIR       = os.path.join(BASE_DIR, "data")
LOCAL_LOG_DIR        = os.path.join(BASE_DIR, "logs") + os.sep
LOCAL_OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs") + os.sep
LOCAL_WEATHER_CSV    = os.path.join(LOCAL_DATA_DIR, "weather_nasa.csv")
LOCAL_ECOMMERCE_PATH = LOCAL_DATA_DIR + os.sep
LOCAL_REVIEWS_PATH   = os.getenv("REVIEWS_PATH", os.path.join(LOCAL_DATA_DIR, "Electronics.jsonl"))

SPARK_APP_NAME   = "EcommerceAnalysis"
SPARK_MASTER     = "local[*]"
SPARK_DRIVER_MEM = "4g"
SPARK_EXEC_MEM   = "4g"
