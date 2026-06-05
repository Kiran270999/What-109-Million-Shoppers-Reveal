
import os
import sys
import logging
import certifi
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Correlation

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEM, SPARK_EXEC_MEM,
    LOCAL_ECOMMERCE_PATH, LOCAL_WEATHER_CSV, LOCAL_LOG_DIR, LOCAL_OUTPUT_DIR,
    MONGO_URI, MONGO_DB, MONGO_COL_WEATHER
)

os.makedirs(LOCAL_LOG_DIR,    exist_ok=True)
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOCAL_LOG_DIR + "analysis3_weather.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def create_spark_session():
    log.info("Initialising SparkSession for Analysis 3")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME + "_Weather")
        .master(SPARK_MASTER)
        .config("spark.driver.host",        "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory",      SPARK_DRIVER_MEM)
        .config("spark.executor.memory",    SPARK_EXEC_MEM)
        .config("spark.sql.shuffle.partitions",   "10")
        .config("spark.sql.adaptive.enabled",     "true")
        .config("spark.mongodb.write.connection.uri", MONGO_URI)
        .config("spark.mongodb.write.database",   MONGO_DB)
        .config("spark.mongodb.write.collection", MONGO_COL_WEATHER)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log.info("SparkSession ready.")
    return spark


def load_ecommerce_daily(spark):
    """
    Load eCommerce CSVs from local disk and aggregate to daily purchase totals.
    Files: 2019-Oct.csv and 2019-Nov.csv in the data/ folder.
    Only purchase events are retained.
    """
    path = LOCAL_ECOMMERCE_PATH + "2019-*.csv"
    log.info("Loading eCommerce data from: %s", path)
    df = (
        spark.read
        .option("header",      "true")
        .option("inferSchema", "true")
        .csv(path)
    )
    df = df.withColumn(
        "event_time",
        F.to_timestamp(F.col("event_time"), "yyyy-MM-dd HH:mm:ss z")
    )
    df = df.withColumn("event_date", F.to_date("event_time"))
    df = df.withColumn("price",      F.col("price").cast(DoubleType()))
    purchases = df.filter(F.col("event_type") == "purchase")
    daily_purchases = (
        purchases.groupBy("event_date")
        .agg(
            F.count("*").alias("purchase_count"),
            F.sum("price").alias("daily_revenue"),
            F.countDistinct("user_id").alias("active_buyers"),
            F.avg("price").alias("avg_order_value")
        )
        .filter(F.col("event_date").isNotNull())
        .orderBy("event_date")
    )
    log.info("Daily purchase data ready: %d days", daily_purchases.count())
    return daily_purchases


def load_weather_data(spark):
    """
    Load NASA POWER weather CSV from local disk.
    File: data/weather_nasa.csv (created by 00_upload_azure.py).
    Averages measurements across five cities to produce one composite daily record.
    Adds a rain_day flag: 1 if precipitation > 1mm, else 0.
    """
    log.info("Loading weather data from: %s", LOCAL_WEATHER_CSV)
    df = (
        spark.read
        .option("header",      "true")
        .option("inferSchema", "true")
        .csv(LOCAL_WEATHER_CSV)
    )
    df = df.withColumn("date",     F.to_date("date"))
    df = df.withColumn("rain_flag", F.when(F.col("precipitation") > 1.0, 1).otherwise(0))
    daily_weather = (
        df.groupBy("date")
        .agg(
            F.avg("temperature").alias("avg_temperature"),
            F.avg("precipitation").alias("avg_precipitation"),
            F.avg("wind_speed").alias("avg_wind_speed"),
            F.avg("humidity").alias("avg_humidity"),
            F.max("rain_flag").alias("rain_day")
        )
        .orderBy("date")
    )
    log.info("Weather data ready: %d days", daily_weather.count())
    return daily_weather


def join_datasets(daily_purchases, daily_weather):
    log.info("Joining eCommerce and weather data on date")
    joined = (
        daily_purchases
        .join(daily_weather, daily_purchases.event_date == daily_weather.date, "inner")
        .drop("date")
        .orderBy("event_date")
    )
    log.info("Joined dataset: %d days", joined.count())
    return joined


def analyse_correlation(joined_df):
    log.info("Sub-analysis 3a: Pearson correlation matrix")
    assembler = VectorAssembler(
        inputCols=[
            "purchase_count", "daily_revenue",
            "avg_temperature", "avg_precipitation",
            "avg_wind_speed",  "avg_humidity"
        ],
        outputCol="features",
        handleInvalid="skip"
    )
    feature_df  = assembler.transform(joined_df).select("features")
    corr_matrix = Correlation.corr(feature_df, "features", "pearson")
    corr_array  = corr_matrix.collect()[0][0].toArray()
    col_names   = [
        "purchase_count", "daily_revenue",
        "avg_temperature", "avg_precipitation",
        "avg_wind_speed",  "avg_humidity"
    ]
    records = []
    for i, col1 in enumerate(col_names):
        for j, col2 in enumerate(col_names):
            records.append({
                "variable_1":   col1,
                "variable_2":   col2,
                "pearson_corr": round(float(corr_array[i][j]), 4)
            })
    log.info("Correlation matrix computed")
    return joined_df.sparkSession.createDataFrame(records)


def analyse_weather_buckets(joined_df):
    log.info("Sub-analysis 3b: Rain vs dry day analysis")
    rain_analysis = (
        joined_df.groupBy("rain_day")
        .agg(
            F.avg("purchase_count").alias("avg_daily_purchases"),
            F.avg("daily_revenue").alias("avg_daily_revenue"),
            F.avg("active_buyers").alias("avg_active_buyers"),
            F.count("*").alias("num_days")
        )
        .withColumn(
            "rain_label",
            F.when(F.col("rain_day") == 1, "Rainy Day").otherwise("Dry Day")
        )
    )
    temp_analysis = (
        joined_df.withColumn(
            "temp_bucket",
            F.when(F.col("avg_temperature") < 0,  "freezing")
            .when(F.col("avg_temperature") < 10,  "cold")
            .when(F.col("avg_temperature") < 20,  "mild")
            .when(F.col("avg_temperature") < 30,  "warm")
            .otherwise("hot")
        )
        .groupBy("temp_bucket")
        .agg(
            F.avg("purchase_count").alias("avg_daily_purchases"),
            F.avg("daily_revenue").alias("avg_daily_revenue"),
            F.count("*").alias("days_in_group")
        )
    )
    return rain_analysis, temp_analysis


def build_regression_model(joined_df):
    log.info("Sub-analysis 3c: Linear Regression model")
    assembler = VectorAssembler(
        inputCols=[
            "avg_temperature", "avg_precipitation",
            "avg_wind_speed",  "avg_humidity"
        ],
        outputCol="features",
        handleInvalid="skip"
    )
    model_df = assembler.transform(joined_df).select("features", "purchase_count")
    train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)
    lr          = LinearRegression(featuresCol="features", labelCol="purchase_count",
                                   maxIter=100, regParam=0.01)
    model       = lr.fit(train_df)
    predictions = model.transform(test_df)
    r2   = RegressionEvaluator(labelCol="purchase_count", metricName="r2").evaluate(predictions)
    rmse = RegressionEvaluator(labelCol="purchase_count", metricName="rmse").evaluate(predictions)
    log.info("R-squared: %.4f | RMSE: %.2f", r2, rmse)
    metrics_df = joined_df.sparkSession.createDataFrame([{
        "model":          "LinearRegression",
        "r2":             round(r2,   4),
        "rmse":           round(rmse, 2),
        "coeff_temp":     round(float(model.coefficients[0]), 4),
        "coeff_precip":   round(float(model.coefficients[1]), 4),
        "coeff_wind":     round(float(model.coefficients[2]), 4),
        "coeff_humidity": round(float(model.coefficients[3]), 4),
        "intercept":      round(model.intercept, 4)
    }])
    return metrics_df, r2, rmse


def save_to_mongodb(df, collection, label):
    """Write Spark DataFrame to MongoDB Atlas using pymongo with certifi TLS."""
    import pymongo
    log.info("Writing '%s' to MongoDB: %s", label, collection)
    try:
        df_pd = df.limit(10000).toPandas()
        df_pd["analysis"] = label
        client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        col    = client[MONGO_DB][collection]
        col.insert_many(df_pd.to_dict("records"))
        log.info("Written %d documents to MongoDB: %s", len(df_pd), collection)
    except Exception as e:
        log.error("MongoDB write failed for '%s': %s", label, e)


def save_csv(df, name):
    path = LOCAL_OUTPUT_DIR + name
    log.info("Saving CSV: %s", path)
    try:
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
    except Exception as e:
        log.error("CSV save failed for '%s': %s", name, e)


def main():
    log.info("=" * 60)
    log.info("Analysis 3: Weather Conditions vs E-Commerce Purchases")
    log.info("=" * 60)
    spark = create_spark_session()
    try:
        daily_purchases = load_ecommerce_daily(spark)
        daily_weather   = load_weather_data(spark)
        joined_df       = join_datasets(daily_purchases, daily_weather)
        joined_df.cache()

        corr_df              = analyse_correlation(joined_df)
        rain_df, temp_df     = analyse_weather_buckets(joined_df)
        metrics_df, r2, rmse = build_regression_model(joined_df)

        save_to_mongodb(joined_df,  MONGO_COL_WEATHER, "daily_joined_data")
        save_to_mongodb(corr_df,    MONGO_COL_WEATHER, "correlation_matrix")
        save_to_mongodb(rain_df,    MONGO_COL_WEATHER, "rain_analysis")
        save_to_mongodb(temp_df,    MONGO_COL_WEATHER, "temperature_buckets")
        save_to_mongodb(metrics_df, MONGO_COL_WEATHER, "regression_metrics")

        save_csv(joined_df,  "analysis3_joined_data")
        save_csv(corr_df,    "analysis3_correlations")
        save_csv(rain_df,    "analysis3_rain_analysis")
        save_csv(metrics_df, "analysis3_regression_metrics")

        log.info("Analysis 3 complete.")
        log.info("R-squared: %.4f | RMSE: %.2f", r2, rmse)
        log.info("Rain vs Dry day comparison:")
        rain_df.show()

    except Exception as e:
        log.error("Analysis 3 failed: %s", e)
        raise
    finally:
        spark.stop()
        log.info("SparkSession stopped.")


if __name__ == "__main__":
    main()
