
import os
import sys
import logging
import certifi
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEM, SPARK_EXEC_MEM,
    LOCAL_ECOMMERCE_PATH, LOCAL_LOG_DIR, LOCAL_OUTPUT_DIR,
    MONGO_URI, MONGO_DB, MONGO_COL_BEHAVIOUR
)

os.makedirs(LOCAL_LOG_DIR,    exist_ok=True)
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOCAL_LOG_DIR + "analysis1_behaviour.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def create_spark_session():
    log.info("Initialising SparkSession for Analysis 1")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME + "_Behaviour")
        .master(SPARK_MASTER)
        .config("spark.driver.host",        "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory",      SPARK_DRIVER_MEM)
        .config("spark.executor.memory",    SPARK_EXEC_MEM)
        .config("spark.sql.shuffle.partitions",   "10")
        .config("spark.sql.adaptive.enabled",     "true")
        .config("spark.mongodb.write.connection.uri", MONGO_URI)
        .config("spark.mongodb.write.database",   MONGO_DB)
        .config("spark.mongodb.write.collection", MONGO_COL_BEHAVIOUR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log.info("SparkSession ready.")
    return spark


def load_ecommerce_data(spark):
    """
    Load both monthly CSV files from local disk.
    Files must be in the data/ folder: 2019-Oct.csv and 2019-Nov.csv
    Schema: event_time, event_type, product_id, category_id,
            category_code, brand, price, user_id, user_session
    """
    path = LOCAL_ECOMMERCE_PATH + "2019-*.csv"
    log.info("Loading eCommerce CSV files from: %s", path)
    df = (
        spark.read
        .option("header",      "true")
        .option("inferSchema", "true")
        .csv(path)
    )
    log.info("Loaded %d rows, %d columns", df.count(), len(df.columns))
    return df


def clean_and_prepare(df):
    log.info("Cleaning and preparing data")
    df = df.withColumn("price", F.col("price").cast(DoubleType()))
    df = df.withColumn(
        "event_time",
        F.to_timestamp(F.col("event_time"), "yyyy-MM-dd HH:mm:ss z")
    )
    df = (
        df
        .withColumn("event_hour",    F.hour("event_time"))
        .withColumn("event_weekday", F.dayofweek("event_time"))
        .withColumn("event_date",    F.to_date("event_time"))
        .withColumn("event_month",   F.month("event_time"))
    )
    df = df.withColumn(
        "category_top",
        F.split(F.col("category_code"), "\\.").getItem(0)
    )
    df = df.dropna(subset=["event_type", "product_id", "event_time"])

    # Remove bot sessions (> 500 events per session)
    session_counts = df.groupBy("user_session").agg(F.count("*").alias("cnt"))
    valid_sessions  = session_counts.filter(F.col("cnt") <= 500)
    df = df.join(valid_sessions.select("user_session"), on="user_session", how="inner")

    df.cache()
    log.info("Clean dataset: %d rows", df.count())
    return df


def analyse_event_distribution(df):
    log.info("Sub-analysis 1a: Event distribution and conversion funnel")
    event_dist = (
        df.groupBy("event_type")
        .agg(
            F.count("*").alias("event_count"),
            F.countDistinct("user_id").alias("unique_users")
        )
        .orderBy(F.desc("event_count"))
    )
    views = df.filter(F.col("event_type") == "view").count()
    carts = df.filter(F.col("event_type") == "cart").count()
    purch = df.filter(F.col("event_type") == "purchase").count()
    funnel_data = [{
        "total_events":             views + carts + purch,
        "views":                    views,
        "carts":                    carts,
        "purchases":                purch,
        "cart_conversion_rate":     round(carts / views * 100, 2) if views > 0 else 0,
        "purchase_conversion_rate": round(purch / views * 100, 2) if views > 0 else 0
    }]
    funnel = df.sparkSession.createDataFrame(funnel_data)
    return event_dist, funnel


def analyse_category_behaviour(df):
    log.info("Sub-analysis 1b: Category revenue analysis")
    purchases = df.filter(F.col("event_type") == "purchase")
    category_stats = (
        purchases.groupBy("category_top")
        .agg(
            F.count("*").alias("total_purchases"),
            F.sum("price").alias("total_revenue"),
            F.avg("price").alias("avg_price"),
            F.countDistinct("user_id").alias("unique_buyers")
        )
        .filter(F.col("category_top").isNotNull())
    )
    window_spec = Window.orderBy(F.desc("total_revenue"))
    category_stats = category_stats.withColumn("revenue_rank", F.rank().over(window_spec))
    total_rev = category_stats.agg(F.sum("total_revenue")).collect()[0][0]
    if total_rev:
        category_stats = category_stats.withColumn(
            "revenue_share_pct",
            F.round((F.col("total_revenue") / total_rev) * 100, 2)
        )
    return category_stats


def analyse_temporal_patterns(df):
    log.info("Sub-analysis 1c: Temporal purchase patterns")
    purchases = df.filter(F.col("event_type") == "purchase")
    hourly = (
        purchases.groupBy("event_hour")
        .agg(
            F.count("*").alias("purchase_count"),
            F.sum("price").alias("total_revenue"),
            F.avg("price").alias("avg_order_value")
        )
        .orderBy("event_hour")
    )
    daily_weekday = (
        purchases.groupBy("event_weekday")
        .agg(
            F.count("*").alias("purchase_count"),
            F.sum("price").alias("total_revenue")
        )
        .withColumn(
            "weekday_name",
            F.when(F.col("event_weekday") == 1, "Sunday")
            .when(F.col("event_weekday") == 2, "Monday")
            .when(F.col("event_weekday") == 3, "Tuesday")
            .when(F.col("event_weekday") == 4, "Wednesday")
            .when(F.col("event_weekday") == 5, "Thursday")
            .when(F.col("event_weekday") == 6, "Friday")
            .otherwise("Saturday")
        )
        .orderBy("event_weekday")
    )
    daily_trend = (
        purchases.groupBy("event_date")
        .agg(
            F.count("*").alias("daily_purchases"),
            F.sum("price").alias("daily_revenue"),
            F.countDistinct("user_id").alias("active_buyers")
        )
        .orderBy("event_date")
    )
    return hourly, daily_weekday, daily_trend


def save_to_mongodb(df, collection, label):
    """Write Spark DataFrame directly to MongoDB using Mongo Spark Connector."""
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
        log.info("CSV saved: %s", path)
    except Exception as e:
        log.error("CSV save failed for '%s': %s", name, e)


def main():
    log.info("=" * 60)
    log.info("Analysis 1: eCommerce Consumer Behaviour Patterns")
    log.info("=" * 60)
    spark = create_spark_session()
    try:
        raw_df   = load_ecommerce_data(spark)
        clean_df = clean_and_prepare(raw_df)

        event_dist, funnel           = analyse_event_distribution(clean_df)
        category_stats               = analyse_category_behaviour(clean_df)
        hourly, daily_weekday, trend = analyse_temporal_patterns(clean_df)

        save_to_mongodb(event_dist,     MONGO_COL_BEHAVIOUR, "event_distribution")
        save_to_mongodb(funnel,         MONGO_COL_BEHAVIOUR, "conversion_funnel")
        save_to_mongodb(category_stats, MONGO_COL_BEHAVIOUR, "category_behaviour")
        save_to_mongodb(hourly,         MONGO_COL_BEHAVIOUR, "hourly_patterns")
        save_to_mongodb(daily_weekday,  MONGO_COL_BEHAVIOUR, "daily_patterns")
        save_to_mongodb(trend,          MONGO_COL_BEHAVIOUR, "daily_trend")

        save_csv(category_stats, "analysis1_category_stats")
        save_csv(trend,          "analysis1_daily_trend")
        save_csv(funnel,         "analysis1_funnel")
        save_csv(hourly,         "analysis1_hourly")

        log.info("Analysis 1 complete.")
        log.info("Top 10 categories by revenue:")
        category_stats.filter(F.col("revenue_rank") <= 10).show(truncate=False)

    except Exception as e:
        log.error("Analysis 1 failed: %s", e)
        raise
    finally:
        spark.stop()
        log.info("SparkSession stopped.")


if __name__ == "__main__":
    main()
