
import os
import sys
import logging
import certifi
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SPARK_APP_NAME, SPARK_MASTER,
    LOCAL_REVIEWS_PATH, LOCAL_LOG_DIR, LOCAL_OUTPUT_DIR,
    MONGO_URI, MONGO_DB, MONGO_COL_SENTIMENT
)

os.makedirs(LOCAL_LOG_DIR,    exist_ok=True)
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOCAL_LOG_DIR + "analysis2_sentiment.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def create_spark_session():
    log.info("Initialising SparkSession for Analysis 2")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME + "_Sentiment")
        .master(SPARK_MASTER)
        .config("spark.driver.host",           "127.0.0.1")
        .config("spark.driver.bindAddress",    "127.0.0.1")
        .config("spark.driver.memory",         "6g")
        .config("spark.executor.memory",       "6g")
        .config("spark.driver.maxResultSize",  "2g")
        .config("spark.sql.shuffle.partitions","10")
        .config("spark.sql.adaptive.enabled",  "true")
        .config("spark.memory.fraction",       "0.6")
        .config("spark.memory.storageFraction","0.3")
        .config("spark.mongodb.write.connection.uri", MONGO_URI)
        .config("spark.mongodb.write.database",   MONGO_DB)
        .config("spark.mongodb.write.collection", MONGO_COL_SENTIMENT)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log.info("SparkSession ready.")
    return spark


def load_reviews(spark):
    """
    Load Electronics.jsonl from local disk.
    Format: JSON Lines - one review per line.
    Fields: rating, title, text, asin, timestamp, helpful_vote, verified_purchase
    File must be in the data/ folder: Electronics.jsonl
    """
    log.info("Loading Electronics.jsonl from: %s", LOCAL_REVIEWS_PATH)
    df = (
        spark.read
        .option("multiLine", "false")
        .json(LOCAL_REVIEWS_PATH)
    )
    log.info("Reviews loaded: %d rows, %d columns", df.count(), len(df.columns))
    return df


def clean_reviews(df):
    """
    Prepare reviews for sentiment analysis.
    - Cast rating to double
    - Combine title and text into review_text
    - Lowercase and remove non-alphabetic characters
    - Remove reviews shorter than 10 characters
    - Assign sentiment labels:
        0 = negative (1-2 stars)
        1 = neutral  (3 stars)
        2 = positive (4-5 stars)
    Note: df.cache() intentionally omitted to prevent OOM on 5M+ reviews
    """
    log.info("Cleaning review data")
    df = df.withColumn("rating", F.col("rating").cast(DoubleType()))
    df = df.withColumn(
        "review_text",
        F.concat_ws(
            " ",
            F.coalesce(F.col("title"), F.lit("")),
            F.coalesce(F.col("text"),  F.lit(""))
        )
    )
    df = df.withColumn(
        "review_clean",
        F.regexp_replace(F.lower(F.col("review_text")), "[^a-zA-Z\\s]", "")
    )
    df = df.dropna(subset=["rating", "review_clean"])
    df = df.filter(F.length("review_clean") >= 10)
    df = df.filter(F.col("rating").between(1.0, 5.0))
    df = df.withColumn(
        "sentiment_label",
        F.when(F.col("rating") <= 2, 0)
        .when(F.col("rating") == 3,  1)
        .otherwise(2)
    )
    df = df.withColumn(
        "review_date",
        F.to_date(F.from_unixtime(F.col("timestamp") / 1000))
    )
    df = df.withColumn("category", F.lit("Electronics"))
    log.info("Clean reviews ready: %d rows", df.count())
    return df


def analyse_rating_distribution(df):
    log.info("Sub-analysis 2a: Rating distribution")
    total = df.count()
    rating_dist = (
        df.groupBy("rating")
        .agg(
            F.count("*").alias("review_count"),
            F.round(F.count("*") / total * 100, 2).alias("percentage")
        )
        .orderBy("rating")
    )
    verified_stats = (
        df.groupBy("verified_purchase")
        .agg(
            F.count("*").alias("review_count"),
            F.avg("rating").alias("avg_rating"),
            F.round(F.count("*") / total * 100, 2).alias("percentage")
        )
    )
    monthly_trend = (
        df.filter(F.col("review_date").isNotNull())
        .withColumn("year_month", F.date_format("review_date", "yyyy-MM"))
        .groupBy("year_month")
        .agg(
            F.count("*").alias("review_count"),
            F.avg("rating").alias("avg_rating")
        )
        .orderBy("year_month")
    )
    return rating_dist, verified_stats, monthly_trend


def extract_top_keywords(df):
    log.info("Sub-analysis 2b: Top keywords by sentiment group")
    tokenizer = Tokenizer(inputCol="review_clean", outputCol="words_raw")
    remover   = StopWordsRemover(inputCol="words_raw", outputCol="words")
    results   = []
    for group_name, label in [("positive", 2), ("negative", 0)]:
        group     = df.filter(F.col("sentiment_label") == label)
        tokenised = tokenizer.transform(group)
        cleaned   = remover.transform(tokenised)
        word_counts = (
            cleaned.select(F.explode("words").alias("word"))
            .filter(F.length("word") > 3)
            .groupBy("word")
            .agg(F.count("*").alias("frequency"))
            .orderBy(F.desc("frequency"))
            .limit(50)
            .withColumn("sentiment_group", F.lit(group_name))
        )
        results.append(word_counts)
        log.info("Keywords extracted for: %s", group_name)
    return results


def build_sentiment_classifier(df):
    """
    TF-IDF + Logistic Regression sentiment classifier.
    MLlib Pipeline stages:
      1. Tokenizer      - splits text into tokens
      2. StopWordsRemover - removes common words
      3. HashingTF      - maps tokens to 10,000 features
      4. IDF            - reduces weight of common words
      5. LogisticRegression - classifies negative/neutral/positive
    Stratified 2% sample per class to prevent OOM on limited-RAM machines.
    """
    log.info("Sub-analysis 2c: TF-IDF + Logistic Regression classifier")
    sample = df.sampleBy(
        "sentiment_label",
        fractions={0: 0.02, 1: 0.02, 2: 0.005},
        seed=42
    )
    log.info("Training sample size: %d", sample.count())
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="review_clean",     outputCol="words_raw"),
        StopWordsRemover(inputCol="words_raw", outputCol="words"),
        HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000),
        IDF(inputCol="raw_features",           outputCol="features", minDocFreq=5),
        LogisticRegression(
            featuresCol="features",
            labelCol="sentiment_label",
            maxIter=10,
            regParam=0.01
        )
    ])
    train_df, test_df = sample.randomSplit([0.8, 0.2], seed=42)
    log.info("Training model...")
    model       = pipeline.fit(train_df)
    predictions = model.transform(test_df)
    accuracy    = MulticlassClassificationEvaluator(
        labelCol="sentiment_label",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)
    log.info("Model accuracy: %.2f%%", accuracy * 100)
    results = predictions.select(
        "rating", "sentiment_label", "prediction", "category"
    ).withColumn(
        "correct", (F.col("sentiment_label") == F.col("prediction")).cast(IntegerType())
    ).withColumn("model_accuracy", F.lit(round(accuracy, 4)))
    return results, accuracy


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
    log.info("Analysis 2: Sentiment and Rating Correlation")
    log.info("=" * 60)
    spark = create_spark_session()
    try:
        reviews_df = load_reviews(spark)
        clean_df   = clean_reviews(reviews_df)

        rating_dist, verified_stats, monthly_trend = analyse_rating_distribution(clean_df)
        keyword_results                             = extract_top_keywords(clean_df)
        classifier_results, accuracy                = build_sentiment_classifier(clean_df)

        save_to_mongodb(rating_dist,        MONGO_COL_SENTIMENT, "rating_distribution")
        save_to_mongodb(verified_stats,     MONGO_COL_SENTIMENT, "verified_stats")
        save_to_mongodb(monthly_trend,      MONGO_COL_SENTIMENT, "monthly_trend")
        for kw_df in keyword_results:
            save_to_mongodb(kw_df,          MONGO_COL_SENTIMENT, "top_keywords")
        save_to_mongodb(classifier_results, MONGO_COL_SENTIMENT, "classifier_results")

        save_csv(rating_dist,    "analysis2_rating_dist")
        save_csv(verified_stats, "analysis2_verified_stats")
        save_csv(monthly_trend,  "analysis2_monthly_trend")

        log.info("Analysis 2 complete. Model accuracy: %.2f%%", accuracy * 100)
        rating_dist.show()

    except Exception as e:
        log.error("Analysis 2 failed: %s", e)
        raise
    finally:
        spark.stop()
        log.info("SparkSession stopped.")


if __name__ == "__main__":
    main()
