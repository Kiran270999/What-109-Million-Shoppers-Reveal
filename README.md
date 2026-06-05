# ecommerce-spark-pipeline

A distributed big data pipeline built with Apache Spark, Azure Blob Storage, and MongoDB Atlas. Analyses 109 million e-commerce events, 5.24 million product reviews, and NASA weather observations across three independent research questions.


## Research Questions


How do consumer browsing and purchasing patterns vary across product categories and time?
How does customer sentiment correlate with product ratings at scale?
Do weather conditions influence daily e-commerce purchase volumes?


## Datasets

eCommerce Behavior Data: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
Amazon Electronics Reviews 2023: https://www.kaggle.com/datasets/wajahat1064/amazon-reviews-data-2023
NASA POWER Historical Weather: https://power.larc.nasa.gov

All datasets stored in Azure Blob Storage prior to processing.

## Technology Stack

Distributed Processing: Apache Spark 3.5.1 (PySpark)
Cloud Storage: Azure Blob Storage
Database: MongoDB Atlas (M0 Free Tier)
Machine Learning: Spark MLlib (TF-IDF, Logistic Regression, Linear Regression)
Visualisation: Matplotlib and Seaborn
Programming Language: Python 3.12
Java Runtime: OpenJDK 11

## Repository Structure

ecommerce-spark-pipeline/
├── src/
│   ├── config.py                    # Central configuration (env-var driven)
│   ├── 00_upload_azure.py           # Weather ingestion + Azure Blob upload
│   ├── 01_analysis1_behaviour.py    # RQ1 — Consumer behaviour analysis
│   ├── 02_analysis2_sentiment.py    # RQ2 — Sentiment + MLlib pipeline
│   ├── 03_analysis3_weather.py      # RQ3 — Weather correlation + regression
│   └── 04_visualisation.py          # Follow-up analysis + 6 chart outputs
├── requirements.txt

## Setup

### Prerequisites

- Java 11 — [Adoptium Temurin 11](https://adoptium.net/temurin/releases/?version=11)
- Python 3.12
- Apache Spark 3.5.1 — [Download](https://spark.apache.org/downloads.html)
- winutils.exe (Windows only) — [cdarlint/winutils](https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin)

### Install Python dependencies


pip install -r requirements.txt

Core dependencies: `pyspark==3.5.1`, `pymongo`, `certifi`, `azure-storage-blob`, `requests`, `pandas`, `matplotlib`, `seaborn`, `setuptools`


MongoDB Spark Connector JAR:  https://repo1.maven.org/maven2/org/mongodb/spark/mongo-spark-connector_2.12/10.3.0/mongo-spark-connector_2.12-10.3.0-all.jar


## Running the Pipeline

### Windows — Individual Steps

Set environment variables in PowerShell:
```
AZURE_ACCOUNT=your_storage_account
AZURE_KEY=your_storage_key
AZURE_CONTAINER=ecommerce-project
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
REVIEWS_PATH=/path/to/Electronics.jsonl
```

Set environment variables in PowerShell (Windows):

```powershell
$env:JAVA_HOME      = "D:\Eclipse Adoptium\jdk-11.0.23+9"
$env:HADOOP_HOME    = "C:\hadoop"
$env:SPARK_HOME     = "C:\spark"
$env:SPARK_LOCAL_IP = "127.0.0.1"
$env:PATH           = "$env:JAVA_HOME\bin;C:\hadoop\bin;C:\spark\bin;" + $env:PATH
```

Run each step:
### Configure credentials
Edit `src/config.py` with your Azure account key, MongoDB URI, and local file paths before running any script.

# Step 0 — Weather ingestion + Azure Blob upload
py -3.12 src\00_upload_azure.py(upload the data sets in azure files )

# Step 1 — Consumer behaviour analysis (15–25 min)
  src\01_analysis1_behaviour.py
  ```powershell
C:\spark\bin\spark-submit.cmd --master "local[*]" --driver-memory 4g --conf "spark.local.dir=D:\spark_temp" --conf "spark.sql.shuffle.partitions=10" --jars "jars\mongo-spark-connector_2.12-10.3.0-all.jar" src\01_analysis1_behaviour.py
```

# Step 2 — Sentiment analysis (20–35 min)
  src\02_analysis2_sentiment.py
```powershell
C:\spark\bin\spark-submit.cmd --master "local[*]" --driver-memory 6g --conf "spark.local.dir=D:\spark_temp" --conf "spark.sql.shuffle.partitions=10" --conf "spark.driver.maxResultSize=2g" --conf "spark.memory.fraction=0.6" --conf "spark.memory.storageFraction=0.3" --jars "jars\mongo-spark-connector_2.12-10.3.0-all.jar" src\02_analysis2_sentiment.py
```

# Step 3 — Weather correlation (10–20 min)
  src\03_analysis3_weather.py
```powershell
C:\spark\bin\spark-submit.cmd --master "local[*]" --driver-memory 4g --conf "spark.local.dir=D:\spark_temp" --conf "spark.sql.shuffle.partitions=10" --jars "jars\mongo-spark-connector_2.12-10.3.0-all.jar" src\03_analysis3_weather.py
```

# Step 4 — Generate charts
py -3.12 src\04_visualisation.py

```powershell
py -3.12 src\04_visualisation.py
```
6 PNG charts saved to `outputs/`


### Methodology Framework

The project follows the CRISP-DM process:

1. **Business Understanding** — e-commerce problem framing, research question definition, metric selection
2. **Data Understanding** — dataset profiling, category distribution, review length statistics, weather coverage
3. **Data Preparation** — event filtering, text cleaning, weather averaging, dataset splitting, join preparation
4. **Modelling** — PySpark GroupBy + Window (RQ1), MLlib TF-IDF + LogReg (RQ2), Pearson + LinReg (RQ3)
5. **Evaluation** — conversion funnel, classification accuracy, ablation across pipeline configurations, R² analysis
6. **Deployment** — reproducible scripts, automated pipeline, MongoDB output, chart generation


### Key Findings


RQ1 — Consumer Behaviour
  Electronics revenue share  :  84.39%  ($381M of total)
  Purchase conversion rate   :  3.0%    (industry benchmark: 2–4%)
  Peak shopping window       :  19:00–21:00
  Weekend uplift             :  +23% vs weekdays

RQ2 — Sentiment Analysis
  Model accuracy             :  75.02%  (TF-IDF + Logistic Regression)
  Five-star review share     :  65.74%  of 5.24M reviews
  Positive keywords          :  great, works, easy, love, quality
  Negative keywords          :  dont, doesnt, return, broken, waste

RQ3 — Weather vs Purchases
  Dry day avg purchases      :  36,921
  Rainy day avg purchases    :  24,241  (52.3% fewer)
  Linear Regression R²       :  −9.96   (weather alone is not predictive)
  Finding                    :  Pre-holiday calendar effects dominate weather



