
import os
import sys
import logging
import warnings
import certifi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MONGO_URI, MONGO_DB,
    MONGO_COL_BEHAVIOUR, MONGO_COL_SENTIMENT, MONGO_COL_WEATHER,
    LOCAL_LOG_DIR, LOCAL_OUTPUT_DIR
)

warnings.filterwarnings("ignore")
os.makedirs(LOCAL_LOG_DIR,    exist_ok=True)
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOCAL_LOG_DIR + "visualisation.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight"
})
sns.set_theme(style="whitegrid")


def get_mongo_client():
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    log.info("Connected to MongoDB Atlas")
    return client


def load_collection(client, collection_name, query=None):
    db   = client[MONGO_DB]
    col  = db[collection_name]
    docs = list(col.find(query or {}, {"_id": 0}))
    df   = pd.DataFrame(docs)
    log.info("Loaded %d records from '%s'", len(df), collection_name)
    return df


def plot_analysis1(client):
    log.info("Generating Analysis 1 visualisations")
    cat_df = load_collection(client, MONGO_COL_BEHAVIOUR, {"analysis": "category_behaviour"})
    if not cat_df.empty and "total_revenue" in cat_df.columns:
        top_cats = (
            cat_df.dropna(subset=["category_top"])
            .groupby("category_top")["total_revenue"].sum()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        colors  = sns.color_palette("Blues_d", len(top_cats))
        bars    = ax.barh(top_cats.index[::-1], top_cats.values[::-1], color=colors[::-1])
        ax.set_xlabel("Total Revenue (EUR)")
        ax.set_title("Fig 2. Top 10 Product Categories by Total Revenue")
        for bar, val in zip(bars, top_cats.values[::-1]):
            ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                    f"EUR {val:,.0f}", va="center", fontsize=8)
        plt.tight_layout()
        path = LOCAL_OUTPUT_DIR + "fig2_top_categories.png"
        plt.savefig(path)
        plt.close()
        log.info("Saved: %s", path)

    hourly_df = load_collection(client, MONGO_COL_BEHAVIOUR, {"analysis": "hourly_patterns"})
    daily_df  = load_collection(client, MONGO_COL_BEHAVIOUR, {"analysis": "daily_patterns"})
    if not hourly_df.empty and not daily_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if "event_hour" in hourly_df.columns and "purchase_count" in hourly_df.columns:
            hourly_sorted = hourly_df.sort_values("event_hour")
            axes[0].plot(hourly_sorted["event_hour"], hourly_sorted["purchase_count"],
                         marker="o", color="#1f77b4", linewidth=2)
            axes[0].fill_between(hourly_sorted["event_hour"], hourly_sorted["purchase_count"],
                                 alpha=0.15, color="#1f77b4")
            axes[0].set_xlabel("Hour of Day")
            axes[0].set_ylabel("Purchase Count")
            axes[0].set_title("Fig 3a. Purchase Volume by Hour of Day")
            axes[0].set_xticks(range(0, 24, 2))
        if "weekday_name" in daily_df.columns and "purchase_count" in daily_df.columns:
            day_order    = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_sorted = (
                daily_df.set_index("weekday_name")
                .reindex([d for d in day_order if d in daily_df["weekday_name"].values])
                .reset_index()
            )
            bar_colors = [
                "#d73027" if d in ["Saturday", "Sunday"] else "#4575b4"
                for d in daily_sorted["weekday_name"]
            ]
            axes[1].bar(daily_sorted["weekday_name"], daily_sorted["purchase_count"], color=bar_colors)
            axes[1].set_xlabel("Day of Week")
            axes[1].set_ylabel("Purchase Count")
            axes[1].set_title("Fig 3b. Purchase Volume by Day of Week (Red = Weekend)")
            axes[1].tick_params(axis="x", rotation=30)
        plt.tight_layout()
        path = LOCAL_OUTPUT_DIR + "fig3_temporal_patterns.png"
        plt.savefig(path)
        plt.close()
        log.info("Saved: %s", path)


def plot_analysis2(client):
    log.info("Generating Analysis 2 visualisations")
    rating_df = load_collection(client, MONGO_COL_SENTIMENT, {"analysis": "rating_distribution"})
    if not rating_df.empty and "rating" in rating_df.columns:
        fig, ax   = plt.subplots(figsize=(8, 5))
        color_map = {1: "#d73027", 2: "#f46d43", 3: "#fdae61", 4: "#74add1", 5: "#313695"}
        rating_sorted = rating_df.sort_values("rating")
        bar_colors    = [color_map.get(int(r), "grey") for r in rating_sorted["rating"]]
        ax.bar(rating_sorted["rating"].astype(str), rating_sorted["review_count"], color=bar_colors)
        ax.set_xlabel("Star Rating")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Fig 4. Distribution of Star Ratings")
        for i, (r, c) in enumerate(zip(rating_sorted["rating"], rating_sorted["review_count"])):
            ax.text(i, c * 1.01, f"{c:,}", ha="center", fontsize=8)
        plt.tight_layout()
        path = LOCAL_OUTPUT_DIR + "fig4_rating_distribution.png"
        plt.savefig(path)
        plt.close()
        log.info("Saved: %s", path)

    kw_df = load_collection(client, MONGO_COL_SENTIMENT, {"analysis": "top_keywords"})
    if not kw_df.empty and "word" in kw_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Fig 5. Top 20 Keywords in Positive vs Negative Reviews",
                     fontsize=13, fontweight="bold")
        for group, color, ax in zip(
            ["positive", "negative"],
            ["#2ecc71", "#e74c3c"],
            axes
        ):
            group_df = kw_df[kw_df["sentiment_group"] == group].nlargest(20, "frequency")
            if not group_df.empty:
                ax.barh(group_df["word"][::-1], group_df["frequency"][::-1],
                        color=color, alpha=0.85)
                ax.set_title("Positive Reviews" if group == "positive" else "Negative Reviews")
                ax.set_xlabel("Frequency")
        plt.tight_layout()
        path = LOCAL_OUTPUT_DIR + "fig5_top_keywords.png"
        plt.savefig(path)
        plt.close()
        log.info("Saved: %s", path)


def plot_analysis3(client):
    log.info("Generating Analysis 3 visualisations")
    rain_df = load_collection(client, MONGO_COL_WEATHER, {"analysis": "rain_analysis"})
    if not rain_df.empty and "rain_label" in rain_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Fig 7. Impact of Rainfall on E-Commerce Activity",
                     fontsize=13, fontweight="bold")
        rain_sorted = rain_df.sort_values("rain_day")
        bar_colors  = ["#f39c12", "#3498db"]
        axes[0].bar(rain_sorted["rain_label"], rain_sorted["avg_daily_purchases"], color=bar_colors)
        axes[0].set_title("Average Daily Purchases")
        axes[0].set_ylabel("Purchases")
        for i, (label, val) in enumerate(
            zip(rain_sorted["rain_label"], rain_sorted["avg_daily_purchases"])
        ):
            axes[0].text(i, val * 1.01, f"{val:,.0f}", ha="center", fontweight="bold")
        axes[1].bar(rain_sorted["rain_label"], rain_sorted["avg_daily_revenue"], color=bar_colors)
        axes[1].set_title("Average Daily Revenue (EUR)")
        axes[1].set_ylabel("Revenue (EUR)")
        for i, (label, val) in enumerate(
            zip(rain_sorted["rain_label"], rain_sorted["avg_daily_revenue"])
        ):
            axes[1].text(i, val * 1.01, f"EUR {val:,.0f}", ha="center", fontweight="bold")
        plt.tight_layout()
        path = LOCAL_OUTPUT_DIR + "fig7_rain_analysis.png"
        plt.savefig(path)
        plt.close()
        log.info("Saved: %s", path)

    ts_df = load_collection(client, MONGO_COL_WEATHER, {"analysis": "daily_joined_data"})
    if not ts_df.empty and "event_date" in ts_df.columns:
        ts_df["event_date"] = pd.to_datetime(ts_df["event_date"])
        ts_sorted = ts_df.sort_values("event_date")
        fig, ax1  = plt.subplots(figsize=(14, 5))
        ax1.plot(ts_sorted["event_date"], ts_sorted["purchase_count"],
                 color="#1f77b4", linewidth=2, label="Daily Purchases")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Purchase Count", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        ax2.plot(ts_sorted["event_date"], ts_sorted["avg_temperature"],
                 color="#d62728", linewidth=1.5, linestyle="--", label="Avg Temperature")
        ax2.set_ylabel("Average Temperature (Celsius)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.set_title("Fig 8. Daily Purchase Volume vs Average Temperature", fontweight="bold")
        fig.autofmt_xdate()
        plt.tight_layout()
        path = LOCAL_OUTPUT_DIR + "fig8_purchases_vs_temperature.png"
        plt.savefig(path)
        plt.close()
        log.info("Saved: %s", path)


def main():
    log.info("=" * 60)
    log.info("Step 4: Follow-up Analysis and Visualisation")
    log.info("=" * 60)
    client = get_mongo_client()
    try:
        plot_analysis1(client)
        plot_analysis2(client)
        plot_analysis3(client)
        log.info("All figures saved to: %s", LOCAL_OUTPUT_DIR)
    except Exception as e:
        log.error("Visualisation failed: %s", e)
        raise
    finally:
        client.close()
        log.info("MongoDB connection closed.")


if __name__ == "__main__":
    main()
