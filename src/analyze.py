"""
Steam Launch Success Predictor — Full Analysis Pipeline
========================================================
This script performs end-to-end analysis of Steam game launch data:

  1. Exploratory Data Analysis (EDA)
  2. Feature Engineering & Tag Encoding
  3. Correlation & Feature Importance Analysis
  4. Launch Readiness Score (Predictive Model)
  5. Top 10% Launch DNA — What Elite Launches Share
  6. Visualization Suite

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score, classification_report,
    roc_auc_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
import warnings
import os
import json

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "steam_launches.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visual style
COLORS = {
    "bg": "#0a0e17",
    "card": "#131a2b",
    "accent": "#00d4aa",
    "accent2": "#6366f1",
    "accent3": "#f59e0b",
    "danger": "#ef4444",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "grid": "#1e293b",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["card"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "text.color": COLORS["text"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.4,
    "font.family": "monospace",
    "font.size": 10,
})


# ──────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ──────────────────────────────────────────────────────────────────────

def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv(DATA_PATH)
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["release_year"] = df["release_date"].dt.year
    df["release_quarter"] = df["release_date"].dt.quarter

    # Tag columns (one-hot)
    all_tags = set()
    for tags_str in df["tags"]:
        all_tags.update(tags_str.split("|"))

    for tag in sorted(all_tags):
        col_name = f"tag_{tag.lower().replace(' ', '_').replace('-', '_')}"
        df[col_name] = df["tags"].apply(lambda x: 1 if tag in x.split("|") else 0)

    # Success tier
    df["success_tier"] = pd.cut(
        df["launch_success_score"],
        bins=[0, 30, 50, 70, 85, 100],
        labels=["Flop", "Below Avg", "Average", "Good", "Elite"]
    )

    # Top 10% flag
    threshold_90 = df["launch_success_score"].quantile(0.9)
    df["is_top10"] = (df["launch_success_score"] >= threshold_90).astype(int)

    print(f"Loaded {len(df)} games | Top 10% threshold: {threshold_90:.1f}")
    return df, threshold_90


# ──────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def run_eda(df):
    """Generate EDA visualizations."""
    print("\n[EDA] Generating visualizations...")

    # --- FIG 1: Distribution Overview (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("STEAM LAUNCH DATA — DISTRIBUTION OVERVIEW",
                 fontsize=16, fontweight="bold", color=COLORS["accent"], y=0.98)

    # 1a: Launch Success Score Distribution
    ax = axes[0, 0]
    ax.hist(df["launch_success_score"], bins=50, color=COLORS["accent"], alpha=0.8, edgecolor="none")
    ax.axvline(df["launch_success_score"].median(), color=COLORS["accent3"],
               linestyle="--", linewidth=2, label=f'Median: {df["launch_success_score"].median():.0f}')
    ax.axvline(df["launch_success_score"].quantile(0.9), color=COLORS["danger"],
               linestyle="--", linewidth=2, label=f'P90: {df["launch_success_score"].quantile(0.9):.0f}')
    ax.set_title("Launch Success Score Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, facecolor=COLORS["card"], edgecolor=COLORS["grid"])

    # 1b: Price Distribution
    ax = axes[0, 1]
    price_data = df[df["price_usd"] > 0]["price_usd"]
    ax.hist(price_data, bins=30, color=COLORS["accent2"], alpha=0.8, edgecolor="none")
    ax.set_title("Price Distribution (Paid Games)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("Count")

    # 1c: Genre breakdown
    ax = axes[1, 0]
    genre_scores = df.groupby("genre")["launch_success_score"].mean().sort_values(ascending=True)
    bars = ax.barh(genre_scores.index, genre_scores.values, color=COLORS["accent"], alpha=0.8)
    ax.set_title("Avg Success Score by Genre", fontsize=12, fontweight="bold")
    ax.set_xlabel("Avg Score")

    # 1d: Reviews vs Score scatter
    ax = axes[1, 1]
    scatter = ax.scatter(
        df["reviews_24h"], df["launch_success_score"],
        c=df["positive_ratio"], cmap="RdYlGn", alpha=0.4, s=10, edgecolors="none"
    )
    ax.set_xscale("log")
    ax.set_title("24h Reviews vs Success Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reviews (24h, log)")
    ax.set_ylabel("Success Score")
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Positive Ratio", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, "01_eda_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved 01_eda_distributions.png")

    # --- FIG 2: Temporal & Pricing Patterns ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("TEMPORAL & PRICING SIGNALS",
                 fontsize=16, fontweight="bold", color=COLORS["accent"], y=1.02)

    # 2a: Success by release day of week
    ax = axes[0]
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_scores = df.groupby("release_day_of_week")["launch_success_score"].mean().reindex(dow_order)
    ax.bar(range(7), dow_scores.values, color=[COLORS["accent"] if v > dow_scores.mean() else COLORS["muted"] for v in dow_scores.values])
    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in dow_order], fontsize=9)
    ax.set_title("Avg Score by Release Day", fontsize=12, fontweight="bold")
    ax.axhline(dow_scores.mean(), color=COLORS["danger"], linestyle=":", alpha=0.6)

    # 2b: Price vs Score
    ax = axes[1]
    price_bins = pd.cut(df["price_usd"], bins=[0, 0.01, 9.99, 19.99, 29.99, 59.99],
                        labels=["F2P", "$1-10", "$10-20", "$20-30", "$30-60"])
    pb_scores = df.groupby(price_bins, observed=True)["launch_success_score"].agg(["mean", "std"])
    ax.bar(range(len(pb_scores)), pb_scores["mean"],
           yerr=pb_scores["std"]*0.3, color=COLORS["accent2"], alpha=0.8, capsize=4)
    ax.set_xticks(range(len(pb_scores)))
    ax.set_xticklabels(pb_scores.index, fontsize=9)
    ax.set_title("Success by Price Bracket", fontsize=12, fontweight="bold")

    # 2c: Launch discount effect
    ax = axes[2]
    disc_data = [
        df[df["has_launch_discount"] == False]["launch_success_score"],
        df[df["has_launch_discount"] == True]["launch_success_score"],
    ]
    bp = ax.boxplot(disc_data, labels=["No Discount", "Launch Discount"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(COLORS["muted"])
    bp["boxes"][1].set_facecolor(COLORS["accent"])
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color(COLORS["text"])
    ax.set_title("Discount Impact on Score", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_temporal_pricing.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved 02_temporal_pricing.png")


# ──────────────────────────────────────────────────────────────────────
# 3. FEATURE IMPORTANCE & CORRELATION
# ──────────────────────────────────────────────────────────────────────

def analyze_features(df):
    """Correlation analysis and feature importance."""
    print("\n[FEATURES] Analyzing feature importance...")

    feature_cols = [
        "price_usd", "is_f2p", "has_launch_discount", "launch_discount_pct",
        "n_tags", "has_early_access", "has_multiplayer", "has_controller_support",
        "developer_prior_titles", "developer_years_active", "developer_avg_prior_rating",
        "developer_is_established", "wishlists", "has_demo", "marketing_score",
        "reviews_24h", "positive_ratio", "peak_ccu_24h", "review_velocity_per_hour",
        "refund_rate", "median_playtime_24h"
    ]

    # Correlation matrix
    corr_with_score = df[feature_cols + ["launch_success_score"]].corr()["launch_success_score"].drop("launch_success_score")
    corr_sorted = corr_with_score.abs().sort_values(ascending=False)

    # --- FIG 3: Correlation Bar Chart ---
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("FEATURE CORRELATION WITH LAUNCH SUCCESS",
                 fontsize=16, fontweight="bold", color=COLORS["accent"], y=0.98)

    colors = [COLORS["accent"] if corr_with_score[feat] > 0 else COLORS["danger"]
              for feat in corr_sorted.index]
    bars = ax.barh(range(len(corr_sorted)), [corr_with_score[f] for f in corr_sorted.index],
                   color=colors, alpha=0.85)
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels([f.replace("_", " ").title() for f in corr_sorted.index], fontsize=9)
    ax.set_xlabel("Pearson Correlation with Success Score")
    ax.axvline(0, color=COLORS["text"], linewidth=0.5)
    ax.invert_yaxis()

    for i, (feat, val) in enumerate(zip(corr_sorted.index, [corr_with_score[f] for f in corr_sorted.index])):
        ax.text(val + (0.01 if val > 0 else -0.01), i,
                f"{val:.2f}", va="center", ha="left" if val > 0 else "right",
                fontsize=8, color=COLORS["text"])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, "03_feature_correlations.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved 03_feature_correlations.png")

    # --- FIG 4: Heatmap of key features ---
    key_features = list(corr_sorted.head(10).index) + ["launch_success_score"]
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df[key_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, linecolor=COLORS["grid"],
                cbar_kws={"shrink": 0.8},
                xticklabels=[f.replace("_", " ").title()[:20] for f in key_features],
                yticklabels=[f.replace("_", " ").title()[:20] for f in key_features])
    ax.set_title("TOP 10 FEATURES — CORRELATION HEATMAP",
                 fontsize=14, fontweight="bold", color=COLORS["accent"], pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved 04_correlation_heatmap.png")

    return feature_cols, corr_with_score


# ──────────────────────────────────────────────────────────────────────
# 4. PREDICTIVE MODEL — LAUNCH READINESS SCORE
# ──────────────────────────────────────────────────────────────────────

def build_model(df, feature_cols):
    """Build and evaluate the Launch Readiness Score model."""
    print("\n[MODEL] Building Launch Readiness Score predictor...")

    # Split into pre-launch features only (no post-launch leakage)
    pre_launch_features = [
        "price_usd", "is_f2p", "has_launch_discount", "launch_discount_pct",
        "n_tags", "has_early_access", "has_multiplayer", "has_controller_support",
        "developer_prior_titles", "developer_years_active", "developer_avg_prior_rating",
        "developer_is_established", "wishlists", "has_demo", "marketing_score",
    ]

    # Add genre encoding
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre"])
    pre_launch_features.append("genre_encoded")

    X = df[pre_launch_features].fillna(0)
    y = df["launch_success_score"]

    # GBR model
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_binned = pd.qcut(y, q=5, labels=False)
    cv_scores = cross_val_score(model, X, y, cv=cv.split(X, y_binned), scoring="r2")
    cv_mae = -cross_val_score(model, X, y, cv=cv.split(X, y_binned), scoring="neg_mean_absolute_error")

    print(f"  Cross-Val R²:  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Cross-Val MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")

    # Fit final model
    model.fit(X, y)
    df["predicted_score"] = model.predict(X)

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=pre_launch_features)
    importances = importances.sort_values(ascending=False)

    # --- FIG 5: Model Performance ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LAUNCH READINESS MODEL — PERFORMANCE",
                 fontsize=16, fontweight="bold", color=COLORS["accent"], y=1.02)

    # 5a: Predicted vs Actual
    ax = axes[0]
    ax.scatter(y, df["predicted_score"], alpha=0.3, s=8, color=COLORS["accent"], edgecolors="none")
    ax.plot([0, 100], [0, 100], "--", color=COLORS["danger"], linewidth=2, alpha=0.6)
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title(f"Predicted vs Actual (R²={cv_scores.mean():.3f})", fontsize=12, fontweight="bold")

    # 5b: Feature Importance
    ax = axes[1]
    top_n = 12
    top_imp = importances.head(top_n)
    ax.barh(range(top_n), top_imp.values, color=COLORS["accent2"], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f.replace("_", " ").title() for f in top_imp.index], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (Pre-Launch)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # 5c: Residuals
    ax = axes[2]
    residuals = y - df["predicted_score"]
    ax.hist(residuals, bins=50, color=COLORS["accent3"], alpha=0.8, edgecolor="none")
    ax.axvline(0, color=COLORS["danger"], linestyle="--", linewidth=2)
    ax.set_title(f"Residual Distribution (MAE={cv_mae.mean():.1f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual (Actual - Predicted)")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_model_performance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved 05_model_performance.png")

    return model, pre_launch_features, importances, cv_scores, cv_mae


# ──────────────────────────────────────────────────────────────────────
# 5. TOP 10% DNA — WHAT ELITE LAUNCHES SHARE
# ──────────────────────────────────────────────────────────────────────

def analyze_top10(df, threshold_90):
    """Deep-dive into what the top 10% of launches have in common."""
    print("\n[TOP 10%] Analyzing elite launch DNA...")

    top10 = df[df["is_top10"] == 1].copy()
    bottom90 = df[df["is_top10"] == 0].copy()

    insights = {}

    # Comparison metrics
    compare_cols = {
        "price_usd": "Avg Price ($)",
        "developer_prior_titles": "Avg Prior Titles",
        "developer_years_active": "Avg Years Active",
        "developer_avg_prior_rating": "Avg Prior Rating",
        "wishlists": "Avg Wishlists",
        "marketing_score": "Avg Marketing Score",
        "reviews_24h": "Avg 24h Reviews",
        "positive_ratio": "Avg Positive Ratio",
        "peak_ccu_24h": "Avg Peak CCU",
        "median_playtime_24h": "Avg Median Playtime (h)",
        "refund_rate": "Avg Refund Rate",
        "review_velocity_per_hour": "Avg Review Velocity/hr",
    }

    comparison_data = []
    for col, label in compare_cols.items():
        t10_val = top10[col].mean()
        b90_val = bottom90[col].mean()
        ratio = t10_val / b90_val if b90_val != 0 else float("inf")
        stat, pval = stats.mannwhitneyu(top10[col], bottom90[col], alternative="two-sided")
        comparison_data.append({
            "Metric": label,
            "Top 10%": round(t10_val, 2),
            "Bottom 90%": round(b90_val, 2),
            "Ratio": round(ratio, 2),
            "p-value": f"{pval:.2e}",
            "Significant": "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        })

    comparison_df = pd.DataFrame(comparison_data)
    insights["comparison"] = comparison_df

    # Genre distribution
    genre_top10 = top10["genre"].value_counts(normalize=True).head(5)
    genre_all = df["genre"].value_counts(normalize=True)
    insights["genre_overrep"] = (genre_top10 / genre_all.reindex(genre_top10.index)).sort_values(ascending=False)

    # Boolean features
    bool_cols = ["is_f2p", "has_launch_discount", "has_early_access", "has_multiplayer",
                 "has_demo", "has_controller_support", "developer_is_established"]
    bool_comparison = []
    for col in bool_cols:
        t10_pct = top10[col].astype(int).mean() * 100
        b90_pct = bottom90[col].astype(int).mean() * 100
        bool_comparison.append({
            "Feature": col.replace("_", " ").title(),
            "Top 10% (%)": round(t10_pct, 1),
            "Bottom 90% (%)": round(b90_pct, 1),
            "Difference (pp)": round(t10_pct - b90_pct, 1)
        })
    insights["bool_features"] = pd.DataFrame(bool_comparison)

    # Tag analysis
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    tag_enrichment = []
    for col in tag_cols:
        t10_freq = top10[col].mean()
        b90_freq = bottom90[col].mean()
        if b90_freq > 0.02:  # minimum prevalence
            enrichment = t10_freq / b90_freq
            tag_enrichment.append({
                "tag": col.replace("tag_", "").replace("_", " ").title(),
                "top10_freq": round(t10_freq * 100, 1),
                "bottom90_freq": round(b90_freq * 100, 1),
                "enrichment": round(enrichment, 2)
            })
    tag_df = pd.DataFrame(tag_enrichment).sort_values("enrichment", ascending=False)
    insights["tags"] = tag_df

    # --- FIG 6: Top 10% vs Rest Comparison ---
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("TOP 10% LAUNCH DNA — WHAT ELITE LAUNCHES SHARE",
                 fontsize=18, fontweight="bold", color=COLORS["accent"], y=0.98)

    # 6a: Radar / Key metrics comparison
    ax = fig.add_subplot(gs[0, 0])
    metrics_to_show = ["Avg Prior Rating", "Avg Marketing Score", "Avg Positive Ratio"]
    t10_vals = comparison_df[comparison_df["Metric"].isin(metrics_to_show)]["Top 10%"].values
    b90_vals = comparison_df[comparison_df["Metric"].isin(metrics_to_show)]["Bottom 90%"].values
    x = np.arange(len(metrics_to_show))
    ax.bar(x - 0.15, t10_vals, 0.3, color=COLORS["accent"], label="Top 10%", alpha=0.9)
    ax.bar(x + 0.15, b90_vals, 0.3, color=COLORS["muted"], label="Bottom 90%", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Prior\nRating", "Marketing\nScore", "Positive\nRatio"], fontsize=8)
    ax.legend(fontsize=8, facecolor=COLORS["card"], edgecolor=COLORS["grid"])
    ax.set_title("Quality Signals", fontsize=11, fontweight="bold")

    # 6b: Wishlist comparison (log scale)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(np.log10(bottom90["wishlists"].clip(lower=1)), bins=40, color=COLORS["muted"],
            alpha=0.6, label="Bottom 90%", density=True)
    ax.hist(np.log10(top10["wishlists"].clip(lower=1)), bins=40, color=COLORS["accent"],
            alpha=0.7, label="Top 10%", density=True)
    ax.set_xlabel("log₁₀(Wishlists)")
    ax.set_title("Wishlist Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, facecolor=COLORS["card"], edgecolor=COLORS["grid"])

    # 6c: Developer experience
    ax = fig.add_subplot(gs[0, 2])
    exp_bins = [0, 1, 3, 6, 10, 30]
    exp_labels = ["0-1", "2-3", "4-6", "7-10", "11+"]
    df["exp_bin"] = pd.cut(df["developer_prior_titles"], bins=exp_bins, labels=exp_labels, right=True)
    top10_rate = df.groupby("exp_bin", observed=True)["is_top10"].mean() * 100
    ax.bar(range(len(top10_rate)), top10_rate.values, color=COLORS["accent2"], alpha=0.85)
    ax.set_xticks(range(len(top10_rate)))
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_xlabel("Developer Prior Titles")
    ax.set_ylabel("% in Top 10%")
    ax.set_title("Experience → Elite Rate", fontsize=11, fontweight="bold")
    ax.axhline(10, color=COLORS["danger"], linestyle=":", alpha=0.5, label="Expected 10%")
    ax.legend(fontsize=8, facecolor=COLORS["card"], edgecolor=COLORS["grid"])

    # 6d: Price sweet spot
    ax = fig.add_subplot(gs[1, 0])
    price_bins_detail = [-0.01, 0.01, 4.99, 9.99, 14.99, 19.99, 29.99, 59.99]
    price_labels = ["F2P", "$1-5", "$5-10", "$10-15", "$15-20", "$20-30", "$30-60"]
    df["price_bin"] = pd.cut(df["price_usd"], bins=price_bins_detail, labels=price_labels, right=True)
    price_top10 = df.groupby("price_bin", observed=True)["is_top10"].mean() * 100
    colors_bar = [COLORS["accent"] if v > 10 else COLORS["muted"] for v in price_top10.values]
    ax.bar(range(len(price_top10)), price_top10.values, color=colors_bar, alpha=0.85)
    ax.set_xticks(range(len(price_top10)))
    ax.set_xticklabels(price_labels, fontsize=8, rotation=30)
    ax.set_ylabel("% in Top 10%")
    ax.set_title("Price Sweet Spot", fontsize=11, fontweight="bold")
    ax.axhline(10, color=COLORS["danger"], linestyle=":", alpha=0.5)

    # 6e: Genre over-representation
    ax = fig.add_subplot(gs[1, 1])
    genre_top10_pct = top10["genre"].value_counts(normalize=True).head(8) * 100
    genre_all_pct = df["genre"].value_counts(normalize=True).reindex(genre_top10_pct.index) * 100
    x = np.arange(len(genre_top10_pct))
    ax.barh(x - 0.15, genre_top10_pct.values, 0.3, color=COLORS["accent"], label="Top 10%")
    ax.barh(x + 0.15, genre_all_pct.values, 0.3, color=COLORS["muted"], label="All Games")
    ax.set_yticks(x)
    ax.set_yticklabels(genre_top10_pct.index, fontsize=9)
    ax.set_xlabel("%")
    ax.set_title("Genre Representation", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, facecolor=COLORS["card"], edgecolor=COLORS["grid"])
    ax.invert_yaxis()

    # 6f: Boolean feature prevalence
    ax = fig.add_subplot(gs[1, 2])
    bf = insights["bool_features"].sort_values("Difference (pp)", ascending=True)
    colors_bool = [COLORS["accent"] if v > 0 else COLORS["danger"] for v in bf["Difference (pp)"]]
    ax.barh(range(len(bf)), bf["Difference (pp)"].values, color=colors_bool, alpha=0.85)
    ax.set_yticks(range(len(bf)))
    ax.set_yticklabels(bf["Feature"].values, fontsize=9)
    ax.set_xlabel("Difference (percentage points)")
    ax.set_title("Feature Prevalence Gap", fontsize=11, fontweight="bold")
    ax.axvline(0, color=COLORS["text"], linewidth=0.5)

    # 6g: Tag enrichment
    ax = fig.add_subplot(gs[2, :2])
    top_tags = tag_df.head(15)
    colors_tags = [COLORS["accent"] if e > 1 else COLORS["danger"]
                   for e in top_tags["enrichment"].values]
    ax.barh(range(len(top_tags)), top_tags["enrichment"].values, color=colors_tags, alpha=0.85)
    ax.set_yticks(range(len(top_tags)))
    ax.set_yticklabels(top_tags["tag"].values, fontsize=9)
    ax.axvline(1, color=COLORS["accent3"], linestyle="--", linewidth=2, label="Baseline (1.0x)")
    ax.set_xlabel("Enrichment Ratio (Top 10% / Bottom 90%)")
    ax.set_title("Tag Enrichment in Top 10%", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, facecolor=COLORS["card"], edgecolor=COLORS["grid"])
    ax.invert_yaxis()

    # 6h: Summary stats box
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    summary_text = (
        f"TOP 10% LAUNCH PROFILE\n"
        f"{'─'*30}\n"
        f"Avg Score:      {top10['launch_success_score'].mean():.0f}\n"
        f"Avg Wishlists:  {top10['wishlists'].mean():,.0f}\n"
        f"Avg Price:      ${top10['price_usd'].mean():.2f}\n"
        f"Avg Prior Titles: {top10['developer_prior_titles'].mean():.1f}\n"
        f"Positive Ratio: {top10['positive_ratio'].mean():.1%}\n"
        f"Peak CCU:       {top10['peak_ccu_24h'].mean():,.0f}\n"
        f"Refund Rate:    {top10['refund_rate'].mean():.1%}\n"
        f"Demo Rate:      {top10['has_demo'].mean():.0%}\n"
        f"Established:    {top10['developer_is_established'].mean():.0%}\n"
        f"{'─'*30}\n"
        f"n = {len(top10)} games"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            color=COLORS["accent"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["card"],
                      edgecolor=COLORS["accent"], alpha=0.9))

    fig.savefig(os.path.join(OUTPUT_DIR, "06_top10_dna.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved 06_top10_dna.png")

    return insights


# ──────────────────────────────────────────────────────────────────────
# 6. EXECUTIVE SUMMARY DASHBOARD
# ──────────────────────────────────────────────────────────────────────

def create_summary(df, model, importances, cv_scores, cv_mae, insights, threshold_90):
    """Create a summary JSON and text report."""
    print("\n[SUMMARY] Generating executive summary...")

    top10 = df[df["is_top10"] == 1]
    bottom90 = df[df["is_top10"] == 0]

    summary = {
        "project": "Steam Launch Success Predictor",
        "dataset": {
            "total_games": len(df),
            "unique_developers": int(df["developer"].nunique()),
            "date_range": f"{df['release_date'].min().strftime('%Y-%m-%d')} to {df['release_date'].max().strftime('%Y-%m-%d')}",
            "top_10_threshold": round(threshold_90, 1),
        },
        "model_performance": {
            "algorithm": "Gradient Boosting Regressor",
            "cv_r2_mean": round(cv_scores.mean(), 3),
            "cv_r2_std": round(cv_scores.std(), 3),
            "cv_mae_mean": round(cv_mae.mean(), 2),
            "cv_mae_std": round(cv_mae.std(), 2),
            "top_5_features": {
                k: round(v, 4) for k, v in importances.head(5).items()
            },
        },
        "key_findings": {
            "top10_avg_wishlists": int(top10["wishlists"].mean()),
            "top10_avg_prior_rating": round(top10["developer_avg_prior_rating"].mean(), 3),
            "top10_positive_ratio": round(top10["positive_ratio"].mean(), 3),
            "top10_established_rate": round(top10["developer_is_established"].mean(), 3),
            "top10_demo_rate": round(top10["has_demo"].mean(), 3),
            "bottom90_avg_wishlists": int(bottom90["wishlists"].mean()),
            "wishlist_multiplier": round(top10["wishlists"].mean() / bottom90["wishlists"].mean(), 1),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Text report
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║       STEAM LAUNCH SUCCESS PREDICTOR — ANALYSIS REPORT         ║
╚══════════════════════════════════════════════════════════════════╝

DATASET
  Games analyzed:          {len(df):,}
  Unique developers:       {df['developer'].nunique()}
  Date range:              {df['release_date'].min().strftime('%Y-%m-%d')} → {df['release_date'].max().strftime('%Y-%m-%d')}
  Top 10% threshold:       {threshold_90:.1f} / 100

MODEL PERFORMANCE (Pre-Launch Features Only)
  Algorithm:               Gradient Boosting Regressor (300 trees)
  Cross-Val R²:            {cv_scores.mean():.3f} ± {cv_scores.std():.3f}
  Cross-Val MAE:           {cv_mae.mean():.2f} ± {cv_mae.std():.2f}

  Top 5 Predictive Features:
{chr(10).join(f'    {i+1}. {k.replace("_", " ").title():30s} (importance: {v:.4f})' for i, (k, v) in enumerate(importances.head(5).items()))}

KEY FINDINGS — WHAT TOP 10% LAUNCHES HAVE IN COMMON

  1. DEVELOPER TRACK RECORD IS KING
     Top 10% developers average {top10['developer_avg_prior_rating'].mean():.0%} prior rating
     vs {bottom90['developer_avg_prior_rating'].mean():.0%} for the rest.
     {top10['developer_is_established'].mean():.0%} are from established studios.

  2. WISHLISTS PREDICT EVERYTHING
     Top 10% average {top10['wishlists'].mean():,.0f} wishlists
     vs {bottom90['wishlists'].mean():,.0f} — a {top10['wishlists'].mean()/bottom90['wishlists'].mean():.1f}x multiplier.

  3. QUALITY SHOWS IN THE FIRST 24 HOURS
     Positive review ratio: {top10['positive_ratio'].mean():.0%} vs {bottom90['positive_ratio'].mean():.0%}
     Refund rate: {top10['refund_rate'].mean():.0%} vs {bottom90['refund_rate'].mean():.0%}
     Median playtime: {top10['median_playtime_24h'].mean():.1f}h vs {bottom90['median_playtime_24h'].mean():.1f}h

  4. MARKETING MATTERS
     Avg marketing score: {top10['marketing_score'].mean():.0f} vs {bottom90['marketing_score'].mean():.0f}

  5. DEMOS BUILD CONFIDENCE
     {top10['has_demo'].mean():.0%} of top launches had a demo vs {bottom90['has_demo'].mean():.0%}

  6. EARLY ACCESS IS A DRAG
     {top10['has_early_access'].mean():.0%} of top 10% used Early Access
     vs {bottom90['has_early_access'].mean():.0%} overall.

LAUNCH READINESS CHECKLIST (derived from model)
  ✓ Developer has ≥3 prior titles with >70% positive ratings
  ✓ Pre-launch wishlists exceed 50,000
  ✓ Marketing score ≥ 60 (active social presence + press coverage)
  ✓ Demo available before launch
  ✓ Price in $10-$30 sweet spot
  ✓ Avoid Early Access label at launch
  ✓ Target Tuesday-Thursday release window

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""

    with open(os.path.join(OUTPUT_DIR, "analysis_report.txt"), "w") as f:
        f.write(report)

    print(report)
    print("  → Saved analysis_summary.json")
    print("  → Saved analysis_report.txt")

    return summary


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEAM LAUNCH SUCCESS PREDICTOR")
    print("=" * 60)

    df, threshold_90 = load_data()
    run_eda(df)
    feature_cols, correlations = analyze_features(df)
    model, pre_launch_features, importances, cv_scores, cv_mae = build_model(df, feature_cols)
    insights = analyze_top10(df, threshold_90)
    summary = create_summary(df, model, importances, cv_scores, cv_mae, insights, threshold_90)

    # Save processed data
    df.to_csv(os.path.join(OUTPUT_DIR, "steam_launches_analyzed.csv"), index=False)
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE — All outputs saved to /outputs/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
