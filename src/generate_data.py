"""
Steam Game Launch Data Generator
================================
Generates a realistic synthetic dataset of ~2,500 Steam game launches
with correlated features that mirror real-world patterns observed on SteamDB/SteamSpy.

Features generated:
  - Game metadata (title, developer, publisher, genre, release date)
  - Pricing signals (launch price, discount flag, F2P flag)
  - Developer history (prior titles, avg prior rating, years active)
  - Tag profile (up to 8 Steam tags per game)
  - Pre-launch signals (wishlists, demo availability, marketing spend proxy)
  - Post-launch (24h) metrics (reviews, positive ratio, peak CCU, refund rate)
  - Derived target: launch_success_score (0-100)

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import json
import os

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS & DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────

N_GAMES = 2500

GENRES = [
    "Action", "Adventure", "RPG", "Strategy", "Simulation",
    "Indie", "Casual", "Sports", "Racing", "Puzzle",
    "Horror", "Platformer", "Roguelike", "Visual Novel", "Survival"
]

GENRE_WEIGHTS = [
    0.18, 0.12, 0.11, 0.09, 0.08,
    0.13, 0.07, 0.04, 0.03, 0.05,
    0.03, 0.03, 0.02, 0.01, 0.01
]

TAGS = [
    "Singleplayer", "Multiplayer", "Co-op", "Online Co-op",
    "Open World", "Story Rich", "Atmospheric", "Difficult",
    "Early Access", "Sandbox", "Crafting", "Pixel Graphics",
    "3D", "2D", "VR", "Controller Support", "Moddable",
    "Procedural Generation", "Base Building", "Turn-Based",
    "Real-Time", "Competitive", "PvP", "PvE",
    "Anime", "Retro", "Sci-Fi", "Fantasy", "Post-Apocalyptic",
    "Dark", "Cute", "Relaxing", "Fast-Paced", "Tactical",
    "Narrative", "Choices Matter", "Exploration", "Hack and Slash",
    "Souls-like", "Metroidvania", "Bullet Hell", "City Builder",
    "Management", "Tower Defense", "Card Game", "Deck Building",
    "Farming", "Fishing", "Cooking", "Music"
]

DEVELOPER_PREFIXES = [
    "Pixel", "Iron", "Neon", "Shadow", "Crystal", "Storm", "Lunar",
    "Frost", "Nova", "Ember", "Void", "Apex", "Nexus", "Rogue",
    "Titan", "Obsidian", "Crimson", "Azure", "Golden", "Silver"
]

DEVELOPER_SUFFIXES = [
    "Games", "Studios", "Interactive", "Entertainment", "Works",
    "Labs", "Forge", "Digital", "Collective", "Software",
    "Creations", "Media", "Arts", "Workshop", "Productions"
]

ADJECTIVES = [
    "Dark", "Lost", "Last", "Eternal", "Fallen", "Rising", "Ancient",
    "Forgotten", "Endless", "Hidden", "Silent", "Crimson", "Iron",
    "Shadow", "Crystal", "Golden", "Frost", "Thunder", "Neon", "Hollow"
]

NOUNS = [
    "Kingdom", "Legends", "Frontier", "Realm", "Odyssey", "Chronicles",
    "Horizon", "Legacy", "Depths", "Citadel", "Wasteland", "Haven",
    "Nexus", "Forge", "Sanctum", "Protocol", "Dominion", "Rift",
    "Eclipse", "Exodus", "Bastion", "Inferno", "Abyss", "Zenith"
]


def generate_game_title(idx):
    """Generate a plausible game title."""
    np.random.seed(idx * 7 + 13)
    pattern = np.random.choice(["adj_noun", "the_noun", "noun_of_noun", "single"])
    if pattern == "adj_noun":
        return f"{np.random.choice(ADJECTIVES)} {np.random.choice(NOUNS)}"
    elif pattern == "the_noun":
        return f"The {np.random.choice(NOUNS)}"
    elif pattern == "noun_of_noun":
        return f"{np.random.choice(NOUNS)} of {np.random.choice(NOUNS)}"
    else:
        return np.random.choice(NOUNS)


def generate_developer_name(idx):
    """Generate a studio name."""
    np.random.seed(idx * 3 + 7)
    return f"{np.random.choice(DEVELOPER_PREFIXES)} {np.random.choice(DEVELOPER_SUFFIXES)}"


def generate_tags(genre, n_tags):
    """Generate a tag set correlated with the genre."""
    genre_tag_affinities = {
        "Action": ["Fast-Paced", "Hack and Slash", "3D", "Controller Support", "Competitive"],
        "Adventure": ["Story Rich", "Exploration", "Atmospheric", "Singleplayer", "Narrative"],
        "RPG": ["Story Rich", "Fantasy", "Choices Matter", "Open World", "Turn-Based"],
        "Strategy": ["Turn-Based", "Real-Time", "Tactical", "Management", "Base Building"],
        "Simulation": ["Management", "Sandbox", "Crafting", "Relaxing", "City Builder"],
        "Indie": ["Pixel Graphics", "2D", "Singleplayer", "Retro", "Atmospheric"],
        "Casual": ["Relaxing", "Cute", "2D", "Puzzle", "Singleplayer"],
        "Sports": ["Multiplayer", "Competitive", "Controller Support", "PvP", "3D"],
        "Racing": ["Fast-Paced", "Multiplayer", "Controller Support", "3D", "Competitive"],
        "Puzzle": ["Singleplayer", "Relaxing", "2D", "Atmospheric", "Narrative"],
        "Horror": ["Dark", "Atmospheric", "Singleplayer", "Story Rich", "3D"],
        "Platformer": ["2D", "Pixel Graphics", "Difficult", "Singleplayer", "Retro"],
        "Roguelike": ["Procedural Generation", "Difficult", "Pixel Graphics", "2D", "Fast-Paced"],
        "Visual Novel": ["Anime", "Narrative", "Choices Matter", "2D", "Story Rich"],
        "Survival": ["Crafting", "Open World", "Multiplayer", "Sandbox", "Base Building"],
    }
    affinity_tags = genre_tag_affinities.get(genre, [])
    # 60% chance each affinity tag is included
    selected = [t for t in affinity_tags if np.random.random() < 0.6]
    # fill remaining from general pool
    remaining = [t for t in TAGS if t not in selected]
    n_fill = max(0, n_tags - len(selected))
    selected += list(np.random.choice(remaining, size=min(n_fill, len(remaining)), replace=False))
    return selected[:n_tags]


# ──────────────────────────────────────────────────────────────────────
# MAIN GENERATION LOGIC
# ──────────────────────────────────────────────────────────────────────

def generate_dataset():
    """Generate the full synthetic dataset with realistic correlations."""
    records = []

    # Pre-generate ~200 unique developers with histories
    n_devs = 200
    dev_names = list(set(generate_developer_name(i) for i in range(n_devs * 2)))[:n_devs]
    dev_experience = {
        name: {
            "prior_titles": max(0, int(np.random.exponential(3))),
            "years_active": max(1, int(np.random.exponential(4) + 1)),
            "avg_prior_rating": np.clip(np.random.normal(0.68, 0.15), 0.2, 0.98),
            "is_established": np.random.random() < 0.25,
        }
        for name in dev_names
    }

    for i in range(N_GAMES):
        # --- Metadata ---
        title = generate_game_title(i)
        genre = np.random.choice(GENRES, p=GENRE_WEIGHTS)
        developer = np.random.choice(dev_names)
        dev_info = dev_experience[developer]
        release_date = datetime(2020, 1, 1) + timedelta(days=int(np.random.uniform(0, 365 * 4)))
        release_month = release_date.month
        release_dow = release_date.strftime("%A")

        # --- Pricing ---
        is_f2p = np.random.random() < 0.08
        if is_f2p:
            price_usd = 0.0
        else:
            price_bucket = np.random.choice(
                [4.99, 9.99, 14.99, 19.99, 24.99, 29.99, 39.99, 49.99, 59.99],
                p=[0.10, 0.18, 0.20, 0.18, 0.12, 0.08, 0.07, 0.04, 0.03]
            )
            price_usd = price_bucket
        has_launch_discount = (not is_f2p) and (np.random.random() < 0.35)
        launch_discount_pct = int(np.random.choice([10, 15, 20, 25])) if has_launch_discount else 0

        # --- Tags ---
        n_tags = np.random.randint(4, 9)
        tags = generate_tags(genre, n_tags)
        has_early_access = "Early Access" in tags
        has_multiplayer = any(t in tags for t in ["Multiplayer", "Co-op", "Online Co-op", "PvP"])
        has_controller = "Controller Support" in tags

        # --- Developer History ---
        prior_titles = dev_info["prior_titles"]
        years_active = dev_info["years_active"]
        avg_prior_rating = dev_info["avg_prior_rating"]
        is_established = dev_info["is_established"]

        # --- Pre-launch Signals ---
        # Wishlists: correlated with dev reputation + genre popularity + price
        wishlist_base = np.random.lognormal(mean=7.5, sigma=1.5)
        wishlist_mult = 1.0
        if is_established:
            wishlist_mult *= np.random.uniform(2.0, 5.0)
        if prior_titles > 5:
            wishlist_mult *= 1.5
        if genre in ["Action", "RPG", "Adventure"]:
            wishlist_mult *= 1.3
        if is_f2p:
            wishlist_mult *= 1.4
        wishlists = int(np.clip(wishlist_base * wishlist_mult, 50, 2_000_000))

        has_demo = np.random.random() < 0.18
        demo_conversion_rate = np.clip(np.random.normal(0.12, 0.05), 0.01, 0.4) if has_demo else None

        # Marketing presence (proxy: social media followers / press mentions)
        marketing_score = np.clip(np.random.normal(40, 25) +
                                  (20 if is_established else 0) +
                                  (10 if wishlists > 50000 else 0), 0, 100)

        # --- LAUNCH OUTCOME (first 24h) ---
        # Build a latent "quality" score that drives outcomes
        quality = np.clip(
            np.random.normal(50, 20)
            + (avg_prior_rating - 0.5) * 30        # dev track record
            + (10 if prior_titles > 3 else 0)       # experience bonus
            + (5 if has_demo else 0)                 # demo = confidence signal
            + (marketing_score - 50) * 0.3           # marketing effect
            - (10 if has_early_access else 0)        # EA penalty on reviews
            + np.random.normal(0, 10),               # noise
            5, 98
        )

        # Reviews in first 24h
        review_base = wishlists * np.random.uniform(0.005, 0.03)
        reviews_24h = int(np.clip(review_base * (quality / 50), 1, 50000))

        # Positive review ratio
        positive_ratio = np.clip(
            0.3 + (quality / 100) * 0.6 + np.random.normal(0, 0.05),
            0.10, 0.99
        )

        # Peak concurrent users
        peak_ccu = int(np.clip(
            wishlists * np.random.uniform(0.01, 0.08) * (quality / 50),
            5, 500_000
        ))

        # Review velocity (reviews per hour in first 24h)
        review_velocity = round(reviews_24h / 24, 2)

        # Refund rate (inversely correlated with quality)
        refund_rate = np.clip(
            0.30 - (quality / 100) * 0.25 + np.random.normal(0, 0.03),
            0.01, 0.50
        )

        # Median playtime in first 24h (hours)
        median_playtime_24h = np.clip(
            (quality / 100) * 4 + np.random.normal(0, 0.8),
            0.2, 12.0
        )

        # --- COMPOSITE LAUNCH SUCCESS SCORE ---
        # Weighted combination of normalized metrics
        score_components = {
            "review_sentiment": positive_ratio * 30,
            "review_volume": min(reviews_24h / 500, 1) * 20,
            "player_engagement": min(peak_ccu / 5000, 1) * 20,
            "retention_signal": min(median_playtime_24h / 3, 1) * 15,
            "low_refund": (1 - refund_rate) * 15,
        }
        launch_success_score = np.clip(sum(score_components.values()), 0, 100)

        records.append({
            "game_id": f"SG-{i+1:04d}",
            "title": title,
            "developer": developer,
            "genre": genre,
            "release_date": release_date.strftime("%Y-%m-%d"),
            "release_month": release_month,
            "release_day_of_week": release_dow,
            "price_usd": round(price_usd, 2),
            "is_f2p": is_f2p,
            "has_launch_discount": has_launch_discount,
            "launch_discount_pct": launch_discount_pct,
            "tags": "|".join(tags),
            "n_tags": n_tags,
            "has_early_access": has_early_access,
            "has_multiplayer": has_multiplayer,
            "has_controller_support": has_controller,
            "developer_prior_titles": prior_titles,
            "developer_years_active": years_active,
            "developer_avg_prior_rating": round(avg_prior_rating, 3),
            "developer_is_established": is_established,
            "wishlists": wishlists,
            "has_demo": has_demo,
            "demo_conversion_rate": round(demo_conversion_rate, 3) if demo_conversion_rate else None,
            "marketing_score": round(marketing_score, 1),
            "reviews_24h": reviews_24h,
            "positive_ratio": round(positive_ratio, 3),
            "peak_ccu_24h": peak_ccu,
            "review_velocity_per_hour": review_velocity,
            "refund_rate": round(refund_rate, 3),
            "median_playtime_24h": round(median_playtime_24h, 2),
            "launch_success_score": round(launch_success_score, 2),
        })

    df = pd.DataFrame(records)
    return df


def main():
    print("Generating Steam launch dataset...")
    df = generate_dataset()

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "steam_launches.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records → {out_path}")

    # Summary stats
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Games:             {len(df):,}")
    print(f"  Unique developers: {df['developer'].nunique()}")
    print(f"  Date range:        {df['release_date'].min()} → {df['release_date'].max()}")
    print(f"  Avg success score: {df['launch_success_score'].mean():.1f}")
    print(f"  Top 10% threshold: {df['launch_success_score'].quantile(0.9):.1f}")
    print(f"  Median price:      ${df['price_usd'].median():.2f}")
    print(f"  % F2P:             {df['is_f2p'].mean()*100:.1f}%")
    print(f"  % Early Access:    {df['has_early_access'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
