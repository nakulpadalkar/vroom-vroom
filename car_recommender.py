from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from data_store import load_inventory_from_database


DATA_PATH = Path("data/truecar_clean_combined.csv")
CACHE_DIR = Path("cache")

PRICE_TIER_SCORE = {
    "below_market": 1.0,
    "near_market": 0.68,
    "above_market": 0.18,
}

RELIABILITY_BY_MAKE = {
    "Toyota": 0.96,
    "Lexus": 0.95,
    "Honda": 0.92,
    "Acura": 0.88,
    "Mazda": 0.86,
    "Subaru": 0.82,
    "Hyundai": 0.78,
    "Kia": 0.76,
    "Nissan": 0.7,
    "Infiniti": 0.68,
    "Ford": 0.66,
    "Chevrolet": 0.64,
    "Buick": 0.63,
    "GMC": 0.62,
    "Volkswagen": 0.6,
    "Volvo": 0.59,
    "Jeep": 0.56,
    "BMW": 0.5,
    "Mercedes-Benz": 0.48,
    "Audi": 0.46,
    "Land Rover": 0.35,
    "Jaguar": 0.34,
    "Maserati": 0.28,
}

FEATURE_KEYWORDS = {
    "apple_carplay": ("apple carplay", "android auto"),
    "awd": ("awd", "all wheel", "4wd", "four wheel"),
    "backup_camera": ("backup camera", "rear camera", "exterior parking camera"),
    "blind_spot": ("blind spot",),
    "heated_seats": ("heated seat", "heated front"),
    "lane_assist": ("lane keeping", "lane departure", "lkas"),
    "moonroof": ("moonroof", "sunroof"),
    "third_row": ("third row", "3rd row"),
}

PRIORITY_OPTIONS = {
    "balanced": "Balanced fit",
    "value": "Best value",
    "reliability": "Reliability",
    "low_mileage": "Low mileage",
    "local": "Closest dealer",
    "ev_hybrid": "EV / hybrid",
}

BODY_STYLE_KEYWORDS = {
    "suv": (
        "suv",
        "crossover",
        "rav4",
        "cr-v",
        "crv",
        "cx-5",
        "cx-50",
        "rogue",
        "escape",
        "equinox",
        "forester",
        "outback",
        "highlander",
        "pilot",
        "telluride",
        "sorento",
        "tucson",
        "santa fe",
        "explorer",
        "grand cherokee",
    ),
    "sedan": (
        "sedan",
        "camry",
        "accord",
        "civic",
        "corolla",
        "altima",
        "sentra",
        "sonata",
        "elantra",
        "malibu",
        "passat",
        "a4",
        "3 series",
    ),
    "truck": ("truck", "f-150", "silverado", "ram ", "tacoma", "tundra", "ranger", "colorado"),
    "hatchback": ("hatchback", "hatch", "prius", "golf", "fit", "leaf", "bolt"),
    "minivan": ("minivan", "sienna", "odyssey", "pacifica", "carnival", "sedona"),
    "coupe": ("coupe", "mustang", "camaro", "challenger", "brz", "gr86"),
}


@dataclass(frozen=True)
class Preferences:
    query: str = ""
    min_price: int = 1000
    max_price: int = 30000
    metro: str = "any"
    fuel_types: tuple[str, ...] = ("any",)
    body_style: str = "any"
    min_year: int = 2016
    max_year: int = 2025
    min_mileage: int = 0
    max_mileage: int = 90000
    max_distance: int = 3000
    priorities: tuple[str, ...] = ("balanced",)
    must_haves: tuple[str, ...] = ()
    live_lookup: bool = True

    @property
    def max_budget(self) -> int:
        return self.max_price


def load_inventory() -> pd.DataFrame:
    df = load_database_inventory_or_empty()
    if df.empty:
        df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["sales_price", "year", "make", "model", "odometer_miles"]).copy()
    numeric_columns = [
        "sales_price",
        "average_market_price",
        "odometer_miles",
        "vehicle_age",
        "dealer_distance_miles",
        "year",
        "feature_count",
        "standard_feature_count",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["search_text"] = (
        df[["title", "make", "model", "trim", "seller_notes", "fuel_type"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.lower()
    )
    df["body_style"] = df["search_text"].map(infer_body_style)
    df["value_delta"] = df["average_market_price"] - df["sales_price"]
    df["value_delta_pct"] = df["value_delta"] / df["average_market_price"].replace(0, pd.NA)
    return df


def load_database_inventory_or_empty() -> pd.DataFrame:
    try:
        return load_inventory_from_database()
    except (OSError, ValueError):
        return pd.DataFrame()


def options() -> dict[str, Any]:
    df = load_inventory()
    metros = (
        df[["source_metro", "source_city", "source_state"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["source_state", "source_city"])
        .to_dict("records")
    )
    fuels = sorted(str(fuel) for fuel in df["fuel_type"].dropna().unique())
    return {
        "metros": metros,
        "fuels": fuels,
        "body_styles": ["suv", "sedan", "truck", "hatchback", "minivan", "coupe"],
        "features": FEATURE_KEYWORDS,
        "priorities": PRIORITY_OPTIONS,
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "min_price": 0,
        "max_price": int(math.ceil(df["sales_price"].quantile(0.98) / 1000) * 1000),
        "min_mileage": 0,
        "max_mileage": int(math.ceil(df["odometer_miles"].quantile(0.98) / 10000) * 10000),
    }


def infer_body_style(text: str) -> str:
    for style, keywords in BODY_STYLE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return style
    return "unknown"


def make_preferences(form: dict[str, Any]) -> Preferences:
    selected_features = form.getlist("must_haves") if hasattr(form, "getlist") else form.get("must_haves", [])
    if isinstance(selected_features, str):
        selected_features = [selected_features]
    must_haves = tuple(value for value in selected_features if value in FEATURE_KEYWORDS)

    selected_fuels = selected_values(form, "fuel_types")
    fuel_types = normalize_choices(selected_fuels, allowed_values=None, default=("any",), allow_any=True)

    selected_priorities = selected_values(form, "priorities")
    priorities = normalize_choices(
        selected_priorities,
        allowed_values=set(PRIORITY_OPTIONS),
        default=("balanced",),
        allow_any=False,
        limit=3,
    )

    min_price, max_price = ordered_pair(
        coerce_int(form.get("min_price"), 1000, 0, 300000),
        coerce_int(form.get("max_price", form.get("max_budget")), 30000, 1000, 300000),
    )
    min_year, max_year = ordered_pair(
        coerce_int(form.get("min_year"), 2016, 1990, 2026),
        coerce_int(form.get("max_year"), 2025, 1990, 2026),
    )
    min_mileage, max_mileage = ordered_pair(
        coerce_int(form.get("min_mileage"), 0, 0, 400000),
        coerce_int(form.get("max_mileage"), 90000, 1, 400000),
    )

    return Preferences(
        query=str(form.get("query") or "").strip(),
        min_price=min_price,
        max_price=max_price,
        metro=form.get("metro") or "any",
        fuel_types=fuel_types,
        body_style=form.get("body_style") or "any",
        min_year=min_year,
        max_year=max_year,
        min_mileage=min_mileage,
        max_mileage=max_mileage,
        max_distance=coerce_int(form.get("max_distance"), 3000, 1, 5000),
        priorities=priorities,
        must_haves=must_haves,
        live_lookup=form.get("live_lookup") == "on",
    )


def selected_values(form: dict[str, Any], key: str) -> list[str]:
    values = form.getlist(key) if hasattr(form, "getlist") else form.get(key, [])
    if isinstance(values, str):
        return [values]
    return list(values)


def normalize_choices(
    values: list[str],
    allowed_values: set[str] | None,
    default: tuple[str, ...],
    allow_any: bool,
    limit: int | None = None,
) -> tuple[str, ...]:
    clean_values = []
    for value in values:
        value = str(value).strip()
        if not value:
            continue
        if allow_any and value == "any":
            return ("any",)
        if allowed_values is None or value in allowed_values:
            clean_values.append(value)

    deduped = tuple(dict.fromkeys(clean_values))
    if not deduped:
        return default
    if limit:
        return deduped[:limit]
    return deduped


def coerce_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def ordered_pair(low_value: int, high_value: int) -> tuple[int, int]:
    return (low_value, high_value) if low_value <= high_value else (high_value, low_value)


def recommend(preferences: Preferences, limit: int = 12) -> dict[str, Any]:
    df = load_inventory()
    filtered, relaxations = filter_inventory(df, preferences)

    if filtered.empty:
        return {
            "recommendations": [],
            "market_summary": market_summary(df, preferences),
            "relaxations": ["No listings matched even after relaxing constraints."],
            "source_count": len(df),
        }

    scored = score_inventory(filtered, preferences)
    top = scored.sort_values(["score", "value_delta"], ascending=[False, False]).head(limit).copy()
    recommendations = [format_listing(row, preferences) for _, row in top.iterrows()]

    live_status = "skipped"
    if preferences.live_lookup:
        client = LiveDataClient()
        recommendations, live_status = client.enrich(recommendations)

    return {
        "recommendations": recommendations,
        "market_summary": market_summary(filtered, preferences),
        "relaxations": relaxations,
        "source_count": len(df),
        "live_status": live_status,
    }


def filter_inventory(df: pd.DataFrame, preferences: Preferences) -> tuple[pd.DataFrame, list[str]]:
    relaxations: list[str] = []

    def apply_filters(
        min_price: float,
        max_price: float,
        min_mileage: float,
        max_mileage: float,
        min_year: int,
        max_year: int,
        distance: float,
    ) -> pd.DataFrame:
        candidate = df[
            (df["sales_price"].between(min_price, max_price))
            & (df["odometer_miles"].between(min_mileage, max_mileage))
            & (df["year"].between(min_year, max_year))
            & (df["dealer_distance_miles"].fillna(distance) <= distance)
        ].copy()
        if preferences.metro != "any":
            candidate = candidate[candidate["source_metro"] == preferences.metro]
        if preferences.query:
            terms = [term for term in re.split(r"\s+", preferences.query.lower()) if term]
            for term in terms:
                candidate = candidate[candidate["search_text"].str.contains(re.escape(term), na=False)]
        if "any" not in preferences.fuel_types:
            fuel_types = {fuel.lower() for fuel in preferences.fuel_types}
            candidate = candidate[candidate["fuel_type"].fillna("").str.lower().isin(fuel_types)]
        if preferences.body_style != "any":
            candidate = candidate[candidate["body_style"] == preferences.body_style]
        return candidate

    attempts = [
        (
            preferences.min_price,
            preferences.max_price,
            preferences.min_mileage,
            preferences.max_mileage,
            preferences.min_year,
            preferences.max_year,
            preferences.max_distance,
            "",
        ),
        (
            preferences.min_price * 0.92,
            preferences.max_price * 1.08,
            max(0, preferences.min_mileage * 0.85),
            preferences.max_mileage * 1.15,
            preferences.min_year - 1,
            preferences.max_year + 1,
            preferences.max_distance,
            "Relaxed price, mileage, and year ranges slightly to find enough matches.",
        ),
        (
            preferences.min_price * 0.85,
            preferences.max_price * 1.15,
            max(0, preferences.min_mileage * 0.65),
            preferences.max_mileage * 1.35,
            preferences.min_year - 2,
            preferences.max_year + 2,
            max(preferences.max_distance, 3000),
            "Relaxed price, mileage, year, and distance ranges to avoid an empty result set.",
        ),
    ]

    for min_price, max_price, min_mileage, max_mileage, min_year, max_year, distance, note in attempts:
        candidate = apply_filters(min_price, max_price, min_mileage, max_mileage, min_year, max_year, distance)
        if len(candidate) >= 4 or note == attempts[-1][-1]:
            if note:
                relaxations.append(note)
            return candidate, relaxations

    return pd.DataFrame(), relaxations


def score_inventory(df: pd.DataFrame, preferences: Preferences) -> pd.DataFrame:
    scored = df.copy()

    price_span = max(preferences.max_price - preferences.min_price, 1)
    mileage_span = max(preferences.max_mileage - preferences.min_mileage, 1)
    year_span = max(preferences.max_year - preferences.min_year, 1)
    scored["price_score"] = 1 - ((scored["sales_price"] - preferences.min_price) / price_span).clip(0, 1)
    scored["mileage_score"] = 1 - ((scored["odometer_miles"] - preferences.min_mileage) / mileage_span).clip(0, 1)
    scored["age_score"] = ((scored["year"] - preferences.min_year) / year_span).clip(0, 1)
    scored["distance_score"] = 1 - (
        scored["dealer_distance_miles"].fillna(preferences.max_distance) / max(preferences.max_distance, 1)
    ).clip(upper=1)
    scored["value_score"] = scored["price_tier"].map(PRICE_TIER_SCORE).fillna(0.45)
    scored["market_delta_score"] = scored["value_delta_pct"].fillna(0).clip(-0.2, 0.2).add(0.2).div(0.4)
    scored["reliability_score"] = scored["make"].map(RELIABILITY_BY_MAKE).fillna(0.58)
    scored["feature_score"] = scored["search_text"].map(lambda text: feature_match_score(text, preferences.must_haves))
    scored["fuel_preference_score"] = scored["fuel_type"].map(lambda fuel: fuel_score(str(fuel), preferences.priorities))

    weights = weights_for_priorities(preferences.priorities)
    scored["score"] = sum(scored[column] * weight for column, weight in weights.items())
    if preferences.must_haves:
        scored["score"] += scored["feature_score"] * 0.07
    return scored


def weights_for_priorities(priorities: tuple[str, ...]) -> dict[str, float]:
    selected = priorities[:3] or ("balanced",)
    weight_maps = [weights_for_priority(priority) for priority in selected]
    averaged = {
        key: sum(weight_map[key] for weight_map in weight_maps) / len(weight_maps)
        for key in weight_maps[0]
    }
    total = sum(averaged.values())
    return {key: value / total for key, value in averaged.items()}


def weights_for_priority(priority: str) -> dict[str, float]:
    base = {
        "price_score": 0.17,
        "value_score": 0.17,
        "market_delta_score": 0.12,
        "mileage_score": 0.14,
        "age_score": 0.13,
        "reliability_score": 0.12,
        "distance_score": 0.08,
        "fuel_preference_score": 0.07,
    }
    if priority == "value":
        base.update({"price_score": 0.22, "value_score": 0.22, "market_delta_score": 0.16, "age_score": 0.08})
    elif priority == "reliability":
        base.update({"reliability_score": 0.24, "mileage_score": 0.18, "age_score": 0.15, "fuel_preference_score": 0.03})
    elif priority == "low_mileage":
        base.update({"mileage_score": 0.27, "age_score": 0.17, "price_score": 0.12, "value_score": 0.12})
    elif priority == "local":
        base.update({"distance_score": 0.22, "price_score": 0.16, "value_score": 0.16})
    elif priority == "ev_hybrid":
        base.update({"fuel_preference_score": 0.22, "price_score": 0.13, "value_score": 0.13})
    total = sum(base.values())
    return {key: value / total for key, value in base.items()}


def feature_match_score(text: str, must_haves: tuple[str, ...]) -> float:
    if not must_haves:
        return 0.5
    matches = 0
    for feature in must_haves:
        if any(keyword in text for keyword in FEATURE_KEYWORDS[feature]):
            matches += 1
    return matches / len(must_haves)


def fuel_score(fuel_type: str, priorities: tuple[str, ...]) -> float:
    fuel = fuel_type.lower()
    if "ev_hybrid" in priorities:
        if "electric" in fuel:
            return 1.0
        if "hybrid" in fuel:
            return 0.88
        if "diesel" in fuel:
            return 0.45
        return 0.25
    if "hybrid" in fuel:
        return 0.72
    if "electric" in fuel:
        return 0.68
    if "diesel" in fuel:
        return 0.5
    return 0.55


def format_listing(row: pd.Series, preferences: Preferences) -> dict[str, Any]:
    feature_hits = matched_features(row["search_text"], preferences.must_haves)
    estimated_tco = estimate_five_year_cost(row)
    reasons = recommendation_reasons(row, preferences, feature_hits)
    return {
        "title": row.get("title") or f"{int(row['year'])} {row['make']} {row['model']}",
        "year": int(row["year"]),
        "make": row["make"],
        "model": row["model"],
        "trim": row.get("trim") if pd.notna(row.get("trim")) else "",
        "price": int(row["sales_price"]),
        "average_market_price": safe_int(row.get("average_market_price")),
        "value_delta": safe_int(row.get("value_delta")),
        "odometer_miles": int(row["odometer_miles"]),
        "fuel_type": row.get("fuel_type") if pd.notna(row.get("fuel_type")) else "Unknown",
        "body_style": row.get("body_style", "unknown"),
        "dealer": row.get("dealer_info") if pd.notna(row.get("dealer_info")) else "Dealer not listed",
        "dealer_city": row.get("dealer_city") if pd.notna(row.get("dealer_city")) else "",
        "dealer_state": row.get("dealer_state") if pd.notna(row.get("dealer_state")) else "",
        "distance": safe_int(row.get("dealer_distance_miles")),
        "price_tier": row.get("price_tier") if pd.notna(row.get("price_tier")) else "unknown",
        "url": row.get("url") if pd.notna(row.get("url")) else "",
        "vin": row.get("vin") if pd.notna(row.get("vin")) else "",
        "image_url": best_image_url(row),
        "deal_label": deal_label(row),
        "monthly_payment": estimate_monthly_payment(row),
        "score": round(float(row["score"]) * 100, 1),
        "feature_hits": feature_hits,
        "estimated_tco": estimated_tco,
        "reasons": reasons,
        "live": {},
    }


def matched_features(text: str, must_haves: tuple[str, ...]) -> list[str]:
    labels = {
        "apple_carplay": "Apple CarPlay / Android Auto",
        "awd": "AWD / 4WD",
        "backup_camera": "Backup camera",
        "blind_spot": "Blind spot warning",
        "heated_seats": "Heated seats",
        "lane_assist": "Lane assist",
        "moonroof": "Sunroof / moonroof",
        "third_row": "Third row",
    }
    return [
        labels[feature]
        for feature in must_haves
        if any(keyword in text for keyword in FEATURE_KEYWORDS[feature])
    ]


def best_image_url(row: pd.Series) -> str:
    for column in ("listing_image_url", "image_url", "photo_url"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return ""


def deal_label(row: pd.Series) -> str:
    tier = row.get("price_tier")
    if tier == "below_market":
        return "Great deal"
    if tier == "near_market":
        return "Fair price"
    if tier == "above_market":
        return "High price"
    return "Recommended"


def estimate_monthly_payment(row: pd.Series) -> int:
    principal = float(row["sales_price"]) * 0.92
    annual_rate = 0.075
    monthly_rate = annual_rate / 12
    months = 60
    payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)
    return int(round(payment))


def recommendation_reasons(row: pd.Series, preferences: Preferences, feature_hits: list[str]) -> list[str]:
    reasons = []
    delta = safe_int(row.get("value_delta"))
    if delta and delta > 500:
        reasons.append(f"${delta:,} below the comparable market average")
    elif row.get("price_tier") == "below_market":
        reasons.append("Listed below market")

    if row["sales_price"] <= preferences.min_price + (preferences.max_price - preferences.min_price) * 0.35:
        reasons.append("Prices well within your selected range")
    if row["odometer_miles"] <= preferences.min_mileage + (preferences.max_mileage - preferences.min_mileage) * 0.45:
        reasons.append("Mileage lands in the lower half of your range")
    if row["year"] >= preferences.min_year + (preferences.max_year - preferences.min_year) * 0.65:
        reasons.append("Year is toward the newer end of your range")
    if row.get("make") in RELIABILITY_BY_MAKE and RELIABILITY_BY_MAKE[row.get("make")] >= 0.82:
        reasons.append(f"{row.get('make')} has a strong reliability signal in this heuristic")
    if feature_hits:
        reasons.append("Matches " + ", ".join(feature_hits[:2]))
    if not reasons:
        reasons.append("Best overall fit after balancing your filters")
    return reasons[:4]


def estimate_five_year_cost(row: pd.Series) -> int:
    price = float(row["sales_price"])
    miles = float(row["odometer_miles"])
    age = max(1, 2026 - int(row["year"]))
    fuel = str(row.get("fuel_type", "")).lower()
    annual_miles = 12000
    fuel_price = 3.65
    if "electric" in fuel:
        fuel_cost = 0.04 * annual_miles * 5
    elif "hybrid" in fuel:
        fuel_cost = annual_miles / 42 * fuel_price * 5
    elif "diesel" in fuel:
        fuel_cost = annual_miles / 28 * 4.1 * 5
    else:
        fuel_cost = annual_miles / 26 * fuel_price * 5
    maintenance = 700 * 5 + max(0, age - 4) * 260 + max(0, miles - 70000) * 0.018
    return int(round(price + fuel_cost + maintenance, -2))


def market_summary(df: pd.DataFrame, preferences: Preferences) -> dict[str, Any]:
    if df.empty:
        return {}
    return {
        "count": int(len(df)),
        "median_price": int(df["sales_price"].median()),
        "median_mileage": int(df["odometer_miles"].median()),
        "median_year": int(df["year"].median()),
        "below_market_count": int((df["price_tier"] == "below_market").sum()),
        "price_min": preferences.min_price,
        "price_max": preferences.max_price,
    }


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value) or math.isinf(float(value)):
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


class LiveDataClient:
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_seconds: int = 24 * 60 * 60) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.disabled = False
        self.cache_dir.mkdir(exist_ok=True)

    def enrich(self, listings: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        attempted = 0
        successes = 0
        for listing in listings[:5]:
            live: dict[str, Any] = {}
            safety = self.safety_rating(listing["year"], listing["make"], listing["model"])
            attempted += 1
            if safety:
                live["safety"] = safety
                successes += 1

            if listing.get("vin"):
                decoded = self.decode_vin(listing["vin"])
                attempted += 1
                if decoded:
                    live["vin"] = decoded
                    successes += 1

            listing["live"] = live

        if attempted == 0:
            return listings, "skipped"
        if successes == 0:
            return listings, "unavailable"
        return listings, "fresh"

    def safety_rating(self, year: int, make: str, model: str) -> dict[str, Any] | None:
        slug = f"safety_{year}_{make}_{model}".lower()
        url = f"https://api.nhtsa.gov/SafetyRatings/modelyear/{year}/make/{make}/model/{model}"
        data = self.get_json(url, slug)
        results = data.get("Results", []) if data else []
        if not results:
            return None

        vehicle_id = results[0].get("VehicleId")
        if not vehicle_id:
            return {"vehicle": results[0].get("VehicleDescription", ""), "overall": "Not rated"}

        detail = self.get_json(
            f"https://api.nhtsa.gov/SafetyRatings/VehicleId/{vehicle_id}",
            f"safety_detail_{vehicle_id}",
        )
        detail_results = detail.get("Results", []) if detail else []
        if not detail_results:
            return None
        rating = detail_results[0]
        return {
            "vehicle": rating.get("VehicleDescription", results[0].get("VehicleDescription", "")),
            "overall": clean_rating(rating.get("OverallRating")),
            "front": clean_rating(rating.get("OverallFrontCrashRating")),
            "side": clean_rating(rating.get("OverallSideCrashRating")),
            "rollover": clean_rating(rating.get("RolloverRating")),
        }

    def decode_vin(self, vin: str) -> dict[str, Any] | None:
        if not vin or len(vin) < 11:
            return None
        data = self.get_json(
            f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json",
            f"vin_{vin}",
        )
        results = data.get("Results", []) if data else []
        if not results:
            return None
        decoded = results[0]
        return {
            "drive_type": blank_to_none(decoded.get("DriveType")),
            "engine": blank_to_none(decoded.get("EngineConfiguration")) or blank_to_none(decoded.get("EngineCylinders")),
            "plant": blank_to_none(decoded.get("PlantCountry")),
            "body_class": blank_to_none(decoded.get("BodyClass")),
        }

    def get_json(self, url: str, cache_key: str) -> dict[str, Any] | None:
        path = self.cache_path(cache_key)
        if path.exists() and time.time() - path.stat().st_mtime < self.ttl_seconds:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
        if self.disabled:
            return None
        try:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
        except requests.RequestException:
            self.disabled = True
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def cache_path(self, key: str) -> Path:
        clean_key = re.sub(r"[^a-z0-9_.-]+", "_", key.lower()).strip("_")
        return self.cache_dir / f"{clean_key}.json"


def clean_rating(value: Any) -> str:
    if value in (None, "", "Not Rated"):
        return "Not rated"
    return str(value)


def blank_to_none(value: Any) -> str | None:
    if value in (None, "", "Not Applicable"):
        return None
    return str(value)
