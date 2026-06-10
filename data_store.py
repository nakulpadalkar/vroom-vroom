from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd


DATABASE_ENV_NAMES = ("TRUECAR_DATABASE_URL", "DATABASE_URL")
DEFAULT_DATABASE_URL = "sqlite:///data/vroom_vroom.sqlite3"
INVENTORY_TABLE = "truecar_inventory"
SCRAPE_RUNS_TABLE = "truecar_scrape_runs"


def database_url() -> str:
    for name in DATABASE_ENV_NAMES:
        value = os.getenv(name)
        if value:
            return value
    return DEFAULT_DATABASE_URL


def database_path(url: str | None = None) -> Path:
    url = url or database_url()
    parsed = urlparse(url)
    if parsed.scheme in ("", "sqlite"):
        if parsed.scheme == "":
            return Path(url)
        if parsed.netloc and parsed.netloc != ".":
            raise ValueError("Only local SQLite database URLs are supported by the built-in data store.")
        if parsed.path in ("", "/"):
            raise ValueError("SQLite database URL must include a file path.")
        if parsed.path == "/:memory:":
            return Path(":memory:")
        path = unquote(parsed.path)
        if parsed.netloc == ".":
            path = f".{path}"
        if os.name == "nt" and resemblances_windows_absolute_path(path):
            path = path.lstrip("/")
        elif os.name == "nt" and path.startswith("/"):
            path = path.lstrip("/")
        return Path(path)
    raise ValueError("Only sqlite:/// database URLs are supported without adding another database driver.")


def resemblances_windows_absolute_path(path: str) -> bool:
    return len(path) >= 4 and path[0] == "/" and path[2] == ":" and path[3] in ("\\", "/")


def connect(url: str | None = None) -> sqlite3.Connection:
    path = database_path(url)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_database(url: str | None = None) -> None:
    with connect(url) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCRAPE_RUNS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_metro TEXT,
                city TEXT,
                state TEXT,
                row_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                message TEXT,
                scraped_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {INVENTORY_TABLE} (
                url TEXT PRIMARY KEY,
                vin TEXT,
                title TEXT,
                year REAL,
                make TEXT,
                model TEXT,
                trim TEXT,
                sales_price REAL,
                average_market_price REAL,
                price_tier TEXT,
                odometer_miles REAL,
                vehicle_age REAL,
                fuel_type TEXT,
                exterior TEXT,
                interior TEXT,
                dealer_info TEXT,
                dealer_city TEXT,
                dealer_state TEXT,
                dealer_distance_miles REAL,
                source_city TEXT,
                source_state TEXT,
                source_metro TEXT,
                feature_count REAL,
                standard_feature_count REAL,
                popular_features TEXT,
                standard_features TEXT,
                options_and_packages TEXT,
                key_highlights TEXT,
                overview_json TEXT,
                feature_groups_json TEXT,
                specs_json TEXT,
                history_json TEXT,
                vehicle_history_summary TEXT,
                vehicle_history_report_url TEXT,
                history_condition_data_as_of TEXT,
                seller_notes TEXT,
                price_description TEXT,
                listed_since TEXT,
                stock_number TEXT,
                listing_image_url TEXT,
                raw_json TEXT,
                scraped_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        for column_name in [
            "key_highlights",
            "overview_json",
            "feature_groups_json",
            "specs_json",
            "history_json",
            "vehicle_history_summary",
            "vehicle_history_report_url",
            "history_condition_data_as_of",
            "listing_image_url",
        ]:
            try:
                connection.execute(f"ALTER TABLE {INVENTORY_TABLE} ADD COLUMN {column_name} TEXT")
            except sqlite3.OperationalError:
                pass


def save_inventory(
    df: pd.DataFrame,
    city: str,
    state: str,
    status: str = "success",
    message: str | None = None,
    url: str | None = None,
) -> int:
    init_database(url)
    if df.empty:
        record_scrape_run(city, state, 0, status, message, url)
        return 0

    rows = normalize_inventory_rows(df)
    with connect(url) as connection:
        for row in rows:
            connection.execute(
                f"""
                INSERT INTO {INVENTORY_TABLE} (
                    url, vin, title, year, make, model, trim, sales_price, average_market_price,
                    price_tier, odometer_miles, vehicle_age, fuel_type, exterior, interior,
                    dealer_info, dealer_city, dealer_state, dealer_distance_miles, source_city,
                    source_state, source_metro, feature_count, standard_feature_count,
                    popular_features, standard_features, options_and_packages, key_highlights,
                    overview_json, feature_groups_json, specs_json, history_json,
                    vehicle_history_summary, vehicle_history_report_url, history_condition_data_as_of,
                    seller_notes, price_description, listed_since, stock_number, listing_image_url, raw_json, scraped_at
                ) VALUES (
                    :url, :vin, :title, :year, :make, :model, :trim, :sales_price, :average_market_price,
                    :price_tier, :odometer_miles, :vehicle_age, :fuel_type, :exterior, :interior,
                    :dealer_info, :dealer_city, :dealer_state, :dealer_distance_miles, :source_city,
                    :source_state, :source_metro, :feature_count, :standard_feature_count,
                    :popular_features, :standard_features, :options_and_packages, :key_highlights,
                    :overview_json, :feature_groups_json, :specs_json, :history_json,
                    :vehicle_history_summary, :vehicle_history_report_url, :history_condition_data_as_of,
                    :seller_notes, :price_description, :listed_since, :stock_number, :listing_image_url, :raw_json, CURRENT_TIMESTAMP
                )
                ON CONFLICT(url) DO UPDATE SET
                    vin=excluded.vin,
                    title=excluded.title,
                    year=excluded.year,
                    make=excluded.make,
                    model=excluded.model,
                    trim=excluded.trim,
                    sales_price=excluded.sales_price,
                    average_market_price=excluded.average_market_price,
                    price_tier=excluded.price_tier,
                    odometer_miles=excluded.odometer_miles,
                    vehicle_age=excluded.vehicle_age,
                    fuel_type=excluded.fuel_type,
                    exterior=excluded.exterior,
                    interior=excluded.interior,
                    dealer_info=excluded.dealer_info,
                    dealer_city=excluded.dealer_city,
                    dealer_state=excluded.dealer_state,
                    dealer_distance_miles=excluded.dealer_distance_miles,
                    source_city=excluded.source_city,
                    source_state=excluded.source_state,
                    source_metro=excluded.source_metro,
                    feature_count=excluded.feature_count,
                    standard_feature_count=excluded.standard_feature_count,
                    popular_features=excluded.popular_features,
                    standard_features=excluded.standard_features,
                    options_and_packages=excluded.options_and_packages,
                    key_highlights=excluded.key_highlights,
                    overview_json=excluded.overview_json,
                    feature_groups_json=excluded.feature_groups_json,
                    specs_json=excluded.specs_json,
                    history_json=excluded.history_json,
                    vehicle_history_summary=excluded.vehicle_history_summary,
                    vehicle_history_report_url=excluded.vehicle_history_report_url,
                    history_condition_data_as_of=excluded.history_condition_data_as_of,
                    seller_notes=excluded.seller_notes,
                    price_description=excluded.price_description,
                    listed_since=excluded.listed_since,
                    stock_number=excluded.stock_number,
                    listing_image_url=excluded.listing_image_url,
                    raw_json=excluded.raw_json,
                    scraped_at=CURRENT_TIMESTAMP
                """,
                row,
            )
        connection.execute(
            f"""
            INSERT INTO {SCRAPE_RUNS_TABLE} (source_metro, city, state, row_count, status, message)
            VALUES (:source_metro, :city, :state, :row_count, :status, :message)
            """,
            {
                "source_metro": source_metro_key(city, state),
                "city": city,
                "state": state.upper(),
                "row_count": len(rows),
                "status": status,
                "message": message,
            },
        )
    return len(rows)


def normalize_inventory_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "url",
        "vin",
        "title",
        "year",
        "make",
        "model",
        "trim",
        "sales_price",
        "average_market_price",
        "price_tier",
        "odometer_miles",
        "vehicle_age",
        "fuel_type",
        "exterior",
        "interior",
        "dealer_info",
        "dealer_city",
        "dealer_state",
        "dealer_distance_miles",
        "source_city",
        "source_state",
        "source_metro",
        "feature_count",
        "standard_feature_count",
        "popular_features",
        "standard_features",
        "options_and_packages",
        "key_highlights",
        "overview_json",
        "feature_groups_json",
        "specs_json",
        "history_json",
        "vehicle_history_summary",
        "vehicle_history_report_url",
        "history_condition_data_as_of",
        "seller_notes",
        "price_description",
        "listed_since",
        "stock_number",
        "listing_image_url",
    ]
    clean = df.copy()
    clean = clean.dropna(subset=["url"])
    rows: list[dict[str, Any]] = []
    for _, record in clean.iterrows():
        row = {column: value_or_none(record.get(column)) for column in columns}
        row["raw_json"] = record.dropna().to_json()
        rows.append(row)
    return rows


def value_or_none(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def record_scrape_run(
    city: str,
    state: str,
    row_count: int,
    status: str,
    message: str | None = None,
    url: str | None = None,
) -> None:
    init_database(url)
    with connect(url) as connection:
        connection.execute(
            f"""
            INSERT INTO {SCRAPE_RUNS_TABLE} (source_metro, city, state, row_count, status, message)
            VALUES (:source_metro, :city, :state, :row_count, :status, :message)
            """,
            {
                "source_metro": source_metro_key(city, state),
                "city": city,
                "state": state.upper(),
                "row_count": row_count,
                "status": status,
                "message": message,
            },
        )


def load_inventory_from_database(url: str | None = None) -> pd.DataFrame:
    init_database(url)
    with connect(url) as connection:
        return pd.read_sql_query(
            f"""
            SELECT *
            FROM {INVENTORY_TABLE}
            WHERE sales_price IS NOT NULL
              AND year IS NOT NULL
              AND make IS NOT NULL
              AND model IS NOT NULL
              AND odometer_miles IS NOT NULL
            ORDER BY scraped_at DESC
            """,
            connection,
        )


def latest_scrape_runs(limit: int = 5, url: str | None = None) -> list[dict[str, Any]]:
    init_database(url)
    with connect(url) as connection:
        rows = connection.execute(
            f"""
            SELECT source_metro, city, state, row_count, status, message, scraped_at
            FROM {SCRAPE_RUNS_TABLE}
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    keys = ["source_metro", "city", "state", "row_count", "status", "message", "scraped_at"]
    return [dict(zip(keys, row, strict=False)) for row in rows]


def source_metro_key(city: str, state: str) -> str:
    return f"{city.strip().lower().replace(' ', '-')}_{state.strip().lower()}"
