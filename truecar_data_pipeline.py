from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://www.truecar.com"
DATA_DIR = Path("data")


DEFAULT_CITIES = [
    ("Boston", "MA"),
    ("New York", "NY"),
    ("Atlanta", "GA"),
    ("Miami", "FL"),
    ("Chicago", "IL"),
    ("Detroit", "MI"),
    ("Dallas", "TX"),
    ("Houston", "TX"),
    ("Austin", "TX"),
    ("Denver", "CO"),
    ("Phoenix", "AZ"),
    ("Seattle", "WA"),
    ("San Francisco", "CA"),
    ("Los Angeles", "CA"),
    ("Minneapolis", "MN"),
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def slug_city(city: str) -> str:
    return city.strip().lower().replace(" ", "-")


def city_key(city: str, state: str) -> str:
    return f"{slug_city(city)}_{state.strip().lower()}"


def truecar_listing_url(
    city: str,
    state: str,
    page: int,
    page_size: int = 100,
    search_radius: int = 75,
    exclude_expanded_delivery: bool = True,
) -> str:
    expanded_delivery = "true" if exclude_expanded_delivery else "false"
    return (
        f"{BASE_URL}/used-cars-for-sale/listings/"
        f"location-{slug_city(city)}-{state.strip().lower()}/"
        f"?stock_type=used&page_size={page_size}&searchRadius={search_radius}"
        f"&excludeExpandedDelivery={expanded_delivery}&page={page}"
    )


def get_soup(url: str, delay_seconds: float = 0.5) -> BeautifulSoup | None:
    time.sleep(delay_seconds)
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Request failed for {url}: {exc}")
        return None
    return BeautifulSoup(response.text, "html.parser")


def first_text(parent: BeautifulSoup, selector: str) -> str | None:
    element = parent.select_one(selector)
    return element.get_text(separator=" ", strip=True) if element else None


def extract_listing_cards(
    city: str,
    state: str,
    max_pages: int = 20,
    page_size: int = 100,
    search_radius: int = 75,
    exclude_expanded_delivery: bool = True,
    delay_seconds: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict[str, str | int | None]] = []
    key = city_key(city, state)

    for page in tqdm(range(1, max_pages + 1), desc=f"{key} listing pages"):
        url = truecar_listing_url(
            city,
            state,
            page,
            page_size=page_size,
            search_radius=search_radius,
            exclude_expanded_delivery=exclude_expanded_delivery,
        )
        soup = get_soup(url, delay_seconds=delay_seconds)
        if soup is None:
            continue

        link_tags = soup.select('a[data-test="cardLinkCover"]')
        if not link_tags:
            print(f"No listings found on page {page} for {city}, {state}. Stopping this city.")
            break

        for tag in link_tags:
            relative_url = tag.get("href")
            if not relative_url:
                continue

            card = tag.find_parent("div", class_="card")
            row = {
                "source_city": city,
                "source_state": state.upper(),
                "source_metro": key,
                "source_page": page,
                "url": urljoin(BASE_URL, relative_url),
            }

            if card:
                row["title"] = first_text(card, '[data-test="vehicleCardInfo"]')
                row["mileage_listed"] = first_text(card, '[data-test="vehicleMileage"]')
                row["list_price_displayed"] = first_text(card, '[data-test="vehicleCardPricingPrice"]')
                row["dealer_info"] = first_text(card, '[data-test="vehicleCardFooter"]')

            rows.append(row)

    df = pd.DataFrame(rows)
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df


def extract_detail_fields(entry: dict, delay_seconds: float = 0.5) -> dict:
    url = entry["url"]
    soup = get_soup(url, delay_seconds=delay_seconds)
    car_data = dict(entry)

    if soup is None:
        car_data["scrape_error"] = "request_failed"
        return car_data

    try:
        detail_container = soup.select_one("div.row.pt-3")
        if detail_container:
            for detail in detail_container.select("div.flex.items-center"):
                text = detail.get_text(separator=" ", strip=True)
                if not text:
                    continue
                if ":" in text:
                    key, value = text.split(":", 1)
                    clean_key = key.strip().lower().replace(" ", "_")
                    car_data[clean_key] = value.strip()
                elif "VIN" in text:
                    car_data["vin"] = text.split("VIN:", 1)[-1].strip()
                elif "Stock Number" in text:
                    car_data["stock_number"] = text.split("Stock Number:", 1)[-1].strip()
                elif "Listed" in text:
                    car_data["listed_since"] = text.strip()

        for heading, output_name in [
            ("Options & packages", "options_and_packages"),
            ("Popular features", "popular_features"),
            ("Standard features", "standard_features"),
        ]:
            header = soup.find("h2", string=heading)
            if header:
                feature_container = header.find_next("div")
                if feature_container:
                    features = [
                        item.get_text(separator=" ", strip=True)
                        for item in feature_container.find_all("div", class_="flex items-center")
                    ]
                    car_data[output_name] = "; ".join(item for item in features if item)

        price_section = soup.find("div", {"id": "usedPriceGraph"})
        if price_section:
            for item in price_section.select('div[data-test="usedListingPriceGraphLineItem"]'):
                label = item.get("data-test-item")
                text = item.get_text(separator="|", strip=True)
                if label and "|" in text:
                    _, value = text.split("|", 1)
                    car_data[label.lower().replace(" ", "_")] = value.strip()

            for bar in price_section.select('div[data-test="priceRangeIconAndRange"]'):
                quality = bar.get("data-test-item")
                range_tag = bar.find("p")
                if quality and range_tag:
                    car_data[f"price_range_{quality.lower()}"] = range_tag.get_text(strip=True)

            description = price_section.find("div", {"data-test": "usedListingPriceGraphDescription"})
            if description:
                car_data["price_description"] = description.get_text(separator=" ", strip=True)

        seller_notes_header = soup.find("h2", string="Seller Notes")
        if seller_notes_header:
            seller_div = seller_notes_header.find_next("div", class_="see-more")
            if seller_div:
                car_data["seller_notes"] = seller_div.get_text(separator=" ", strip=True)

    except Exception as exc:  # noqa: BLE001 - notebooks should retain partial rows.
        car_data["scrape_error"] = str(exc)

    return car_data


def scrape_city(
    city: str,
    state: str,
    max_pages: int = 20,
    page_size: int = 100,
    search_radius: int = 75,
    exclude_expanded_delivery: bool = True,
    output_dir: Path | str = DATA_DIR,
    delay_seconds: float = 0.5,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = city_key(city, state)

    links = extract_listing_cards(
        city=city,
        state=state,
        max_pages=max_pages,
        page_size=page_size,
        search_radius=search_radius,
        exclude_expanded_delivery=exclude_expanded_delivery,
        delay_seconds=delay_seconds,
    )
    links.to_csv(output_dir / f"truecar_links_{key}.csv", index=False)
    (output_dir / f"truecar_links_{key}.json").write_text(
        links.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    details = [
        extract_detail_fields(row, delay_seconds=delay_seconds)
        for row in tqdm(links.to_dict(orient="records"), desc=f"{key} detail pages")
    ]
    details_df = pd.DataFrame(details)
    details_df.to_csv(output_dir / f"truecar_details_{key}.csv", index=False)
    (output_dir / f"truecar_details_{key}.json").write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )
    return details_df


def scrape_cities(
    cities: Iterable[tuple[str, str]],
    max_pages: int = 20,
    page_size: int = 100,
    search_radius: int = 75,
    exclude_expanded_delivery: bool = True,
    output_dir: Path | str = DATA_DIR,
    delay_seconds: float = 0.5,
) -> dict[str, pd.DataFrame]:
    results = {}
    for city, state in cities:
        key = city_key(city, state)
        results[key] = scrape_city(
            city=city,
            state=state,
            max_pages=max_pages,
            page_size=page_size,
            search_radius=search_radius,
            exclude_expanded_delivery=exclude_expanded_delivery,
            output_dir=output_dir,
            delay_seconds=delay_seconds,
        )
    return results


def parse_money(value) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"[\d,]+(?:\.\d+)?", str(value))
    return float(match.group(0).replace(",", "")) if match else None


def parse_integer(value) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"[\d,]+", str(value))
    return int(match.group(0).replace(",", "")) if match else None


def parse_dealer_location(value) -> pd.Series:
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"([A-Za-z .'-]+),\s*([A-Z]{2})\s*\((\d+)\s+miles?\s+away\)", text)
    if not match:
        return pd.Series([None, None, None])
    return pd.Series([match.group(1).strip(), match.group(2), int(match.group(3))])


def split_title(value) -> pd.Series:
    if pd.isna(value):
        return pd.Series([None, None, None, None])
    parts = re.sub(r"^Used\s+", "", str(value).strip()).split()
    year = parts[0] if len(parts) > 0 and re.fullmatch(r"\d{4}", parts[0]) else None
    make = parts[1] if len(parts) > 1 else None
    model = parts[2] if len(parts) > 2 else None
    trim = " ".join(parts[3:]) if len(parts) > 3 else None
    return pd.Series([year, make, model, trim])


def clean_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"\.(?=[A-Za-z])", ". ", text)
    return text or None


def count_items(value) -> int:
    if pd.isna(value) or not str(value).strip():
        return 0
    return len([item for item in re.split(r"\s*;\s*|\s*,\s*", str(value)) if item.strip()])


def price_tier(price: float | None, average_price: float | None) -> str | None:
    if price is None or average_price in (None, 0) or pd.isna(price) or pd.isna(average_price):
        return None
    ratio = price / average_price
    if ratio <= 0.95:
        return "below_market"
    if ratio >= 1.05:
        return "above_market"
    return "near_market"


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "rows",
                "unique_urls",
                "unique_vins",
                "missing_sales_price",
                "missing_odometer_miles",
                "missing_year",
                "missing_make",
                "missing_model",
            ],
            "value": [
                len(df),
                df["url"].nunique() if "url" in df else None,
                df["vin"].nunique() if "vin" in df else None,
                int(df["sales_price"].isna().sum()) if "sales_price" in df else None,
                int(df["odometer_miles"].isna().sum()) if "odometer_miles" in df else None,
                int(df["year"].isna().sum()) if "year" in df else None,
                int(df["make"].isna().sum()) if "make" in df else None,
                int(df["model"].isna().sum()) if "model" in df else None,
            ],
        }
    )


def load_city_detail_files(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("truecar_details_*_*.csv"))
    frames = []
    for file in files:
        source_metro = file.stem.replace("truecar_details_", "")
        df = pd.read_csv(file, dtype=str)
        if "source_metro" not in df:
            df["source_metro"] = source_metro
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def clean_truecar_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()

    if "source_metro" in df:
        parsed_source = df["source_metro"].str.extract(r"(?P<source_city_slug>.+)_(?P<source_state_slug>[a-z]{2})$")
        if "source_city" not in df:
            df["source_city"] = parsed_source["source_city_slug"].str.replace("-", " ", regex=False).str.title()
        if "source_state" not in df:
            df["source_state"] = parsed_source["source_state_slug"].str.upper()

    for column in ["title", "seller_notes", "price_description"]:
        if column in df:
            df[column] = df[column].apply(clean_text)

    if "title" in df:
        title_parts = df["title"].apply(split_title)
        title_parts.columns = ["year_from_title", "make_from_title", "model_from_title", "trim_from_title"]
        df = pd.concat([df, title_parts], axis=1)

    for column, fallback in [
        ("year", "year_from_title"),
        ("make", "make_from_title"),
        ("model", "model_from_title"),
        ("trim", "trim_from_title"),
    ]:
        if column not in df:
            df[column] = None
        if fallback in df:
            df[column] = df[column].fillna(df[fallback])

    price_source = df["list_price"] if "list_price" in df else pd.Series(index=df.index, dtype=object)
    if "list_price_displayed" in df:
        price_source = price_source.fillna(df["list_price_displayed"])
    df["sales_price"] = price_source.apply(parse_money)

    average_source = df["average_list_price"] if "average_list_price" in df else pd.Series(index=df.index, dtype=object)
    df["average_market_price"] = average_source.apply(parse_money)

    mileage_source = df["mileage_listed"] if "mileage_listed" in df else pd.Series(index=df.index, dtype=object)
    if "mileage" in df:
        detail_mileage = df["mileage"].where(~df["mileage"].fillna("").str.contains("miles away", case=False))
        mileage_source = mileage_source.fillna(detail_mileage)
    df["odometer_miles"] = mileage_source.apply(parse_integer)

    location_source = pd.Series(index=df.index, dtype=object)
    if "mileage" in df:
        location_source = location_source.fillna(df["mileage"].where(df["mileage"].fillna("").str.contains("miles away", case=False)))
    if "dealer_info" in df:
        location_source = location_source.fillna(df["dealer_info"])

    location_parts = location_source.apply(parse_dealer_location)
    location_parts.columns = ["dealer_city", "dealer_state", "dealer_distance_miles"]
    df = pd.concat([df, location_parts], axis=1)

    df["vehicle_age"] = pd.to_numeric(df["year"], errors="coerce").apply(
        lambda year: 2026 - year if pd.notna(year) else None
    )
    df["feature_count"] = df.get("popular_features", pd.Series(index=df.index, dtype=object)).apply(count_items)
    df["standard_feature_count"] = df.get("standard_features", pd.Series(index=df.index, dtype=object)).apply(count_items)
    df["price_tier"] = [
        price_tier(price, avg)
        for price, avg in zip(df["sales_price"], df["average_market_price"], strict=False)
    ]

    dedupe_key = df["vin"].where(df.get("vin", pd.Series(index=df.index)).notna(), df["url"])
    df["_dedupe_key"] = dedupe_key
    df["_completeness"] = df.notna().sum(axis=1)
    df = (
        df.sort_values(["_dedupe_key", "_completeness"], ascending=[True, False])
        .drop_duplicates(subset=["_dedupe_key"], keep="first")
        .drop_duplicates(subset=["url"], keep="first")
        .drop(columns=["_dedupe_key", "_completeness"])
        .reset_index(drop=True)
    )

    preferred_columns = [
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
        "seller_notes",
        "price_description",
        "listed_since",
        "stock_number",
    ]
    columns = [column for column in preferred_columns if column in df.columns]
    columns += [column for column in df.columns if column not in columns]
    return df[columns]


def build_clean_dataset(
    data_dir: Path | str = DATA_DIR,
    output_csv: Path | str = DATA_DIR / "truecar_clean_combined.csv",
    report_csv: Path | str = DATA_DIR / "truecar_quality_report.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_city_detail_files(data_dir=data_dir)
    clean = clean_truecar_dataset(raw)
    report = quality_report(clean) if not clean.empty else pd.DataFrame()

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_csv, index=False)
    if not report.empty:
        report.to_csv(report_csv, index=False)
    return clean, report
