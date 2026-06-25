from __future__ import annotations

import argparse
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from data_store import record_scrape_run, save_inventory, source_metro_key
from truecar_live import format_delay


BASE_URL = "https://www.herbchambers.com"
DEFAULT_USED_INVENTORY_URL = f"{BASE_URL}/used-inventory/index.htm?geoZip=02151&geoRadius=0"
DEFAULT_NEW_INVENTORY_URL = f"{BASE_URL}/new-inventory/index.htm?geoZip=02151&geoRadius=0"
DEFAULT_INVENTORY_URL = DEFAULT_USED_INVENTORY_URL
HERB_CHAMBERS_CITY = "Herb Chambers"
HERB_CHAMBERS_NEW_CITY = "Herb Chambers New"
HERB_CHAMBERS_STATE = "MA"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1",
}


@dataclass(frozen=True)
class HerbRefreshResult:
    city: str
    state: str
    status: str
    listing_count: int
    saved_count: int
    message: str

    @property
    def source_metro(self) -> str:
        return source_metro_key(self.city, self.state)

    def as_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "state": self.state,
            "source_metro": self.source_metro,
            "status": self.status,
            "listing_count": self.listing_count,
            "saved_count": self.saved_count,
            "message": self.message,
        }


def request_headers(referer: str | None = None) -> dict[str, str]:
    headers = dict(HEADERS)
    env_overrides = {
        "HERBCHAMBERS_USER_AGENT": "User-Agent",
        "HERBCHAMBERS_ACCEPT_LANGUAGE": "Accept-Language",
        "HERBCHAMBERS_COOKIE": "Cookie",
    }
    for env_name, header_name in env_overrides.items():
        if os.getenv(env_name):
            headers[header_name] = os.environ[env_name]
    if referer:
        headers["Referer"] = referer
    return headers


def sleep_for_delay(delay_seconds: float | tuple[float, float]) -> None:
    if isinstance(delay_seconds, tuple):
        low, high = delay_seconds
        time.sleep(random.uniform(min(low, high), max(low, high)))
    else:
        time.sleep(delay_seconds)


def get_soup(
    url: str,
    delay_seconds: float | tuple[float, float] = 0.5,
    referer: str | None = None,
) -> BeautifulSoup | None:
    sleep_for_delay(delay_seconds)
    try:
        response = requests.get(url, headers=request_headers(referer=referer), timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 403:
            print("Herb Chambers request blocked with HTTP 403. Stopping this source.")
        print(f"Request failed for {url}: {exc}")
        return None
    return BeautifulSoup(response.text, "html.parser")


def parse_money(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[\d,]+(?:\.\d+)?", value)
    return float(match.group(0).replace(",", "")) if match else None


def parse_integer(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"[\d,]+", value)
    return int(match.group(0).replace(",", "")) if match else None


def clean_text(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip()


def split_title(title: str | None) -> tuple[int | None, str | None, str | None, str | None]:
    if not title:
        return None, None, None, None
    clean = re.sub(r"^(Used|New|Certified)\s+", "", title.strip(), flags=re.IGNORECASE)
    parts = clean.split()
    if len(parts) < 3 or not parts[0].isdigit():
        return None, None, None, None
    year = int(parts[0])
    make = parts[1]
    model = parts[2]
    trim = " ".join(parts[3:]) if len(parts) > 3 else None
    return year, make, model, trim


def listing_card_containers(soup: BeautifulSoup) -> list:
    return soup.select(".vehicle-card-details-container")


def price_description(card) -> str | None:
    pieces = []
    for label in card.select("dl.pricing-detail dt"):
        value = label.find_next_sibling("dd")
        label_text = clean_text(label.get_text(" ", strip=True))
        value_text = clean_text(value.get_text(" ", strip=True)) if value else None
        if label_text and value_text:
            pieces.append(f"{label_text}: {value_text}")
    return "; ".join(pieces) or None


def first_price(card, selector: str) -> float | None:
    element = card.select_one(selector)
    return parse_money(element.get_text(" ", strip=True)) if element else None


def text_from_class(card, class_name: str) -> str | None:
    element = card.select_one(f".{class_name}")
    return clean_text(element.get_text(" ", strip=True)) if element else None


def parse_herb_chambers_card(
    card,
    inventory_url: str,
    source_city: str = HERB_CHAMBERS_CITY,
) -> dict[str, Any] | None:
    title_link = card.select_one(".vehicle-card-title a[href]")
    if not title_link:
        return None

    title = clean_text(title_link.get_text(" ", strip=True))
    detail_url = urljoin(BASE_URL, title_link.get("href", ""))
    question_link = card.select_one("[data-vin]")
    year, make, model, trim = split_title(title)
    if question_link:
        year = parse_integer(question_link.get("data-year")) or year
        make = question_link.get("data-make") or make
        model = question_link.get("data-model") or model
        trim = question_link.get("data-trim") or trim

    exterior = text_from_class(card, "exteriorColor")
    interior = text_from_class(card, "interiorColor")
    if exterior:
        exterior = exterior.replace(" Exterior", "").strip()
    if interior:
        interior = interior.replace(" Interior", "").strip()

    dealer_info = None
    dealer = card.select_one(".accountName span[aria-hidden='true']")
    if dealer:
        dealer_info = clean_text(dealer.get_text(" ", strip=True))

    image = card.find("img", src=re.compile(r"vehicle|inventory|photos|pictures", re.IGNORECASE))
    listing_image_url = urljoin(BASE_URL, image["src"]) if image and image.get("src") else None

    sales_price = first_price(card, ".salePrice .price-value") or first_price(card, ".askingPrice .price-value")
    odometer_miles = parse_integer(text_from_class(card, "highlight-badge"))

    return {
        "url": detail_url,
        "vin": question_link.get("data-vin") if question_link else None,
        "title": title,
        "year": year,
        "make": make,
        "model": model,
        "trim": trim,
        "sales_price": sales_price,
        "odometer_miles": odometer_miles,
        "fuel_type": None,
        "exterior": exterior,
        "interior": interior,
        "dealer_info": dealer_info,
        "dealer_city": "Boston" if dealer_info and "Boston" in dealer_info else None,
        "dealer_state": HERB_CHAMBERS_STATE,
        "dealer_distance_miles": None,
        "source_city": source_city,
        "source_state": HERB_CHAMBERS_STATE,
        "source_metro": source_metro_key(source_city, HERB_CHAMBERS_STATE),
        "popular_features": None,
        "standard_features": None,
        "options_and_packages": None,
        "seller_notes": None,
        "price_description": price_description(card),
        "stock_number": question_link.get("data-stock") if question_link else parse_stock(text_from_class(card, "stockNumber")),
        "listing_image_url": listing_image_url,
        "return_to": inventory_url,
        "inventory_source": inventory_source_name(inventory_url),
        "drivetrain": text_from_class(card, "driveLine"),
        "engine": text_from_class(card, "engine"),
    }


def parse_stock(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"Stock\s*#?\s*([A-Za-z0-9-]+)", value)
    return match.group(1) if match else value


def herb_inventory_url(base_url: str, page: int) -> str:
    if page <= 1:
        return base_url
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}start={(page - 1) * 18}"


def inventory_source_name(inventory_url: str) -> str:
    return "herb_chambers_new" if "/new-inventory/" in inventory_url else "herb_chambers"


def inventory_source_city(inventory_url: str) -> str:
    return HERB_CHAMBERS_NEW_CITY if inventory_source_name(inventory_url) == "herb_chambers_new" else HERB_CHAMBERS_CITY


def extract_herb_chambers_inventory(
    inventory_url: str = DEFAULT_INVENTORY_URL,
    max_pages: int = 1,
    delay_seconds: float | tuple[float, float] = 0.5,
) -> pd.DataFrame:
    rows = []
    source_city = inventory_source_city(inventory_url)
    for page in range(1, max_pages + 1):
        page_url = herb_inventory_url(inventory_url, page)
        soup = get_soup(page_url, delay_seconds=delay_seconds, referer=BASE_URL)
        if soup is None:
            break
        cards = listing_card_containers(soup)
        if not cards:
            print(f"No Herb Chambers listing cards found on page {page}.")
            break
        for card in cards:
            row = parse_herb_chambers_card(card, page_url, source_city=source_city)
            if row:
                rows.append(row)
    df = pd.DataFrame(rows)
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df


def refresh_herb_chambers_inventory(
    inventory_url: str = DEFAULT_INVENTORY_URL,
    max_pages: int = 1,
    listing_delay_min: float = 8.0,
    listing_delay_max: float = 25.0,
) -> HerbRefreshResult:
    delay = (min(listing_delay_min, listing_delay_max), max(listing_delay_min, listing_delay_max))
    source_city = inventory_source_city(inventory_url)
    try:
        inventory = extract_herb_chambers_inventory(
            inventory_url=inventory_url,
            max_pages=max_pages,
            delay_seconds=delay,
        )
    except Exception as exc:  # noqa: BLE001 - external source failures are recoverable.
        message = f"Herb Chambers scrape failed: {exc}"
        record_scrape_run(source_city, HERB_CHAMBERS_STATE, 0, "failed", message)
        return HerbRefreshResult(source_city, HERB_CHAMBERS_STATE, "failed", 0, 0, message)

    if inventory.empty:
        message = "No Herb Chambers listing cards were found."
        record_scrape_run(source_city, HERB_CHAMBERS_STATE, 0, "empty", message)
        return HerbRefreshResult(source_city, HERB_CHAMBERS_STATE, "empty", 0, 0, message)

    saved_count = save_inventory(
        inventory,
        city=source_city,
        state=HERB_CHAMBERS_STATE,
        message=f"Herb Chambers source: {inventory_url}",
    )
    message = f"Saved {saved_count} Herb Chambers listings from {len(inventory)} parsed cards."
    return HerbRefreshResult(source_city, HERB_CHAMBERS_STATE, "success", len(inventory), saved_count, message)


def schedule_herb_chambers_inventory_refresh(
    start_delay_seconds: float = 7200,
    **refresh_kwargs: Any,
) -> HerbRefreshResult:
    start_delay_seconds = max(0.0, float(start_delay_seconds))
    source_city = inventory_source_city(refresh_kwargs.get("inventory_url", DEFAULT_INVENTORY_URL))
    message = f"Scheduled {source_city} inventory download to start in {format_delay(start_delay_seconds)}."
    record_scrape_run(source_city, HERB_CHAMBERS_STATE, 0, "scheduled", message)

    def run_later() -> None:
        time.sleep(start_delay_seconds)
        refresh_herb_chambers_inventory(**refresh_kwargs)

    thread = threading.Thread(target=run_later, name=f"{source_metro_key(source_city, HERB_CHAMBERS_STATE)}-download", daemon=True)
    thread.start()
    return HerbRefreshResult(source_city, HERB_CHAMBERS_STATE, "scheduled", 0, 0, message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Herb Chambers used inventory into the configured database.")
    parser.add_argument("--inventory-url", default=DEFAULT_INVENTORY_URL)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--listing-delay-min", type=float, default=8.0)
    parser.add_argument("--listing-delay-max", type=float, default=25.0)
    parser.add_argument("--start-delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    refresh_kwargs = {
        "inventory_url": args.inventory_url,
        "max_pages": args.max_pages,
        "listing_delay_min": args.listing_delay_min,
        "listing_delay_max": args.listing_delay_max,
    }
    if args.start_delay_seconds > 0:
        result = schedule_herb_chambers_inventory_refresh(
            start_delay_seconds=args.start_delay_seconds,
            **refresh_kwargs,
        )
    else:
        result = refresh_herb_chambers_inventory(**refresh_kwargs)
    print(result.as_dict())


if __name__ == "__main__":
    main()
