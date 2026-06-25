from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_store import latest_scrape_runs, record_scrape_run, save_inventory, source_metro_key
from truecar_data_pipeline import (
    DEFAULT_CITIES,
    clean_truecar_dataset,
    extract_detail_fields,
    extract_listing_cards,
)


@dataclass(frozen=True)
class RefreshResult:
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


def refresh_truecar_inventory(
    city: str,
    state: str,
    max_pages: int = 10,
    page_size: int = 25,
    search_radius: int = 250,
    detail_limit: int = 0,
    listing_delay_min: float = 8.0,
    listing_delay_max: float = 25.0,
    detail_delay_min: float = 3.0,
    detail_delay_max: float = 10.0,
) -> RefreshResult:
    city = city.strip()
    state = state.strip().upper()
    listing_delay = ordered_delay_window(listing_delay_min, listing_delay_max)
    detail_delay = ordered_delay_window(detail_delay_min, detail_delay_max)
    try:
        links = extract_listing_cards(
            city=city,
            state=state,
            max_pages=max_pages,
            page_size=page_size,
            search_radius=search_radius,
            delay_seconds=listing_delay,
        )
    except Exception as exc:  # noqa: BLE001 - web scraping can fail in several recoverable ways.
        message = f"Listing scrape failed: {exc}"
        record_scrape_run(city, state, 0, "failed", message)
        return RefreshResult(city, state, "failed", 0, 0, message)

    if links.empty:
        message = "No TrueCar listing cards were found. TrueCar may have blocked the request."
        record_scrape_run(city, state, 0, "empty", message)
        return RefreshResult(city, state, "empty", 0, 0, message)

    limited_links = links if detail_limit <= 0 else links.head(detail_limit)
    details = []
    stopped_early = False
    for row in limited_links.to_dict(orient="records"):
        detail = extract_detail_fields(row, delay_seconds=detail_delay)
        details.append(detail)
        if detail.get("detail_error"):
            stopped_early = True
            break

    raw = pd.DataFrame(details)
    clean = clean_truecar_dataset(raw)
    saved_count = save_inventory(clean, city=city, state=state)
    detail_message = "all found cards" if detail_limit <= 0 else f"{len(limited_links)} detail pages"
    stop_message = " Stopped early after a blocked or unavailable detail page." if stopped_early else ""
    message = (
        f"Saved {saved_count} cleaned listings from {len(links)} listing cards across a "
        f"{search_radius}-mile radius using {detail_message}. Randomized page pauses were enabled."
        f"{stop_message}"
    )
    return RefreshResult(city, state, "success", len(links), saved_count, message)


def ordered_delay_window(low: float, high: float) -> tuple[float, float]:
    low = max(0.0, float(low))
    high = max(0.0, float(high))
    return (min(low, high), max(low, high))


def schedule_truecar_inventory_refresh(
    city: str,
    state: str,
    start_delay_seconds: float = 7200,
    **refresh_kwargs: Any,
) -> RefreshResult:
    city = city.strip()
    state = state.strip().upper()
    start_delay_seconds = max(0.0, float(start_delay_seconds))
    message = f"Scheduled {city}, {state} inventory download to start in {format_delay(start_delay_seconds)}."
    record_scrape_run(city, state, 0, "scheduled", message)

    def run_later() -> None:
        time.sleep(start_delay_seconds)
        refresh_truecar_inventory(city=city, state=state, **refresh_kwargs)

    thread = threading.Thread(target=run_later, name=f"truecar-download-{source_metro_key(city, state)}", daemon=True)
    thread.start()
    return RefreshResult(city, state, "scheduled", 0, 0, message)


def format_delay(seconds: float) -> str:
    minutes = seconds / 60
    if minutes < 120:
        return f"{minutes:.0f} minutes"
    return f"{minutes / 60:.1f} hours"


def refresh_market_choices() -> list[dict[str, str]]:
    return [
        {
            "source_metro": source_metro_key(city, state),
            "city": city,
            "state": state,
            "label": f"{city}, {state}",
        }
        for city, state in DEFAULT_CITIES
    ]


def parse_source_metro(source_metro: str) -> tuple[str, str]:
    for choice in refresh_market_choices():
        if choice["source_metro"] == source_metro:
            return choice["city"], choice["state"]
    city_slug, _, state = source_metro.rpartition("_")
    city = city_slug.replace("-", " ").title()
    return city, state.upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh live TrueCar inventory into the configured database.")
    parser.add_argument("--city", default="Boston")
    parser.add_argument("--state", default="MA")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--search-radius", type=int, default=250)
    parser.add_argument("--detail-limit", type=int, default=0, help="0 means enrich every listing card found.")
    parser.add_argument("--listing-delay-min", type=float, default=8.0)
    parser.add_argument("--listing-delay-max", type=float, default=25.0)
    parser.add_argument("--detail-delay-min", type=float, default=3.0)
    parser.add_argument("--detail-delay-max", type=float, default=10.0)
    parser.add_argument("--start-delay-seconds", type=float, default=0.0)
    parser.add_argument("--recent-runs", action="store_true")
    args = parser.parse_args()

    if args.recent_runs:
        for run in latest_scrape_runs():
            print(run)
        return

    refresh_kwargs = {
        "max_pages": args.max_pages,
        "page_size": args.page_size,
        "search_radius": args.search_radius,
        "detail_limit": args.detail_limit,
        "listing_delay_min": args.listing_delay_min,
        "listing_delay_max": args.listing_delay_max,
        "detail_delay_min": args.detail_delay_min,
        "detail_delay_max": args.detail_delay_max,
    }
    if args.start_delay_seconds > 0:
        result = schedule_truecar_inventory_refresh(
            city=args.city,
            state=args.state,
            start_delay_seconds=args.start_delay_seconds,
            **refresh_kwargs,
        )
    else:
        result = refresh_truecar_inventory(city=args.city, state=args.state, **refresh_kwargs)
    print(result.as_dict())


if __name__ == "__main__":
    main()
