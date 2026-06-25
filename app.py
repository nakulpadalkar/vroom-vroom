from __future__ import annotations

from flask import Flask, render_template, request

from car_recommender import Preferences, make_preferences, options, recommend
from data_store import latest_scrape_runs
from herb_chambers_live import (
    DEFAULT_INVENTORY_URL as HERB_CHAMBERS_INVENTORY_URL,
    DEFAULT_NEW_INVENTORY_URL as HERB_CHAMBERS_NEW_INVENTORY_URL,
    refresh_herb_chambers_inventory,
    schedule_herb_chambers_inventory_refresh,
)
from truecar_live import (
    parse_source_metro,
    refresh_market_choices,
    refresh_truecar_inventory,
    schedule_truecar_inventory_refresh,
)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    preference_options = options()
    preferences = make_preferences(request.form) if request.method == "POST" else Preferences()
    results = recommend(preferences) if request.method == "POST" else None
    return render_template(
        "index.html",
        options=preference_options,
        preferences=preferences,
        results=results,
        refresh_markets=refresh_market_choices(),
        recent_runs=latest_scrape_runs(),
        refresh_result=None,
    )


@app.route("/refresh", methods=["POST"])
def refresh():
    inventory_source = request.form.get("inventory_source", "truecar")
    schedule_minutes = schedule_delay_minutes(request.form)

    if inventory_source in {"herb_chambers", "herb_chambers_new"}:
        default_herb_url = (
            HERB_CHAMBERS_NEW_INVENTORY_URL
            if inventory_source == "herb_chambers_new"
            else HERB_CHAMBERS_INVENTORY_URL
        )
        herb_kwargs = {
            "inventory_url": request.form.get(f"{inventory_source}_url")
            or request.form.get("herb_inventory_url")
            or default_herb_url,
            "max_pages": bounded_int(request.form.get("herb_max_pages"), 1, 1, 10),
            "listing_delay_min": bounded_float(request.form.get("listing_delay_min"), 8.0, 0.25, 120.0),
            "listing_delay_max": bounded_float(request.form.get("listing_delay_max"), 25.0, 0.25, 180.0),
        }
        if schedule_minutes > 0:
            result = schedule_herb_chambers_inventory_refresh(
                start_delay_seconds=schedule_minutes * 60,
                **herb_kwargs,
            )
        else:
            result = refresh_herb_chambers_inventory(**herb_kwargs)
    else:
        city, state = parse_source_metro(request.form.get("refresh_metro", "boston_ma"))
        refresh_kwargs = {
            "max_pages": bounded_int(request.form.get("max_pages"), 10, 1, 50),
            "page_size": bounded_int(request.form.get("page_size"), 25, 10, 100),
            "search_radius": bounded_int(request.form.get("search_radius"), 250, 25, 500),
            "detail_limit": bounded_int(request.form.get("detail_limit"), 0, 0, 5000),
            "listing_delay_min": bounded_float(request.form.get("listing_delay_min"), 8.0, 0.25, 120.0),
            "listing_delay_max": bounded_float(request.form.get("listing_delay_max"), 25.0, 0.25, 180.0),
            "detail_delay_min": bounded_float(request.form.get("detail_delay_min"), 3.0, 0.25, 60.0),
            "detail_delay_max": bounded_float(request.form.get("detail_delay_max"), 10.0, 0.25, 120.0),
        }
        if schedule_minutes > 0:
            result = schedule_truecar_inventory_refresh(
                city=city,
                state=state,
                start_delay_seconds=schedule_minutes * 60,
                **refresh_kwargs,
            )
        else:
            result = refresh_truecar_inventory(city=city, state=state, **refresh_kwargs)
    preference_options = options()
    preferences = Preferences(metro=result.source_metro if result.saved_count else "any")
    return render_template(
        "index.html",
        options=preference_options,
        preferences=preferences,
        results=None,
        refresh_markets=refresh_market_choices(),
        recent_runs=latest_scrape_runs(),
        refresh_result=result.as_dict(),
    )


def bounded_int(value: str | None, default: int, low: int, high: int) -> int:
    try:
        number = int(value or default)
    except ValueError:
        return default
    return max(low, min(high, number))


def bounded_float(value: str | None, default: float, low: float, high: float) -> float:
    try:
        number = float(value or default)
    except ValueError:
        return default
    return max(low, min(high, number))


def schedule_delay_minutes(form) -> float:
    if form.get("schedule_delay_minutes") not in (None, ""):
        return bounded_float(form.get("schedule_delay_minutes"), 60.0, 0.0, 60.0)
    return min(bounded_float(form.get("schedule_delay_hours"), 1.0, 0.0, 24.0) * 60, 60.0)


if __name__ == "__main__":
    app.run(debug=True)
