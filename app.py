from __future__ import annotations

from flask import Flask, render_template, request

from car_recommender import Preferences, make_preferences, options, recommend
from data_store import latest_scrape_runs
from truecar_live import parse_source_metro, refresh_market_choices, refresh_truecar_inventory


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
    city, state = parse_source_metro(request.form.get("refresh_metro", "boston_ma"))
    result = refresh_truecar_inventory(
        city=city,
        state=state,
        max_pages=bounded_int(request.form.get("max_pages"), 10, 1, 50),
        page_size=bounded_int(request.form.get("page_size"), 100, 10, 100),
        search_radius=bounded_int(request.form.get("search_radius"), 250, 25, 500),
        detail_limit=bounded_int(request.form.get("detail_limit"), 0, 0, 5000),
        delay_seconds=bounded_float(request.form.get("delay_seconds"), 0.75, 0.25, 5.0),
    )
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


if __name__ == "__main__":
    app.run(debug=True)
