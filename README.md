# vroom-vroom
Building Small web scraping application for Cars.com

## Flask car advisor

Run the recommendation app:

```powershell
.\.venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000.

Inventory refreshes are stored in SQLite. Set `TRUECAR_DATABASE_URL` or `DATABASE_URL` to choose the database file:

```powershell
$env:TRUECAR_DATABASE_URL = "sqlite:///data/vroom_vroom.sqlite3"
```

If TrueCar blocks plain script requests, provide browser-like request context:

```powershell
$env:TRUECAR_USER_AGENT = "Mozilla/5.0 ..."
$env:TRUECAR_ACCEPT_LANGUAGE = "en-US,en;q=0.9"
$env:TRUECAR_REFERER = "https://www.truecar.com/"
$env:TRUECAR_COOKIE = "cookie_name=cookie_value; ..."
```

The scraper sends a fuller browser-style header set by default, but a `403` can still mean TrueCar requires a real browser session, JavaScript-generated state, or permitted partner/API access. Use small page/detail limits and delays; do not use proxy rotation to bypass access controls.

Refresh TrueCar data from the command line:

```powershell
.\.venv\Scripts\python.exe truecar_live.py --city Boston --state MA --max-pages 10 --page-size 25 --search-radius 250 --detail-limit 0 --start-delay-seconds 7200
```

The refresh/download flow is intentionally separate from the search filters. It pulls used-car inventory for the selected market across a 250-mile radius; `--detail-limit 0` means enrich every listing card found. Delayed downloads use randomized pauses by default: 8-25 seconds between listing pages and 3-10 seconds between detail pages, and stop when a page is blocked.

Herb Chambers used inventory can be downloaded separately:

```powershell
.\.venv\Scripts\python.exe herb_chambers_live.py --inventory-url "https://www.herbchambers.com/used-inventory/index.htm?geoZip=02151&geoRadius=0" --max-pages 1 --start-delay-seconds 7200
```

Herb Chambers new inventory uses the same parser with the new inventory endpoint:

```powershell
.\.venv\Scripts\python.exe herb_chambers_live.py --inventory-url "https://www.herbchambers.com/new-inventory/index.htm?geoZip=02151&geoRadius=0" --max-pages 1 --start-delay-seconds 7200
```

The app ranks cars from `data/truecar_clean_combined.csv` against your price, year, and mileage ranges, location, multiple fuel types, body style, feature, and up to three buying-priority preferences. When enabled, it enriches top results with live NHTSA safety and VIN data and caches those responses in `cache/`.
