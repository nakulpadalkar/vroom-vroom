from bs4 import BeautifulSoup
import json

with open("test.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

data = {}

# 1. Overview
overview_panel = soup.find(id="tabs_panel__r_5c__0")
if overview_panel:
    # Basic details
    details_container = overview_panel.find("div", {"data-test": "vehicleDetailsOverviewDetails"})
    if details_container:
        items = details_container.find_all("div", class_="flex items-center justify-between")
        for item in items:
            spans = item.find_all("span")
            if len(spans) >= 2:
                key = spans[0].text.strip()
                val = spans[1].text.strip()
                if key:
                    data[f"overview_{key}"] = val

    # Highlights
    highlights_container = overview_panel.find("div", {"data-test": "vehicleDetailsOverviewKeyHighlights"})
    if highlights_container:
        highlights = highlights_container.find_all("div", {"data-test": "vehicleDetailsOverviewKeyHighlight"})
        data["overview_highlights"] = [h.text.strip() for h in highlights]

# 2. Features
features_panel = soup.find(id="tabs_panel__r_5c__1")
if features_panel:
    features_container = features_panel.find("div", {"data-test": "vehicleDetailsKeyFeatures"})
    if features_container:
        categories = features_container.find_all("div", class_="flex gap-x-8 items-start")
        for cat in categories:
            title_el = cat.find("div", class_="text-18")
            if title_el:
                title = title_el.text.strip()
                lis = cat.find_all("li")
                data[f"feature_{title}"] = [li.text.strip() for li in lis]

# 3. Specs
specs_panel = soup.find(id="tabs_panel__r_5c__2")
if specs_panel:
    categories = specs_panel.find_all("div", class_="text-12")
    for cat in categories:
        title_el = cat.find("div", class_="text-18")
        if title_el:
            lis = cat.find_all("li")
            for li in lis:
                span = li.find("span")
                div = li.find("div", class_="flex text-12")
                if span and div:
                    data[f"spec_{span.text.strip()}"] = div.text.strip()

# 4. History
history_panel = soup.find(id="tabs_panel__r_5c__3")
if history_panel:
    history_tab = history_panel.find(id="history-tab")
    if history_tab:
        lis = history_tab.find_all("li")
        history_items = []
        for li in lis:
            history_items.append(li.text.strip().replace(" \n", "").strip())
        data["history"] = history_items

print(json.dumps(data, indent=2))
