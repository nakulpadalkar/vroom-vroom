from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
import os
from tqdm import tqdm

def extract_car_details(entry):
    headers = {'User-Agent': 'Mozilla/5.0'}
    link = entry["url"]
    car_data = entry.copy()

    response = requests.get(link, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        container = soup.select_one("div.row.pt-3")
        if container:
            details = container.select("div.flex.items-center")
            for detail in details:
                text = detail.get_text(separator=" ", strip=True)
                if ":" in text:
                    key, value = text.split(":", 1)
                    car_data[key.strip().lower().replace(" ", "_")] = value.strip()
                elif "VIN" in text:
                    car_data['vin'] = text.split("VIN:")[1].strip()
                elif "Stock Number" in text:
                    car_data['stock_number'] = text.split("Stock Number:")[1].strip()
                elif "miles" in text:
                    car_data['mileage'] = text.strip()
                elif "Listed" in text:
                    car_data['listed_since'] = text.strip()

        options_container = soup.find('h2', string="Options & packages")
        if options_container:
            options_list = [
                item.get_text(separator=" ", strip=True)
                for item in options_container.find_next("div").find_all("div", class_="flex items-center")
            ]
            car_data['options_and_packages'] = ", ".join(options_list)

        popular_container = soup.find('h2', string="Popular features")
        if popular_container:
            features_list = [
                item.get_text(separator=" ", strip=True)
                for item in popular_container.find_next("div").find_all("div", class_="flex items-center")
            ]
            car_data['popular_features'] = ", ".join(features_list)

        standard_container = soup.find('h2', string="Standard features")
        if standard_container:
            std_features_list = [
                item.get_text(separator=" ", strip=True)
                for item in standard_container.find_next("div").find_all("div", class_="flex items-center")
            ]
            car_data['standard_features'] = ", ".join(std_features_list)

        price_section = soup.find('div', {'id': 'usedPriceGraph'})
        if price_section:
            line_items = price_section.select('div[data-test="usedListingPriceGraphLineItem"]')
            for item in line_items:
                label = item.get("data-test-item")
                text = item.get_text(separator="|", strip=True)
                if label and "|" in text:
                    _, value = text.split("|")
                    car_data[label.lower().replace(" ", "_")] = value.strip()

            quality_bars = price_section.select('div[data-test="priceRangeIconAndRange"]')
            for bar in quality_bars:
                quality = bar.get("data-test-item")
                range_tag = bar.find("p")
                if quality and range_tag:
                    car_data[f"price_range_{quality.lower()}"] = range_tag.text.strip()

            description = price_section.find('div', {'data-test': 'usedListingPriceGraphDescription'})
            if description:
                car_data["price_description"] = description.get_text(separator=" ", strip=True)

        seller_notes_header = soup.find('h2', string="Seller Notes")
        if seller_notes_header:
            seller_div = seller_notes_header.find_next('div', class_='see-more')
            if seller_div:
                car_data['seller_notes'] = seller_div.get_text(separator=" ", strip=True)

    except Exception as e:
        print(f"Error extracting from {link}: {e}")

    return car_data

def scrape_all_car_details(json_path, output_json, output_csv):
    with open(json_path, "r") as f:
        car_links = json.load(f)

    results = []
    for entry in tqdm(car_links, desc="Scraping car details"):
        car_info = extract_car_details(entry)
        results.append(car_info)

    df = pd.DataFrame(results)

    def split_title(title):
        if not isinstance(title, str): return [None]*4
        parts = title.replace("Used", "").strip().split(" ")
        year = parts[0] if len(parts) > 0 else None
        make = parts[1] if len(parts) > 1 else None
        model = parts[2] if len(parts) > 2 else None
        trim = " ".join(parts[3:]) if len(parts) > 3 else None
        return [year, make, model, trim]

    df[['year', 'make', 'model', 'trim']] = df['title'].apply(lambda x: pd.Series(split_title(x)))

    if 'dealer_info' in df.columns:
        df[['city_state', 'distance']] = df['dealer_info'].str.extract(r'(.+?,\s*[A-Z]{2})\s*\((.*?)\)', expand=True)
        df[['city', 'state']] = df['city_state'].str.extract(r'(.+),\s*([A-Z]{2})')

    df['price_description'] = df['price_description'].str.replace(r'([a-zA-Z])(\$)', r'\1 \2', regex=True)
    df['seller_notes'] = df['seller_notes'].str.replace(r'\.(\w)', r'. \1', regex=True)

    df.to_csv(output_csv, index=False)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Cleaned and saved car details to:\n- {output_csv}\n- {output_json}")

def scrape_city_state(city, state, max_pages=20):
    city_slug = city.lower().replace(" ", "-")
    state_slug = state.lower()
    location_key = f"{city_slug}_{state_slug}"

    all_links = []
    os.makedirs("./data", exist_ok=True)

    for i in tqdm(range(1, max_pages + 1), desc=f"{location_key} - Page"):
        url = f"https://www.truecar.com/used-cars-for-sale/listings/location-{city_slug}-{state_slug}/?page={i}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        link_tags = soup.select('a[data-test="cardLinkCover"]')

        for tag in link_tags:
            relative_link = tag.get("href")
            full_link = "https://www.truecar.com" + relative_link if relative_link else None
            if full_link:
                card = tag.find_parent("div", class_="card")
                car = {"url": full_link}
                try:
                    title_elem = card.select_one('[data-test="vehicleCardInfo"]')
                    if title_elem:
                        car["title"] = title_elem.get_text(separator=" ", strip=True)

                    mileage_elem = card.select_one('[data-test="vehicleMileage"]')
                    if mileage_elem:
                        car["mileage_listed"] = mileage_elem.get_text(separator=" ", strip=True)

                    price_elem = card.select_one('[data-test="vehicleCardPricingPrice"]')
                    if price_elem:
                        car["list_price_displayed"] = price_elem.get_text(separator=" ", strip=True)

                    dealer_elem = card.select_one('[data-test="vehicleCardFooter"]')
                    if dealer_elem:
                        car["dealer_info"] = dealer_elem.get_text(separator=" ", strip=True)
                except Exception as e:
                    print(f"Error extracting card metadata: {e}")

                all_links.append(car)

    json_path = f"./data/truecar_links_{location_key}.json"
    csv_path = f"./data/truecar_links_{location_key}.csv"
    output_json = f"./data/truecar_details_{location_key}.json"
    output_csv = f"./data/truecar_details_{location_key}.csv"

    with open(json_path, "w") as f:
        json.dump(all_links, f, indent=2)
    pd.DataFrame(all_links).to_csv(csv_path, index=False)

    print(f"Saved {len(all_links)} links for {city.title()}, {state.upper()}")
    scrape_all_car_details(json_path=json_path, output_json=output_json, output_csv=output_csv)

if __name__ == "__main__":
    # We will limit max_pages=3 for testing/initial dataset just so it finishes quickly,
    # or max_pages=20 to get the full thing? The user asked for "the initial dataset". Let's do max_pages=5 to have enough data without taking forever.
    scrape_city_state("Boston", "MA", max_pages=5)
