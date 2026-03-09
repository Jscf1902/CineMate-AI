import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.themoviedb.org"

MOVIE_URL = "https://www.themoviedb.org/movie?page={}"
TV_URL = "https://www.themoviedb.org/tv?page={}"

TARGET = 1000

headers = {
    "User-Agent": "Mozilla/5.0"
}

def get_urls(session, url, content_type, retries=3):
    for attempt in range(retries):
        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            urls = []
            cards = soup.find_all("div", class_="card style_1")

            for card in cards:
                link = card.find("a", class_="image")
                if link and link.get("href"):
                    full_url = BASE_URL + link["href"]
                    urls.append({
                        "type": content_type,
                        "url": full_url
                    })
            return urls
        except requests.exceptions.RequestException as e:
            print(f"Error on {url}")
            print(f"Attempt {attempt+1}/{retries}")
            time.sleep(3)
    print(f"Skipping page: {url}")
    return []


def scrape_category(base_url, content_type):
    collected = {}
    page = 1
    while len(collected) < TARGET:
        url = base_url.format(page)
        print(f"{content_type.upper()} | Page {page}")
        page_urls = get_urls(session, url, content_type)
        for item in page_urls:
            collected[item["url"]] = item
        print(f"Collected: {len(collected)}")
        page += 1
        time.sleep(1)
    return list(collected.values())[:TARGET]

if __name__ == "__main__":

    session = requests.Session()
    print("\nScraping MOVIES\n")

    movies = scrape_category(MOVIE_URL, "movie")
    print("\nScraping TV SERIES\n")

    series = scrape_category(TV_URL, "tv")
    all_data = movies + series

    df = pd.DataFrame(all_data)
    df.to_csv("data/raw/tmdb_urls.csv", index=False)

    print("\nDataset created")
    print(f"Movies: {len(movies)}")
    print(f"Series: {len(series)}")
    print(f"Total: {len(df)}")