import re
from bs4 import BeautifulSoup
import requests

def scrape_and_parse(url):
    """
    Scrapes a URL and parses its content to extract
    title and clean body text.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # --- 1. Scrape ---
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error scraping URL: {e}")
        raise ValueError(f"Error scraping URL: {e}")

    # --- 2. Parse ---
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else 'No Title Found'
        
        content_tags = soup.find('article') or soup.find('main') or soup.find_all('p')
        if content_tags:
            if isinstance(content_tags, list):
                body_text = ' '.join([tag.get_text(separator=' ') for tag in content_tags])
            else:
                body_text = content_tags.get_text(separator=' ')
        else:
            body_text = soup.body.get_text(separator=' ') if soup.body else ''
        
        clean_text = re.sub(r'\s+', ' ', body_text).strip().lower()
        
        if not clean_text or len(clean_text.split()) == 0:
            raise ValueError("Could not extract meaningful content.")
            
        return title, clean_text

    except Exception as e:
        print(f"Error parsing content: {e}")
        raise ValueError(f"Error parsing content: {e}")

print("File 'streamlit_app/utils/parser.py' written.")
