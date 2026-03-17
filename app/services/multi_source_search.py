import logging
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import wikipedia
from app.config import settings
from app.services.source_credibility import get_source_credibility

logger = logging.getLogger(__name__)

def search_newsapi(query: str, limit: int = 5) -> list[dict]:
    """Search NewsAPI for generic news."""
    if not settings.NEWS_API_KEY:
        logger.warning("No NewsAPI key found, skipping NewsAPI search.")
        return []
        
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": limit,
        "apiKey": settings.NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for a in data.get("articles", []):
            results.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "Unknown"),
                "description": a.get("description", "")
            })
        return results
    except Exception as e:
        logger.error(f"NewsAPI search failed: {e}")
        return []

def search_google_news(query: str, limit: int = 5) -> list[dict]:
    """Search Google News RSS feed for the query."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        
        # Parse RSS XML
        root = ET.fromstring(resp.content)
        results = []
        
        # Iterate over item elements
        for item in root.findall(".//item")[:limit]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            # The source is usually in a <source> tag in Google News RSS
            source_elem = item.find("source")
            source = source_elem.text if source_elem is not None else "Google News"
            description = item.findtext("description", "")
            
            results.append({
                "title": title,
                "url": link,
                "source": source,
                "description": description # Usually contains HTML in GN RSS, but we only use it for metadata
            })
            
        return results
    except Exception as e:
        logger.error(f"Google News RSS search failed: {e}")
        return []

def search_factcheck_sites(query: str, limit: int = 3) -> list[dict]:
    """Search trusted fact-checking sites specifically via Google News RSS filtering."""
    sites = ["reuters.com", "apnews.com", "snopes.com", "politifact.com", "bbc.com"]
    site_query = " OR ".join([f"site:{s}" for s in sites])
    advanced_query = f"{query} ({site_query})"
    
    return search_google_news(advanced_query, limit=limit)

def search_wikipedia(query: str, limit: int = 2) -> list[dict]:
    """Fallback search using Wikipedia for background knowledge when news APIs fail."""
    try:
        logger.info(f"📚 Searching Wikipedia fallback for: '{query}'")
        search_results = wikipedia.search(query, results=limit)
        results = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                results.append({
                    "title": page.title,
                    "url": page.url,
                    "source": "Wikipedia",
                    "description": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
                })
            except wikipedia.exceptions.DisambiguationError as e:
                # Just take the first disambiguation option if necessary
                if e.options:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    results.append({
                        "title": page.title,
                        "url": page.url,
                        "source": "Wikipedia",
                        "description": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch wikipedia page {title}: {e}")
                continue
        return results
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return []

def multi_source_search(query: str) -> list[dict]:
    """
    Search multiple sources, combine results, remove duplicates, 
    and rank by credibility.
    """
    logger.info(f"🌐 Running multi-source search for: '{query}'")
    
    newsapi_results = search_newsapi(query, limit=5)
    gn_results = search_google_news(query, limit=5)
    fc_results = search_factcheck_sites(query, limit=3)
    
    all_results = newsapi_results + gn_results + fc_results
    
    # Deduplicate by URL
    unique_urls = set()
    deduped_results = []
    
    for item in all_results:
        url = item.get("url")
        if not url or url in unique_urls:
            continue
            
        # Add basic credibility score for ranking
        source_name = item.get("source", "")
        item["credibility_score"] = get_source_credibility(source_name, url)
        
        unique_urls.add(url)
        deduped_results.append(item)
        
    # Sort by credibility score (descending)
    deduped_results.sort(key=lambda x: x["credibility_score"], reverse=True)
    
    # Return top 8 most credible and relevant articles
    final_results = deduped_results[:8]
    
    # --- FALLBACK LAYER: WIKIPEDIA ---
    if not final_results:
        logger.warning(f"⚠️ No news articles found for '{query}'. Falling back to Wikipedia.")
        wiki_results = search_wikipedia(query, limit=3)
        for item in wiki_results:
            item["credibility_score"] = get_source_credibility(item["source"], item["url"])
            final_results.append(item)
            
    logger.info(f"✅ Compiled {len(final_results)} unique articles from multiple sources.")
    return final_results
