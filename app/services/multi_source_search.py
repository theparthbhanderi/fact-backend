import logging
import urllib.parse
import xml.etree.ElementTree as ET
import wikipedia
import concurrent.futures
from app.config import settings
from app.services.source_credibility import get_source_credibility
from app.services.http_client import http_client
from app.services.cache_service import get_or_set_json

logger = logging.getLogger(__name__)

SCIENCE_KEYWORDS = ["vaccine", "dna", "virus", "cancer", "medical", "space", "planet"]


def search_scientific_sources(claim: str, limit: int = 3) -> list[dict]:
    """
    Scientific / medical / space fallback. For now, simulate using Wikipedia summaries.
    Returns items with the same shape as other search results: title/url/source/description.
    """
    text = (claim or "").lower()
    if not any(k in text for k in SCIENCE_KEYWORDS):
        return []

    try:
        logger.info("🔬 Searching scientific sources (simulated via Wikipedia) for: %r", claim[:120])
        search_results = wikipedia.search(claim, results=limit)
        results = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                results.append(
                    {
                        "title": page.title,
                        "url": page.url,
                        "source": "Wikipedia (Scientific)",
                        "description": page.summary[:500] + "..." if len(page.summary) > 500 else page.summary,
                    }
                )
            except Exception as e:
                logger.warning("Scientific Wikipedia fetch failed for %r: %s", title, e)
                continue
        return results
    except Exception as e:
        logger.error("Scientific source search failed: %s", e)
        return []


def search_google_fact_check(query: str, limit: int = 3) -> list[dict]:
    """Search Google Fact Check Tools API for verified claims."""
    if not settings.GOOGLE_FACT_CHECK_API_KEY:
        logger.warning("Missing API key; skipping Google Fact Check search. key=%s", "GOOGLE_FACTCHECK_API_KEY")
        return []
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={urllib.parse.quote(query)}&key={settings.GOOGLE_FACT_CHECK_API_KEY}"
    try:
        data = get_or_set_json(
            "search_google_fact_check",
            {"url": url},
            lambda: http_client.get_json(url),
        )
        results = []
        for claim in data.get("claims", [])[:limit]:
            claim_review = claim.get("claimReview", [{}])[0]
            results.append({
                "title": claim.get("text", ""),
                "url": claim_review.get("url", ""),
                "source": claim_review.get("publisher", {}).get("name", "Google Fact Check API"),
                "description": f"Verdict: {claim_review.get('textualRating', 'Unknown')}"
            })
        return results
    except Exception as e:
        logger.error(f"Google Fact Check API failed: {e}")
        return []

def search_gnews(query: str, limit: int = 5) -> list[dict]:
    """Search GNews API."""
    if not settings.GNEWS_API_KEY:
        logger.warning("Missing API key; skipping GNews search. key=%s", "GNEWS_API_KEY")
        return []
    url = f"https://gnews.io/api/v4/search?q={urllib.parse.quote(query)}&lang=en&max={limit}&apikey={settings.GNEWS_API_KEY}"
    try:
        data = get_or_set_json("search_gnews", {"url": url}, lambda: http_client.get_json(url))
        results = []
        for a in data.get("articles", []):
            results.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "GNews API"),
                "description": a.get("description", ""),
                "published_at": a.get("publishedAt", "") or "",
                "image": a.get("image", "") or "",
            })
        return results
    except Exception as e:
        logger.error(f"GNews API failed: {e}")
        return []

def search_newsdata(query: str, limit: int = 5) -> list[dict]:
    """Search NewsData.io API."""
    if not settings.NEWSDATA_API_KEY:
        logger.warning("Missing API key; skipping NewsData search. key=%s", "NEWSDATA_API_KEY")
        return []
        
    # Remove question marks or complex chars that might break strict APIs
    clean_query = query.replace("?", "").replace("!", "")
    
    url = f"https://newsdata.io/api/1/news?apikey={settings.NEWSDATA_API_KEY}&q={urllib.parse.quote(clean_query)}&language=en"
    try:
        data = get_or_set_json("search_newsdata", {"url": url}, lambda: http_client.get_json(url))
        results = []
        for a in data.get("results", [])[:limit]:
            results.append({
                "title": a.get("title", ""),
                "url": a.get("link", ""),
                "source": str(a.get("source_id", "NewsData")).capitalize(),
                "description": a.get("description", ""),
                "published_at": a.get("pubDate", "") or "",
                "image": a.get("image_url", "") or a.get("image", "") or "",
            })
        return results
    except Exception as e:
        logger.error(f"NewsData API failed: {e}")
        return []

def search_tavily(query: str, limit: int = 5) -> list[dict]:
    """Search Tavily AI Web Search API."""
    if not settings.TAVILY_API_KEY:
        logger.warning("Missing API key; skipping Tavily search. key=%s", "TAVILY_API_KEY")
        return []
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": limit
    }
    try:
        # Avoid caching secrets by not including api_key in cache key.
        cache_payload = {k: v for k, v in payload.items() if k != "api_key"}
        data = get_or_set_json("search_tavily", cache_payload, lambda: http_client.post_json(url, json=payload), ttl_s=24 * 3600)
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source": "Tavily Search API",
                "description": r.get("content", "")
            })
        return results
    except Exception as e:
        logger.error(f"Tavily API failed: {e}")
        return []

def search_newsapi(query: str, limit: int = 5) -> list[dict]:
    """Search NewsAPI for generic news."""
    if not settings.NEWS_API_KEY:
        logger.warning("Missing API key; skipping NewsAPI search. key=%s", "NEWSAPI_KEY")
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
        cache_params = {k: v for k, v in params.items() if k != "apiKey"}
        data = get_or_set_json("search_newsapi", {"url": url, "params": cache_params}, lambda: http_client.get_json(url, params=params))
        
        results = []
        for a in data.get("articles", []):
            results.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "Unknown"),
                "description": a.get("description", "") or "",
                "published_at": a.get("publishedAt", "") or "",
                "image": a.get("urlToImage", "") or "",
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
        xml_text = get_or_set_json("search_google_news_rss", {"url": url}, lambda: http_client.get_text(url))
        
        # Parse RSS XML
        root = ET.fromstring(xml_text.encode("utf-8", errors="ignore"))
        results = []
        
        # Iterate over item elements
        for item in root.findall(".//item")[:limit]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            # The source is usually in a <source> tag in Google News RSS
            source_elem = item.find("source")
            source = source_elem.text if source_elem is not None else "Google News"
            description = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "") or ""
            # Try media:content for images (namespace-agnostic)
            image = ""
            for child in list(item):
                tag = child.tag.lower()
                if tag.endswith("content") and ("media" in tag or "content" in tag):
                    image = child.attrib.get("url", "") or ""
                    if image:
                        break
            
            results.append({
                "title": title,
                "url": link,
                "source": source,
                "description": description, # Usually contains HTML in GN RSS, but we only use it for metadata
                "published_at": pub_date,
                "image": image,
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

def multi_source_search(queries: list[str]) -> list[dict]:
    """
    Search multiple sources concurrently across multiple expanded queries, 
    combine results, remove duplicates, and rank by credibility.
    """
    logger.info(f"🌐 Running parallel multi-source search for {len(queries)} queries...")
    
    all_results = []
    
    # Define mapping of search functions
    search_funcs = [
        search_newsapi,
        search_google_news,
        search_factcheck_sites,
        search_google_fact_check,
        search_gnews,
        search_newsdata,
        search_tavily
    ]
    
    # We will execute each API for each query concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_search = {}
        for query in queries:
            for func in search_funcs:
                future = executor.submit(func, query, 3) # Limit to top 3 per API per query
                future_to_search[future] = (func.__name__, query)
                
        for future in concurrent.futures.as_completed(future_to_search):
            func_name, query_used = future_to_search[future]
            try:
                data = future.result()
                if data:
                    all_results.extend(data)
            except Exception as e:
                logger.error(f"Execution failed for {func_name} on '{query_used}': {e}")

    # Scientific / medical / space enrichment (simulated via Wikipedia for now)
    if queries:
        sci = search_scientific_sources(queries[0], limit=3)
        if sci:
            all_results.extend(sci)

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
    
    # Return top 10 most credible and relevant articles across all queries
    final_results = deduped_results[:10]
    
    # --- FALLBACK LAYER: WIKIPEDIA ---
    if not final_results and queries:
        fallback_query = queries[0]
        logger.warning(f"⚠️ No news articles found. Falling back to Wikipedia for: '{fallback_query}'")
        wiki_results = search_wikipedia(fallback_query, limit=3)
        for item in wiki_results:
            item["credibility_score"] = get_source_credibility(item["source"], item["url"])
            final_results.append(item)
            
    logger.info(f"✅ Compiled {len(final_results)} unique articles from parallel multi-source search.")
    return final_results
