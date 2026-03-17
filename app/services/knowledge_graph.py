import logging
import requests
from typing import Dict, Any, Optional, List
from app.config import settings

logger = logging.getLogger(__name__)

def check_google_knowledge_graph(query: str) -> Optional[Dict[str, Any]]:
    """Query the Google Knowledge Graph API for baseline entity facts."""
    api_key = getattr(settings, "GOOGLE_KG_API_KEY", None) or getattr(settings, "GOOGLE_KNOWLEDGE_GRAPH_API_KEY", None)
    if not api_key:
        logger.warning("Missing API key; skipping Google Knowledge Graph. key=%s", "GOOGLE_KG_API_KEY")
        return None
        
    url = f"https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        'query': query,
        'key': api_key,
        'limit': 1,
        'indent': True
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            items = data.get("itemListElement", [])
            if items:
                result = items[0].get("result", {})
                snippet = (
                    (result.get("detailedDescription", {}) or {}).get("articleBody", "")
                    or result.get("description", "")
                    or ""
                ).strip()
                return {
                    "source": "Google Knowledge Graph",
                    "title": result.get("name", ""),
                    "description": result.get("description", ""),
                    "detailed_description": result.get("detailedDescription", {}).get("articleBody", ""),
                    "url": result.get("detailedDescription", {}).get("url", "")
                    ,
                    # Structured evidence fields (used by knowledge verifier)
                    "snippet": snippet,
                }
    except Exception as e:
        logger.error(f"Google Knowledge Graph query failed: {e}")
        
    return None

def check_wikidata_sparql(entity_name: str) -> Optional[Dict[str, Any]]:
    """Basic Wikidata SPARQL query for factual baseline fallback."""
    url = "https://query.wikidata.org/sparql"
    # A simple query looking for the entity's description
    query = f"""
    SELECT ?item ?itemLabel ?itemDescription WHERE {{
      ?item rdfs:label "{entity_name}"@en.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 1
    """
    
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "AIFactChecker/1.0 (Contact: factchecker@example.com)"
    }
    
    try:
        response = requests.get(url, params={'query': query}, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                item = bindings[0]
                return {
                    "source": "Wikidata",
                    "title": item.get("itemLabel", {}).get("value", ""),
                    "description": item.get("itemDescription", {}).get("value", ""),
                    "url": item.get("item", {}).get("value", "")
                }
    except Exception as e:
        logger.error(f"Wikidata SPARQL query failed: {e}")
        
    return None

def fetch_knowledge_graph_fallback(claim: str) -> List[Dict[str, Any]]:
    """Attempts to fetch factual baseline data from Knowledge Graphs if news search fails."""
    logger.info(f"🧠 Initiating Knowledge Graph fallback for claim: '{claim}'")
    results = []
    
    # 1. Try Google Knowledge Graph
    gkg_data = check_google_knowledge_graph(claim)
    if gkg_data:
        snippet = f"{gkg_data['description']}. {gkg_data.get('detailed_description', '')}".strip()
        results.append({
            "title": gkg_data["title"],
            "url": gkg_data["url"],
            "source": gkg_data["source"],
            "text": snippet,
            "snippet": snippet,
            "relevance_score": 1.0 # Highest baseline relevance
        })
        
    # 2. Try Wikidata
    wiki_data = check_wikidata_sparql(claim)
    if wiki_data:
        results.append({
            "title": wiki_data["title"],
            "url": wiki_data["url"],
            "source": wiki_data["source"],
            "text": wiki_data["description"],
            "snippet": wiki_data["description"],
            "relevance_score": 0.9
        })
        
    return results
