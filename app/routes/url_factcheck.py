from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List
import logging
from app.services.url_scraper import fetch_article_content
from app.services.article_analyzer import extract_article_claims
from app.services.factcheck_engine import run_fact_check_pipeline
import asyncio

# Initialize Router
router = APIRouter()
logger = logging.getLogger(__name__)

class UrlRequest(BaseModel):
    url: str # Using str instead of HttpUrl for easier generic parsing

@router.post("/fact-check-url")
async def fact_check_url(request: UrlRequest) -> Dict[str, Any]:
    """
    1) Receives a URL
    2) Fetches the article content using Jina Reader/Newspaper3k
    3) Uses LLM to extract up to 5 main factual claims
    4) Runs standard fact-checking pipeline on each claim asynchronously
    5) Returns combined verdict
    """
    url = request.url
    logger.info(f"📨 Received URL for fact-check: {url}")

    if not url or not url.startswith("http"):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL format. Please provide a full valid URL starting with http:// or https://"
        )

    try:
        # 1. Fetch Article
        logger.info("📡 Step 1: Fetching article content...")
        article_data = fetch_article_content(url)
        
        # 2. Extract Claims
        logger.info("🧠 Step 2: Extracting factual claims from article text...")
        claims = extract_article_claims(article_data["text"], max_claims=5)
        
        if not claims:
            logger.warning("⚠️ No factual claims could be extracted from this article.")
            raise HTTPException(
                status_code=400,
                detail="Unable to detect any factual claims in this article to verify."
            )
            
        logger.info(f"🎯 Step 3: Running fact check on {len(claims)} individual claims...")
        
        # 3. Create a unified result object structure 
        # Since run_fact_check_pipeline is synchronous right now, we can run them sequentially or in thread pool
        final_claims_results = []
        
        # We process them sequentially for stability and rate limits. If speed is vital, 
        # this can be refactored to asyncio.gather with a background thread pool executor.
        for index, claim_text in enumerate(claims):
            try:
                logger.info(f"🔎 Fact checking claim {index+1}/{len(claims)}: '{claim_text}'")
                claim_result = run_fact_check_pipeline(claim_text)
                final_claims_results.append(claim_result)
            except Exception as e:
                logger.error(f"❌ Failed to fact check individual claim '{claim_text}': {e}")
                # We can choose to skip it or add an error object
                final_claims_results.append({
                    "claim": claim_text,
                    "verdict": "Unverified",
                    "confidence": 0,
                    "explanation": f"Pipeline failed to verify this specific claim: {str(e)}",
                    "evidence": []
                })

        # 4. Construct response
        response = {
            "article_title": article_data["title"],
            "source_url": article_data["source"],
            "claims": final_claims_results
        }
        
        logger.info(f"✅ Successfully processed URL fact check with {len(final_claims_results)} claims evaluated.")
        return response

    except HTTPException as he:
        # Re-raise HTTP exceptions to pass them to client
        raise he
    except ValueError as ve:
        logger.error(f"❌ Value Error during URL fact-checking: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"❌ Error during URL fact-checking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
