import logging
from app.services.llm_analyzer import get_client
from app.config import settings

logger = logging.getLogger(__name__)

class EvidenceSummarizer:
    """Summarizes long article texts to prevent LLM context confusion and hallucination."""
    
    def __init__(self):
        self.model = settings.LLM_MODEL_NAME
        try:
            self.client = get_client()
        except Exception as e:
            logger.warning(f"Could not initialize LLM client for summarizer: {e}")
            self.client = None
            
    def summarize(self, text: str, source: str = "Unknown") -> str:
        """
        Summarize article content focusing on headline, key facts, and official statements.
        Limits the output to a concise 300-500 word summary.
        """
        if not text or len(text.strip()) < 100:
            return text
            
        if not self.client:
            # Fallback to simple truncation if LLM is unavailable
            return text[:1500] + "..."
            
        prompt = f"""You are a professional fact-checking assistant.
Please summarize the following article excerpt. 
Focus ONLY on extracting:
1. The main headline/topic
2. Key verifiable facts
3. Official statements or quotes

Limit your response to 300-500 words. Be objective and concise.

Source: {source}
Text:
{text[:4000]} # Limit input length to save context
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for strict extraction
                max_tokens=600
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"✅ Summarized article from '{source}' ({len(text)} -> {len(summary)} chars)")
            return summary
        except Exception as e:
            logger.error(f"Summarization failed for {source}: {e}")
            # Fallback
            return text[:1500] + "..."

# Singleton
summarizer = EvidenceSummarizer()

def summarize_article(text: str, source: str = "Unknown") -> str:
    """Helper to summarize article text."""
    return summarizer.summarize(text, source)
