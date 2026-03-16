import logging
from typing import List, Dict, Any, Tuple
from app.services.llm_analyzer import get_client
from app.config import settings
import json

logger = logging.getLogger(__name__)

class EvidenceConsensusAnalyzer:
    """Analyzes extracted evidence to find consensus, agreement, or contradictions."""
    
    def __init__(self):
        self.model = settings.LLM_MODEL_NAME
        try:
            self.client = get_client()
        except Exception as e:
            logger.warning(f"Could not init LLM for consensus: {e}")
            self.client = None
            
    def analyze_consensus(self, claim: str, summarized_articles: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Analyze if sources agree or contradict each other regarding the claim.
        Returns:
            status: "AGREEMENT", "CONTRADICTION", or "INCONCLUSIVE"
            agreement_bonus: Float confidence modifier.
        """
        if not summarized_articles or len(summarized_articles) < 2:
            return "INCONCLUSIVE", 0.0
            
        if not self.client:
            return "INCONCLUSIVE", 0.0
            
        # Format sources for LLM
        sources_text = ""
        for i, art in enumerate(summarized_articles, 1):
            credibility = art.get('credibility_score', 0.5)
            sources_text += f"\nSource {i} (Credibility: {credibility:.2f}):\n{art.get('summary', art.get('content', ''))[:1000]}\n"
            
        prompt = f"""You are a logical consistency analyzer.
Claim: "{claim}"

Review the following sources and determine if they universally agree, if they contradict each other, or if it's inconclusive.

Sources:
{sources_text}

Determine the consensus.
Rules:
- If sources reliably state the same conclusion regarding the claim -> "AGREEMENT"
- If sources provide conflicting facts regarding the claim -> "CONTRADICTION"
- Otherwise -> "INCONCLUSIVE"

Respond ONLY with a valid JSON object matching this schema:
{{
  "status": "AGREEMENT|CONTRADICTION|INCONCLUSIVE",
  "high_credibility_agreement_count": <integer of highly credible sources that agree>
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            status = result.get("status", "INCONCLUSIVE")
            hca_count = result.get("high_credibility_agreement_count", 0)
            
            bonus = 0.0
            if status == "AGREEMENT" and hca_count >= 3:
                bonus = 0.15 # Strong agreement bonus
            elif status == "AGREEMENT" and hca_count >= 2:
                bonus = 0.05
                
            logger.info(f"⚖️ Consensus Analysis: {status} (Bonus: +{bonus})")
            return status, bonus
            
        except Exception as e:
            logger.error(f"Consensus analysis failed: {e}")
            return "INCONCLUSIVE", 0.0

consensus_analyzer = EvidenceConsensusAnalyzer()

def analyze_evidence_consensus(claim: str, summarized_articles: List[Dict[str, Any]]) -> Tuple[str, float]:
    return consensus_analyzer.analyze_consensus(claim, summarized_articles)
