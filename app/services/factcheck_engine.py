import logging
import difflib
from app.services.claim_normalizer import normalize_claim
from app.services.multi_source_search import multi_source_search
from app.services.evidence_retriever import retrieve_relevant_evidence
from app.services.evidence_summarizer import summarize_article
from app.services.evidence_consensus import analyze_evidence_consensus
from app.services.llm_analyzer import analyze_claim_with_llm
from app.services.confidence_engine import calculate_confidence
from app.services.spell_corrector import correct_claim_text
from app.database import SessionLocal
from app.services.history_service import save_fact_check, search_fact_checks

logger = logging.getLogger(__name__)

def run_fact_check_pipeline(claim: str) -> dict:
    """
    Run the complete, optimized fact-checking pipeline.

    Pipeline:
        Claim Normalization -> Historic Fast Path Check -> 
        Multi-Source Search -> Retrieve Relevant -> Summarize Evidence -> 
        Consensus Check -> LLM Strict Analysis -> 
        Multi-Signal Confidence Scoring -> Result

    Args:
        claim: The raw user news claim to fact-check.

    Returns:
        Structured fact-check dictionary payload.
    """
    logger.info(f"🚀 Starting fact-check pipeline for: '{claim}'")

    # ── Step 0: Spell Correction & Normalization ────────────────────
    corrected_claim = correct_claim_text(claim)
    norm_claim = normalize_claim(corrected_claim)
    logger.info(f"🧼 Normalized claim to: '{norm_claim}'")
    
    db = SessionLocal()
    try:
        past_checks = search_fact_checks(db, norm_claim, limit=5)
        for pc in past_checks:
            # Check for high string similarity (IMPROVEMENT 8)
            if difflib.SequenceMatcher(None, norm_claim.lower(), pc.claim.lower()).ratio() > 0.85:
                logger.info("⚡ FAST PATH: Found similar historic fact-check. Returning cached DB row.")
                return {
                    "original_claim": claim,
                    "corrected_claim": corrected_claim,
                    "normalized_claim": norm_claim,
                    "verdict": pc.verdict,
                    "confidence": pc.confidence,
                    "confidence_breakdown": {"cached": True},
                    "explanation": pc.explanation,
                    "evidence": pc.evidence
                }
    finally:
        db.close()

    # ── Step 1 & 2: RAG Retrieval ─────────────────────────────────
    logger.info("📰 Step 1 & 2: RAG Retrieval from Persistent FAISS & Live Fallback...")
    retrieval = retrieve_relevant_evidence(norm_claim, top_k=3)
    relevant = retrieval.get("relevant_articles", [])

    if not relevant:
        logger.warning("No relevant evidence found via RAG or Live Search — returning UNVERIFIED.")
        return {
            "original_claim": claim,
            "corrected_claim": corrected_claim,
            "normalized_claim": norm_claim,
            "verdict": "UNVERIFIED",
            "confidence": 0.3,
            "confidence_breakdown": {
                "llm_confidence": 0.0,
                "avg_similarity": 0.0,
                "avg_source_score": 0.0,
                "agreement_score": 0.0
            },
            "explanation": "No relevant evidence could be found in the knowledge base or via live multi-source search to verify this claim.",
            "evidence": [],
        }

    logger.info(f"📝 Step 2.5: Summarizing {len(relevant)} evidence articles...")
    summarized_evidence = []
    for art in relevant:
        # Use existing 'text' if run across generic extractor, fallback to description. 
        # Multi_source currently outputs description mostly, except if newspaper3k injected text.
        text_content = art.get("text", art.get("description", ""))
        summary_val = summarize_article(text_content, source=art.get("source", "Unknown"))
        
        summarized_evidence.append({
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "source": art.get("source", ""),
            "score": art.get("score", 0.0),
            "text": summary_val,  # Repackaging text variable with summary for downstream LLM analyzer
            "content": summary_val # Maintaining naming flexibility
        })

    # ── Step 3: Consensus Analysis ────────────────────────────────
    logger.info("⚖️ Step 3: Analyzing evidence consensus...")
    consensus_status, agreement_score = analyze_evidence_consensus(norm_claim, summarized_evidence)
    
    if consensus_status == "CONTRADICTION":
        logger.warning("🚨 Explicit Contradiction detected. Overriding verdict to UNVERIFIED.")
        
        # We can still proceed to save this to history as an Unverified dispute.
        dispute_result = {
            "original_claim": claim,
            "corrected_claim": corrected_claim,
            "normalized_claim": norm_claim,
            "verdict": "UNVERIFIED",
            "confidence": 0.3,
            "confidence_breakdown": {
                "llm_confidence": 0.0,
                "avg_similarity": 0.0,
                "avg_source_score": 0.0,
                "agreement_score": 0.0
            },
            "explanation": "Significant contradiction detected across extracted highly credible sources. The claim cannot be reliably resolved as True or False without further context.",
            "evidence": summarized_evidence
        }
        
        _save_to_db(dispute_result)
        return dispute_result

    # ── Step 4: LLM analysis (RAG) ────────────────────────────────
    logger.info(f"🤖 Step 4: Sending {len(summarized_evidence)} summaries to LLM...")
    llm_result = analyze_claim_with_llm(norm_claim, summarized_evidence)

    # ── Step 5: Multi-signal confidence scoring ───────────────────
    logger.info("📊 Step 5: Calculating multi-signal confidence...")
    
    confidence_result = calculate_confidence(
        llm_confidence=llm_result["confidence"],
        evidence_list=summarized_evidence,
        agreement_score=agreement_score
    )

    # ── Step 6: Build final response ──────────────────────────────
    result = {
        "original_claim": claim,
        "corrected_claim": corrected_claim,
        "normalized_claim": norm_claim,
        "verdict": llm_result["verdict"],
        "confidence": confidence_result["final_confidence"],
        "confidence_breakdown": {
            "llm_confidence": confidence_result["llm_confidence"],
            "avg_similarity": confidence_result["avg_similarity"],
            "avg_source_score": confidence_result["avg_source_score"],
            "agreement_score": confidence_result["agreement_score"]
        },
        "explanation": llm_result["explanation"],
        "evidence": summarized_evidence,
    }

    logger.info(
        f"✅ Pipeline complete: {result['verdict']} "
        f"({result['confidence']:.0%} confidence)"
    )

    # ── Step 7: Save to database ──────────────────────────────────
    _save_to_db(result)

    return result

def _save_to_db(result: dict):
    logger.info("💾 Saving result to history database...")
    db = SessionLocal()
    try:
        # Save original claim so user dashboard mapping is identical
        save_fact_check(
            db=db,
            original_claim=result["original_claim"],
            claim=result["corrected_claim"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            evidence=result["evidence"]
        )
    except Exception as e:
        logger.error(f"DB Save exception: {e}")
    finally:
        db.close()
