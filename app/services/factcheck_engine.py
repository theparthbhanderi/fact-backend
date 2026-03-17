import logging
import difflib
import time
from typing import Optional, List, Dict, Any
from app.services.claim_normalizer import normalize_claim
from app.services.spell_corrector import correct_claim_text
from app.services.claim_processor import process_raw_claims
from app.database import SessionLocal
from app.services.history_service import save_fact_check, search_fact_checks

# Multi-LLM Agents
from app.services.claim_extractor import extract_primary_claims
from app.services.claim_memory_engine import search_similar_claim_memory
from app.services.claim_memory_engine import store_claim_memory
from app.services.query_expander import generate_search_queries
from app.services.evidence_retriever import retrieve_relevant_evidence
from app.services.knowledge_verifier import verify_claim_with_knowledge_graph
from app.services.evidence_analyst import extract_facts_from_evidence
from app.services.factcheck_judge import generate_verdict_from_dossier
from app.services.validation_agent import validate_reasoning_logic
from app.services.confidence_engine import calculate_confidence
from app.services.evidence_consensus_engine import analyze_evidence_consensus
from app.services.social_signal_analyzer import analyze_social_signals, social_sources_to_evidence

logger = logging.getLogger(__name__)

def _count_sources(evidence: List[Dict[str, Any]]) -> int:
    return len({(e.get("source") or "").strip() for e in evidence if (e.get("source") or "").strip()})


def _log_stage(stage: str, claim_id: str, start_time: float, evidence: Optional[List[Dict[str, Any]]] = None) -> None:
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    num_sources = _count_sources(evidence or [])
    logger.info(
        "stage=%s claim_id=%s execution_time_ms=%s number_of_sources=%s",
        stage,
        claim_id,
        elapsed_ms,
        num_sources,
    )

def run_fact_check_pipeline(raw_input: str) -> dict:
    """
    Run the complete Multi-LLM fact-checking pipeline.

    Flow:
    Input -> LLM Claim Extractor -> LLM Query Expander -> Parallel Search APIs -> 
    LLM Evidence Analyst -> LLM Fact-Check Judge -> LLM Validator -> Confidence Engine -> Output
    """
    logger.info(f"🚀 Starting Multi-LLM pipeline for input: '{raw_input}'")

    t0 = time.perf_counter()
    extracted_claims = extract_primary_claims(raw_input) or []
    _log_stage("claim_extraction", claim_id="request", start_time=t0, evidence=[])
    raw_claims = extracted_claims if extracted_claims else [raw_input]
    claim_objects = process_raw_claims(raw_input=raw_input, claims=raw_claims)
    if not claim_objects:
        claim_objects = process_raw_claims(raw_input=raw_input, claims=[raw_input])

    results: list[dict] = []
    for idx, claim_obj in enumerate(claim_objects):
        claim_id = f"claim_{idx+1}"
        claim = (claim_obj.get("claim") or "").strip()
        norm_claim = (claim_obj.get("normalized_claim") or "").strip() or normalize_claim(claim)
        try:
            # Memory layer (after claim extraction, before full pipeline)
            mem = search_similar_claim_memory(norm_claim)
            if mem.get("memory_match"):
                boosted_conf = min(float(mem.get("confidence", 0.0) or 0.0) + 0.05, 0.98)
                results.append(
                    {
                        "claim_id": claim_id,
                        "claim_hash": claim_obj.get("hash"),
                        "language": claim_obj.get("language"),
                        "entities": claim_obj.get("entities") or [],
                        "topics": claim_obj.get("topics") or [],
                        "original_claim": raw_input,
                        "corrected_claim": claim,
                        "normalized_claim": norm_claim,
                        "verdict": mem.get("verdict", "UNVERIFIED"),
                        "confidence": boosted_conf,
                        "confidence_breakdown": {
                            "llm_confidence": 0.0,
                            "avg_similarity": float(mem.get("similarity_score", 0.0) or 0.0),
                            "avg_source_score": 0.0,
                            "agreement_score": 0.0,
                            "knowledge_score": 0.0,
                            "memory_hit": True,
                        },
                        "explanation": mem.get("explanation", ""),
                        "evidence": mem.get("evidence", []) or [],
                    }
                )
                continue

            results.append(
                _run_single_claim_pipeline(
                    raw_input=raw_input,
                    claim=claim,
                    claim_obj=claim_obj,
                    claim_id=claim_id,
                )
            )
        except Exception as e:
            logger.exception("Single-claim pipeline failed; returning UNVERIFIED for claim. claim_id=%s error=%s", claim_id, e)
            corrected_claim = correct_claim_text(claim)
            norm_claim = normalize_claim(corrected_claim)
            fallback = _build_unverified_response(raw_input, corrected_claim, norm_claim)
            fallback["claim_id"] = claim_id
            fallback["claim_hash"] = claim_obj.get("hash")
            fallback["language"] = claim_obj.get("language")
            fallback["entities"] = claim_obj.get("entities") or []
            fallback["topics"] = claim_obj.get("topics") or []
            results.append(fallback)

    # Backward-compatible top-level fields remain sourced from the first claim result.
    primary = results[0] if results else _build_unverified_response(raw_input, raw_input, normalize_claim(raw_input))
    aggregated = {**primary, "claims": results}
    return aggregated


def _run_single_claim_pipeline(raw_input: str, claim: str, claim_obj: Dict[str, Any], claim_id: str) -> dict:
    t_start = time.perf_counter()
    corrected_claim = correct_claim_text(claim)
    norm_claim = (claim_obj.get("normalized_claim") or "").strip() or normalize_claim(corrected_claim)
    logger.info(f"🧼 Target normalized claim: '{norm_claim}' (claim_id={claim_id})")

    # ── Social Signal Analyzer (pre-news) ──────────────────────────
    t_social = time.perf_counter()
    social_signals = analyze_social_signals(norm_claim)
    _log_stage("social_signal_analysis", claim_id=claim_id, start_time=t_social, evidence=[])
    social_evidence = social_sources_to_evidence(norm_claim, social_signals.get("social_sources", []) or [])

    # DB Check for fast path
    db = SessionLocal()
    try:
        past_checks = search_fact_checks(db, norm_claim, limit=5)
        for pc in past_checks:
            if difflib.SequenceMatcher(None, norm_claim.lower(), pc.claim.lower()).ratio() > 0.85:
                logger.info("⚡ FAST PATH: Found similar historic fact-check in DB. claim_id=%s", claim_id)
                cached_evidence = pc.evidence or []
                unique_sources = len({(e.get("source") or "").strip() for e in cached_evidence if (e.get("source") or "").strip()})
                knowledge_score = 0.0
                # Social signals (cached fast-path)
                social_signals = analyze_social_signals(norm_claim)
                social_evidence = social_sources_to_evidence(norm_claim, social_signals.get("social_sources", []) or [])
                social_sorted = sorted(
                    social_evidence, key=lambda x: float(x.get("similarity_score", 0.0) or 0.0), reverse=True
                )[:2]

                if unique_sources < 3:
                    kg_result = verify_claim_with_knowledge_graph(norm_claim)
                    kg_evidence = kg_result.get("knowledge_evidence", []) or []
                    knowledge_score = float(kg_result.get("knowledge_score", 0.0) or 0.0)
                    if kg_evidence:
                        kg_sorted = sorted(
                            kg_evidence,
                            key=lambda x: float(x.get("similarity_score", x.get("score", 0.0)) or 0.0),
                            reverse=True,
                        )
                        news_sorted = sorted(
                            cached_evidence,
                            key=lambda x: float(x.get("similarity_score", x.get("score", 0.0)) or 0.0),
                            reverse=True,
                        )
                        reserved_kg = kg_sorted[:3]
                        combined = reserved_kg + social_sorted + [e for e in news_sorted if e not in reserved_kg and e not in social_sorted]
                        cached_evidence = combined[:12]
                        for idx, e in enumerate(cached_evidence, 1):
                            e["evidence_rank"] = idx
                elif social_sorted:
                    # No KG merge, still keep a couple social items
                    news_sorted = sorted(
                        cached_evidence,
                        key=lambda x: float(x.get("similarity_score", x.get("score", 0.0)) or 0.0),
                        reverse=True,
                    )
                    cached_evidence = (social_sorted + [e for e in news_sorted if e not in social_sorted])[:12]
                    for idx, e in enumerate(cached_evidence, 1):
                        e["evidence_rank"] = idx

                # Backfill FAISS claim memory for future hits (best-effort)
                try:
                    store_claim_memory(
                        claim=pc.claim,
                        verdict=pc.verdict,
                        confidence=float(pc.confidence),
                        explanation=pc.explanation,
                        evidence=cached_evidence,
                        search_queries=[],
                    )
                except Exception:
                    pass

                return {
                    "claim_id": claim_id,
                    "claim_hash": claim_obj.get("hash"),
                    "language": claim_obj.get("language"),
                    "entities": claim_obj.get("entities") or [],
                    "topics": claim_obj.get("topics") or [],
                    "original_claim": raw_input,
                    "corrected_claim": corrected_claim,
                    "normalized_claim": norm_claim,
                    "verdict": pc.verdict,
                    "confidence": pc.confidence,
                    "confidence_breakdown": {
                        "llm_confidence": 0.0,
                        "avg_similarity": 0.0,
                        "avg_source_score": 0.0,
                        "agreement_score": 0.0,
                        "knowledge_score": knowledge_score,
                        "cached": True,
                    },
                    "explanation": pc.explanation,
                    "evidence": cached_evidence,
                }
    finally:
        db.close()

    # ── Step 1: Query Expansion (LLM 2) ──────────────────────────
    t1 = time.perf_counter()
    logger.info("🔍 Step 1: Generating search queries via LLM... claim_id=%s", claim_id)
    expanded_queries = generate_search_queries(norm_claim)
    if not expanded_queries:
        expanded_queries = [norm_claim]
    _log_stage("query_expansion", claim_id=claim_id, start_time=t1, evidence=[])

    # ── Step 2: RAG Retrieval (Parallel Live Search) ─────────────
    t2 = time.perf_counter()
    logger.info("📰 Step 2: RAG Retrieval via Parallel Search APIs... claim_id=%s", claim_id)
    retrieval = retrieve_relevant_evidence(norm_claim, expanded_queries, top_k=5)
    relevant_articles = retrieval.get("relevant_articles", [])
    _log_stage("evidence_retrieval", claim_id=claim_id, start_time=t2, evidence=relevant_articles)

    # Append a small number of social evidence items (low credibility) without replacing news evidence.
    social_sorted = []
    if social_evidence:
        social_sorted = sorted(
            social_evidence, key=lambda x: float(x.get("similarity_score", 0.0) or 0.0), reverse=True
        )[:2]
        relevant_articles = (relevant_articles or []) + social_sorted

    knowledge_score = 0.0
    # ── Knowledge Graph Verification fallback (weak evidence) ───────
    unique_sources = len({(e.get("source") or "").strip() for e in relevant_articles if (e.get("source") or "").strip()})
    if unique_sources < 3:
        logger.info(
            "Knowledge verification triggered (weak evidence). claim_id=%s evidence_count=%s unique_sources=%s",
            claim_id,
            len(relevant_articles),
            unique_sources,
        )
        kg_result = verify_claim_with_knowledge_graph(norm_claim)
        kg_evidence = kg_result.get("knowledge_evidence", []) or []
        knowledge_score = float(kg_result.get("knowledge_score", 0.0) or 0.0)
        if kg_evidence:
            # Keep some dedicated slots for knowledge evidence so it can't be fully truncated by high-similarity news snippets.
            kg_sorted = sorted(
                kg_evidence,
                key=lambda x: float(x.get("similarity_score", x.get("score", 0.0)) or 0.0),
                reverse=True,
            )
            news_sorted = sorted(
                relevant_articles,
                key=lambda x: float(x.get("similarity_score", x.get("score", 0.0)) or 0.0),
                reverse=True,
            )

            reserved_kg = kg_sorted[:3]
            combined = reserved_kg + social_sorted + [e for e in news_sorted if e not in reserved_kg and e not in social_sorted]
            relevant_articles = combined[:12]
            for idx, e in enumerate(relevant_articles, 1):
                e["evidence_rank"] = idx
    elif social_sorted:
        # No KG merge, still keep a couple social items
        news_sorted = sorted(
            relevant_articles,
            key=lambda x: float(x.get("similarity_score", x.get("score", 0.0)) or 0.0),
            reverse=True,
        )
        relevant_articles = (social_sorted + [e for e in news_sorted if e not in social_sorted])[:12]
        for idx, e in enumerate(relevant_articles, 1):
            e["evidence_rank"] = idx

    if not relevant_articles:
        logger.warning("No relevant news articles found. Triggering Knowledge Graph Fallback... claim_id=%s", claim_id)
        from app.services.knowledge_graph import fetch_knowledge_graph_fallback
        fallback_results = fetch_knowledge_graph_fallback(norm_claim)

        if fallback_results:
            logger.info("Found fallback facts from Knowledge Graph. claim_id=%s", claim_id)
            relevant_articles = fallback_results
        else:
            resp = _build_unverified_response(raw_input, corrected_claim, norm_claim)
            resp["claim_id"] = claim_id
            return resp

    # ── Step 3: Evidence Analyst (LLM 3) ─────────────────────────
    t3 = time.perf_counter()
    logger.info("📝 Step 3: LLM Evidence Analyst extracting facts... claim_id=%s", claim_id)
    fact_dossier = extract_facts_from_evidence(norm_claim, relevant_articles)
    _log_stage("evidence_analysis", claim_id=claim_id, start_time=t3, evidence=relevant_articles)

    # ── Step 4: Cross-Source Evidence Consensus Engine ────────────
    consensus = analyze_evidence_consensus(norm_claim, relevant_articles)
    agreement_score = float(consensus.get("agreement_score", 0.0) or 0.0)

    # ── Step 5: Fact-Check Judge (LLM 4) ─────────────────────────
    t4 = time.perf_counter()
    logger.info("⚖️ Step 4: LLM Fact-Check Judge evaluating dossier... claim_id=%s", claim_id)
    judge_result = generate_verdict_from_dossier(norm_claim, fact_dossier, consensus=consensus, social=social_signals)
    _log_stage("verdict_generation", claim_id=claim_id, start_time=t4, evidence=relevant_articles)

    verdict = judge_result["verdict"]
    explanation = judge_result["explanation"]
    llm_confidence = judge_result["confidence"]

    # ── Step 6: Validation Agent (LLM 5) ─────────────────────────
    t5 = time.perf_counter()
    logger.info("🛡️ Step 5: Validation Agent verifying logic... claim_id=%s", claim_id)
    max_retries = 2
    for attempt in range(0, max_retries + 1):  # initial + up to 2 retries
        is_logical = validate_reasoning_logic(verdict, explanation)
        if is_logical:
            break
        if attempt >= max_retries:
            break
        logger.warning("Validation failed; retrying judge. claim_id=%s attempt=%s", claim_id, attempt + 1)
        judge_result_retry = generate_verdict_from_dossier(
            norm_claim,
            fact_dossier
            + "\n\nCRITICAL NOTE: Your explanation must logically support your verdict. Include supporting/contradicting counts and plausibility.",
            consensus=consensus,
            social=social_signals,
        )
        verdict = judge_result_retry["verdict"]
        explanation = judge_result_retry["explanation"]
        llm_confidence = judge_result_retry["confidence"]
    _log_stage("validation", claim_id=claim_id, start_time=t5, evidence=relevant_articles)

    # ── Step 6: Multi-signal confidence scoring ───────────────────
    logger.info("📊 Step 6: Calculating multi-signal confidence... claim_id=%s", claim_id)

    confidence_result = calculate_confidence(
        llm_confidence=llm_confidence,
        evidence_list=relevant_articles,
        agreement_score=agreement_score,
        knowledge_score=knowledge_score,
    )

    # ── Step 7: Build final response ──────────────────────────────
    result = {
        "claim_id": claim_id,
        "claim_hash": claim_obj.get("hash"),
        "language": claim_obj.get("language"),
        "entities": claim_obj.get("entities") or [],
        "topics": claim_obj.get("topics") or [],
        "original_claim": raw_input,
        "corrected_claim": corrected_claim,
        "normalized_claim": norm_claim,
        "verdict": verdict,
        "confidence": confidence_result["final_confidence"],
        "confidence_breakdown": {
            "llm_confidence": confidence_result["llm_confidence"],
            "avg_similarity": confidence_result["avg_similarity"],
            "avg_source_score": confidence_result["avg_source_score"],
            "agreement_score": confidence_result["agreement_score"],
            "knowledge_score": confidence_result.get("knowledge_score", 0.0),
        },
        "explanation": explanation,
        "evidence": relevant_articles,
        "search_queries": expanded_queries,
    }

    logger.info(
        "✅ Pipeline complete. claim_id=%s verdict=%s confidence=%s",
        claim_id,
        result["verdict"],
        result["confidence"],
    )

    # Structured log (Phase 13)
    try:
        sources_count = _count_sources(relevant_articles or [])
        exec_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info(
            "FACT_CHECK_PIPELINE_COMPLETED claim=%r verdict=%s confidence=%.2f sources_count=%s execution_time_ms=%s",
            norm_claim[:200],
            str(result.get("verdict", "")),
            float(result.get("confidence", 0.0) or 0.0),
            sources_count,
            exec_ms,
        )
    except Exception:
        pass

    _save_to_db(result)
    return result

def _build_unverified_response(raw, corrected, norm):
    return {
        "original_claim": raw,
        "corrected_claim": corrected,
        "normalized_claim": norm,
        "verdict": "UNVERIFIED",
        "confidence": 0.3,
        "confidence_breakdown": {
            "llm_confidence": 0.0,
            "avg_similarity": 0.0,
            "avg_source_score": 0.0,
            "agreement_score": 0.0
        },
        "explanation": "No relevant evidence could be found via live multi-source search to verify this claim.",
        "evidence": [],
    }

def _build_dispute_response(raw, corrected, norm, evidence):
    result = {
        "original_claim": raw,
        "corrected_claim": corrected,
        "normalized_claim": norm,
        "verdict": "DISPUTED",
        "confidence": 0.3,
        "confidence_breakdown": {
            "llm_confidence": 0.0,
            "avg_similarity": 0.0,
            "avg_source_score": 0.0,
            "agreement_score": 0.0
        },
        "explanation": "Significant contradiction detected across extracted highly credible sources. The claim cannot be reliably resolved as True or False without further context.",
        "evidence": evidence
    }
    _save_to_db(result)
    return result

def _save_to_db(result: dict):
    logger.info("💾 Saving result to history database...")
    db = SessionLocal()
    try:
        save_fact_check(
            db=db,
            original_claim=result["original_claim"],
            claim=result["corrected_claim"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            evidence=result["evidence"],
            search_queries=result.get("search_queries", []) or [],
        )
    except Exception as e:
        logger.error(f"DB Save exception: {e}")
    finally:
        db.close()
