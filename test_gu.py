import sys
import json
import logging

logging.basicConfig(level=logging.INFO)

from app.services.translation_service import translate_fact_check_result

sample_data = {
    "claim": "Eating garlic cures COVID-19.",
    "verdict": "FALSE",
    "explanation": "There is no scientific evidence that garlic cures COVID-19.",
    "evidence": [{"title": "WHO Myths", "url": "http://who.int", "text": "Garlic is a healthy food, but there is no evidence it protects from COVID-19."}]
}

if __name__ == "__main__":
    print("Testing Gujarati Translation...")
    res = translate_fact_check_result(sample_data, "gu")
    print("\n--- RESULT ---")
    print(json.dumps(res, indent=2, ensure_ascii=False))
