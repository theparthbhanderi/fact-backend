import sys
import json
import asyncio
from app.services.translation_service import translate_fact_check_result

sample_data = {
    "claim": "Eating garlic cures COVID-19.",
    "verdict": "FALSE",
    "explanation": "There is no scientific evidence that garlic cures COVID-19.",
    "evidence": [{"title": "WHO Myths", "text": "Garlic is a healthy food, but there is no evidence it protects from COVID-19."}]
}

print("Testing Gujarati Translation...")
res = translate_fact_check_result(sample_data, "gu")
print(json.dumps(res, indent=2, ensure_ascii=False))
