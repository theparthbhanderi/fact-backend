import sys
import json
import logging

logging.basicConfig(level=logging.INFO)

from app.services.translation_service import translate_fact_check_result

sample_data = {
    "claim": "Eating garlic cures COVID-19 and prevents all flus according to some random internet post found on twitter.",
    "verdict": "FALSE",
    "explanation": "There is no scientific evidence that garlic cures COVID-19. Multiple health organizations state that while garlic has antimicrobial properties, it does not prevent viral infections like COVID-19. Please rely on WHO guidelines.",
    "evidence": [
        {"title": "WHO Mythbusters: Garlic and COVID", "url": "http://who.int", "text": "Garlic is a healthy food that may have some antimicrobial properties. However, there is no evidence from the current outbreak that eating garlic has protected people from the new coronavirus."},
        {"title": "CDC Advice on Diet and COVID", "url": "http://cdc.gov", "text": "Eating a healthy diet is important for immune function, but no specific food prevents COVID-19. Garlic supplements are not recommended for prevention."},
        {"title": "FactCheck.org: Does Garlic Cure Coronavirus?", "url": "http://factcheck.org", "text": "Viral social media posts claim boiling garlic cures coronavirus. This is false. No food has been proven to cure the disease."},
        {"title": "Healthline: Garlic Benefits", "url": "http://healthline.com", "text": "Garlic may help boost your immune system, which can reduce severity of colds, but it will not cure the SARS-CoV-2 virus entirely by itself."}
    ]
}

if __name__ == "__main__":
    print("Testing Large Payload Gujarati Translation...")
    res = translate_fact_check_result(sample_data, "gu")
    print("\n--- RESULT ---")
    if res.get("claim") == sample_data["claim"]:
        print("FAILED TO TRANSLATE (Fallback kicked in)")
    else:
        print("SUCCESS")
        print(json.dumps(res, indent=2, ensure_ascii=False))
