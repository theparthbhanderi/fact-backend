import sys
import json
import logging
from app.services.multi_source_search import multi_source_search
import traceback

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    try:
        query = "Is garlic a cure for COVID-19?"
        print(f"Testing multi-source search with query: {query}")
        results = multi_source_search(query)
        print("\n--- RESULTS ---")
        for i, r in enumerate(results):
            print(f"{i+1}. [{r.get('source')}] {r.get('title')}\n   {r.get('url')}")
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
