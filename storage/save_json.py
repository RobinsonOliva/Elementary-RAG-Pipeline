# storage/save_json.py
import json

def save_to_json(records, path="imported_data.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
