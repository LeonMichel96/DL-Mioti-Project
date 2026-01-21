import json
import time
from pathlib import Path
from engine import MagicJudgeEngine

# --- CONFIGURATION ---
INPUT_FILE = Path("curated_questions.json")
OUTPUT_FILE = Path("curated_questions_owl.json")

def run_batch_inference():
    # 1. Validation
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file '{INPUT_FILE}' not found.")
        return

    # 2. Load Data
    print(f"[LOG] Loading questions from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Initialize Engine
    print("[LOG] Awakening the Judge's Familiar (Loading Engine)...")
    print("(This may take 30-60 seconds to load the vector database into memory)")
    judge = MagicJudgeEngine(force_rebuild=False)

    results = []
    print(f"[LOG] Starting inference on {len(data)} questions...\n")

    # 4. Processing Loop
    for i, entry in enumerate(data):
        question_id = entry.get("id", "Unknown")
        question_text = entry.get("question", "")
        
        print(f"Doing question {i+1}/{len(data)} (ID: {question_id})...")
        
        try:
            # Query the engine
            stream_response = judge.query(question_text, history=[])
            
            full_response = ""
            for token in stream_response:
                full_response += str(token.delta)
            
            entry["owl_answer"] = full_response.strip()

        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
            entry["owl_answer"] = f"SYSTEM ERROR: {str(e)}"

        results.append(entry)
        
        # Save incrementally every 10 questions (just in case it crashes)
        if (i + 1) % 10 == 0:
            print(f"   (Saving partial progress to {OUTPUT_FILE}...)")
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Small pause to be nice to the API
        time.sleep(0.2)

    # 5. Final Save
    print(f"\n[LOG] Final save to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("[SUCCESS] Batch inference complete.")

if __name__ == "__main__":
    run_batch_inference()
