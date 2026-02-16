import os
import re
import json
from sys import exception
import time
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY in .env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if not GEMINI_MODEL:
    raise SystemExit("Missing GEMINI_MODEL in .env")


client = genai.Client(api_key=GEMINI_API_KEY)

DEFAULT_BATCH = 10
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds (exponential)
DEFAULT_CONTEXT_FILE = "screening-context.txt"

PROMPT_INSTRUCTION = (
    "You will receive a JSON array of papers (each with index, title, abstract). For each paper return an object with keys:\n"
    "- index: integer, same as input\n"
    "- TEST: either \"INCLUDE\" or \"EXCLUDE\"\n"
    "- REASON: only present when TEST is \"EXCLUDE\" — one of: \"TOPIC\", \"PUBLICATION_TYPE\", \"LANGUAGE\"\n\n"
    "Rules:\n"
    "1) Return ONLY a single JSON array and nothing else — no explanation, no markdown, no extra text.\n"
    "2) Output must be valid JSON parseable by a strict JSON parser.\n"
    "3) Use uppercase for TEST and REASON exactly as specified.\n"
    "4) If multiple exclusion reasons apply, pick the single most important reason using this priority order: TOPIC > PUBLICATION_TYPE > LANGUAGE.\n"
    "5) If title or abstract are empty or insufficient to decide, default to \"EXCLUDE\" with REASON \"TOPIC\".\n"
    "6) Base your decision ONLY on the provided title and abstract text; do not infer outside information.\n\n"
    "7) **I only want papers that have a hybrid approach combining subsymbolic and symbolic methods for reasoning or learning. Exclude papers that focus solely on subsymbolic methods (e.g., deep learning, neural networks) or solely on symbolic methods (e.g., logic-based systems, rule-based reasoning). The hybrid approach should integrate both paradigms to leverage their complementary strengths.**\n\n"
    "Example (must match this structure exactly):\n"
    "[{\"index\":0,\"TEST\":\"INCLUDE\"},{\"index\":1,\"TEST\":\"EXCLUDE\",\"REASON\":\"TOPIC\"}]\n\n"
)

BASE_FILTERED_DIR = Path("data") / "filtered"

def build_prompt(context_text, items):
    payload = [{"index": it["index"], "title": it["title"], "abstract": it["abstract"]} for it in items]
    prompt = PROMPT_INSTRUCTION + "Context:\n" + context_text.strip() + "\n\nPapers:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    prompt += "\n\nRespond now with the JSON array as specified."
    return prompt


def call_model(prompt):
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return resp.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            # CASE: 429 Rate Limit
            # Wait longer to let the quota reset
                print(f"  [!] Rate Limit hit. Cooling down for 60 seconds...")
                time.sleep(60)
    
            elif "503" in error_str or "SERVICE_UNAVAILABLE" in error_str:
                # CASE: 503 Service Unavailable
                # Calculate exponential backoff: 2s, 4s, 8s...
                wait_time = 2 * (2 ** (attempt - 1)) 
                print(f"  [!] Server busy (503). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            
            attempt += 1
                
            if attempt >= MAX_RETRIES:
                raise e 

def extract_json_array(text):
    # find first JSON array in text
    arr_match = re.search(r'\[.*\]', text, flags=re.DOTALL)
    obj_match = None
    if not arr_match:
        # fallback: find first JSON object or array-like content
        obj_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    match = arr_match or obj_match
    if not match:
        raise ValueError("No JSON array/object found in model response.")
    snippet = match.group(0)
    return json.loads(snippet)

def get_filtered_data(rows, title_col, abstract_col, context_text, batch_size):
    results = {}        
    for start in tqdm(range(0, len(rows), batch_size), desc="Batches"):
        batch_rows = rows[start:start+batch_size]
        items = []
        for i, r in enumerate(batch_rows):
            items.append({
                "index": start + i,
                "title": str(r.get(title_col, "")),
                "abstract": str(r.get(abstract_col, ""))
            })
        prompt = build_prompt(context_text, items)
        try:
            raw = call_model(prompt)
        except Exception as e:
            print(f"Error calling model for batch starting at index {start}: {e}", flush=True)
            return results  # return what we have so far

        parsed = extract_json_array(raw)
        if not isinstance(parsed, list):
            raise ValueError("Model returned JSON that is not a list.")
        for entry in parsed:
            idx = int(entry.get("index"))
            test = str(entry.get("TEST", "")).upper()
            reason = entry.get("REASON", "") if test == "EXCLUDE" else ""
            results[idx] = {"TEST": "INCLUDE" if test == "INCLUDE" else "EXCLUDE", "REASON": reason}

    return results


def process(input_csv: Path, context_text: str, batch_size: int, out_dir: Path):
    df = pd.read_csv(input_csv)

    # detect Title and Abstract columns case-insensitively (handles the provided header)
    title_col = next((c for c in df.columns if c.strip().lower() == "title" or "title" in c.strip().lower()), None)
    abstract_col = next((c for c in df.columns if c.strip().lower() == "abstract" or "abstract" in c.strip().lower()), None)
    if not title_col or not abstract_col:
        raise SystemExit("Input CSV must contain Title and Abstract columns. Found: " + ", ".join(df.columns))

    rows = df.to_dict(orient='records')
    results = get_filtered_data(rows, title_col, abstract_col, context_text, batch_size)

    # ensure every row has a result (default INCLUDE if missing)
    out_rows = []
    for i, orig in enumerate(rows):
        res = results.get(i, {"TEST": "INCLUDE"})
        merged = dict(orig)  # keep original columns (Authors, Title, Abstract, etc.)
        merged.update(res)   # add TEST
        out_rows.append(merged)
    out_df = pd.DataFrame(out_rows)

    # produce included and excluded files; keep REASON only for excluded rows
    included = out_df[out_df['TEST'] == 'INCLUDE'].drop(columns=['TEST'])
    excluded = out_df[out_df['TEST'] == 'EXCLUDE'].drop(columns=['TEST'])

    # ensure output dir exists
    out_dir.mkdir(parents=True, exist_ok=True)
    included_path = out_dir / "included.csv"
    excluded_path = out_dir / "excluded.csv"
    included.to_csv(included_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    # prepare log: count included and excluded per reason
    total_included = len(included)
    total_excluded = len(excluded)
    # make sure REASON column exists in excluded
    if 'REASON' not in excluded.columns:
        excluded['REASON'] = ''
    reason_counts = excluded['REASON'].fillna('UNKNOWN').replace({'': 'UNKNOWN'}).value_counts().to_dict()

    # write summary log
    log_path = out_dir / "log.txt"
    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write(f"Processing log for input: {input_csv}\n")
        lf.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        lf.write(f"Total rows processed: {len(out_df)}\n")
        lf.write(f"Included: {total_included}\n")
        lf.write(f"Excluded: {total_excluded}\n\n")
        lf.write("Excluded counts by reason:\n")
        for reason, cnt in reason_counts.items():
            lf.write(f"- {reason}: {cnt}\n")

    print(f"Wrote {total_included} included and {total_excluded} excluded rows.")
    print(f"Wrote files: {included_path}, {excluded_path}, log: {log_path}")
    return {
        'included_path': str(included_path),
        'excluded_path': str(excluded_path),
        'log_path': str(log_path),
        'included_count': total_included,
        'excluded_count': total_excluded,
        'reason_counts': reason_counts,
    }

def filter_papers_gemini(folder_name: str, context: str = None, batch_size: int = DEFAULT_BATCH, out_dir: Path = None):
    if context is None and Path(DEFAULT_CONTEXT_FILE).exists():
        context = Path(DEFAULT_CONTEXT_FILE).read_text()
    else:
        raise SystemExit("Either --context-file (existing) or --context-text must be provided.")

    # determine input cleaned CSV under data/clean/<extraction>/cleaned.csv
    input_csv = Path("data") / "clean" / folder_name / "cleaned.csv"
    if not input_csv.exists():
        raise SystemExit(f"Input cleaned CSV not found: {input_csv}")

    # output folder: data/filtered/<extraction>/
    if out_dir is None:
        out_dir = BASE_FILTERED_DIR / folder_name
    
    process(input_csv, context, batch_size, out_dir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("extraction", help="Extraction folder name under data/clean/ to process (e.g. extraction-1)")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    args = p.parse_args()

    filter_papers_gemini(args.extraction, args.batch, BASE_FILTERED_DIR / args.extraction)

if __name__ == "__main__":
    main()