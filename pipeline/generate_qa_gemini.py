#!/usr/bin/env python3
"""
Medical Q&A Pipeline - Optimized
- API key rotation (rotates on quota/error, prompts for new key if all exhausted)
- Checkpoint/resume: stops and resumes exactly where you left off
- Structured JSON output via Gemini responseSchema (no parsing failures)
- No input or output truncation
- Final output: one merged training.jsonl
"""

import argparse
import os
import re
import json
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "gemini-2.5-flash-lite"
API_DELAY           = 2      # seconds between successful calls
RETRY_DELAY         = 8      # seconds between retries
MAX_RETRIES         = 4      # retries per key before rotating
BATCH_DELAY         = 5      # seconds between files
CONFIDENCE_CUTOFF   = 0.6    # pairs below this go to review, not training
CHUNK_SIZE          = 12000  # chars per chunk — keeps output safely under 65k tokens
CHUNK_OVERLAP       = 400    # chars of overlap between chunks to avoid boundary loss
CHECKPOINT_FILE     = "checkpoint.json"
SUMMARY_FILE        = "pipeline_summary.json"
TRAINING_FILE       = "training.jsonl"
REVIEW_FILE         = "low_confidence_review.jsonl"

TOPICS = [
    "constitution", "temperament", "age_group", "gender_symptoms",
    "pathology_theme", "miasm", "prognosis", "modalities_better",
    "modalities_worse", "seasonal_variations", "time_aggravations",
    "discharges", "ulcer_character", "odour", "bone_periosteum",
    "nervous_system", "gastrointestinal", "female_reproductive",
    "male_reproductive", "respiratory", "cardiovascular",
    "mental_emotional", "pain_type", "physical_signs",
    "therapeutic_indications", "potency_response", "complementary_remedies",
]

# Gemini responseSchema — forces valid JSON at the token-sampler level
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "question":   {"type": "STRING"},
            "answer":     {"type": "STRING"},
            "excerpt":    {"type": "STRING"},
            "topic":      {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["question", "answer", "excerpt", "topic", "confidence"],
    },
}

# ── Graceful shutdown ─────────────────────────────────────────────────────────
_shutdown_requested = False

def _handle_sigint(sig, frame):
    global _shutdown_requested
    print("\n\n⚠  Interrupt received — finishing current file then saving checkpoint...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _handle_sigint)

# ── Checkpoint ────────────────────────────────────────────────────────────────

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return {"completed": [], "failed": [], "started_at": datetime.now().isoformat()}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def is_done(self, remedy_id: str) -> bool:
        return remedy_id in self.data["completed"]

    def mark_done(self, remedy_id: str, stats: dict):
        if remedy_id not in self.data["completed"]:
            self.data["completed"].append(remedy_id)
        self.data.setdefault("stats", {})[remedy_id] = stats
        self.save()

    def mark_failed(self, remedy_id: str, reason: str):
        entry = {"id": remedy_id, "reason": reason, "time": datetime.now().isoformat()}
        self.data["failed"] = [f for f in self.data["failed"] if f["id"] != remedy_id]
        self.data["failed"].append(entry)
        self.save()

# ── API Key Pool ──────────────────────────────────────────────────────────────

class KeyPool:
    """Rotates through API keys; prompts for a new one if all are exhausted."""

    def __init__(self, keys: List[str]):
        self.keys = list(keys)
        self.exhausted: set = set()
        self.current_idx = 0

    @property
    def current(self) -> Optional[str]:
        available = [k for k in self.keys if k not in self.exhausted]
        if not available:
            return None
        return available[self.current_idx % len(available)]

    def rotate(self, mark_exhausted: bool = False):
        available = [k for k in self.keys if k not in self.exhausted]
        if mark_exhausted and available:
            key = available[self.current_idx % len(available)]
            self.exhausted.add(key)
            print(f"  🔑 Key ...{key[-6:]} marked exhausted.")

        available = [k for k in self.keys if k not in self.exhausted]
        if not available:
            print("\n⚠  All API keys exhausted or failed.")
            new_key = input("  Enter a new Gemini API key (or press Enter to abort): ").strip()
            if new_key:
                self.keys.append(new_key)
                self.exhausted.discard(new_key)
                print(f"  ✓ New key added.")
            else:
                print("  Aborting.")
                sys.exit(1)

        self.current_idx = (self.current_idx + 1) % max(1, len(self.keys) - len(self.exhausted))

# ── Gemini Client ─────────────────────────────────────────────────────────────

class GeminiClient:
    def __init__(self, key_pool: KeyPool, model: str = DEFAULT_MODEL):
        self.pool = key_pool
        self.model = model
        self.call_count = 0
        self.base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def call(self, prompt: str, temperature: float = 0.4) -> Optional[List[dict]]:
        """
        Call Gemini with structured output (responseSchema).
        Returns a parsed list of dicts, or None on total failure.
        """
        import requests

        keys_tried = 0
        total_keys = len([k for k in self.pool.keys if k not in self.pool.exhausted])

        while keys_tried <= total_keys:
            key = self.pool.current
            if not key:
                self.pool.rotate()
                key = self.pool.current

            url = f"{self.base}?key={key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 65536,
                    "temperature": temperature,
                    "responseMimeType": "application/json",
                    "responseSchema": RESPONSE_SCHEMA,
                },
            }

            for attempt in range(MAX_RETRIES):
                self.call_count += 1
                print(f"  📡 API call #{self.call_count} (key ...{key[-6:]}, attempt {attempt+1})... ", end="", flush=True)

                try:
                    r = requests.post(url, json=payload, timeout=180)

                    if r.status_code == 429:
                        print("Rate limited.")
                        self.pool.rotate(mark_exhausted=True)
                        keys_tried += 1
                        break

                    if r.status_code in (401, 403):
                        print(f"Auth error ({r.status_code}).")
                        self.pool.rotate(mark_exhausted=True)
                        keys_tried += 1
                        break

                    if r.status_code == 503:
                        wait = 15 * (attempt + 1)
                        print(f"Server busy (503) — waiting {wait}s...")
                        time.sleep(wait)
                        continue

                    r.raise_for_status()
                    data = r.json()

                    candidates = data.get("candidates", [])
                    if not candidates:
                        reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                        print(f"Blocked: {reason}")
                        time.sleep(RETRY_DELAY)
                        continue

                    finish_reason = candidates[0].get("finishReason", "")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = parts[0].get("text", "").strip() if parts else ""

                    if not text:
                        print(f"Empty response (finishReason={finish_reason}).")
                        time.sleep(RETRY_DELAY)
                        continue

                    # With responseSchema, this should always succeed.
                    # If it still fails it means the output was truncated mid-JSON
                    # (maxOutputTokens hit). Retry with a smaller ask.
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        print(f"Truncated JSON (finishReason={finish_reason}) — will retry with smaller ask")
                        time.sleep(RETRY_DELAY)
                        # Shrink the prompt's pair-count target for next attempt
                        prompt = prompt.replace(
                            "Aim for 20-60 pairs depending on text length",
                            "Aim for AT MOST 30 pairs. Be concise."
                        )
                        continue
                    if not isinstance(result, list):
                        print(f"Unexpected type: {type(result)}")
                        time.sleep(RETRY_DELAY)
                        continue

                    print(f"OK — {len(result)} pairs (finishReason={finish_reason})")
                    time.sleep(API_DELAY)
                    return result

                except Exception as e:
                    print(f"Error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY + attempt * 2)
                    else:
                        self.pool.rotate()
                        keys_tried += 1
                        break

        print("  ✗ All keys failed for this call.")
        return None

# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(remedy_name: str, text: str) -> str:
    topics_str = "\n".join(f"- {t}" for t in TOPICS)
    return f"""You are extracting training data from a homeopathic remedy monograph.

REMEDY: {remedy_name}

TEXT:
{text}

Extract every distinct clinical fact as a Question-Answer pair. Cover as many of these topics as are present in the text:
{topics_str}

Rules:
- question: specific clinical question about {remedy_name}
- answer: complete, standalone answer drawn from the text
- excerpt: the 1-2 sentence quote from the text that supports the answer
- topic: one of the topic names listed above
- confidence: number 0.5-1.0 reflecting how clearly the text supports this pair
- Aim for 15-30 pairs per chunk depending on text length
"""

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks on sentence boundaries where possible.
    Overlap ensures facts that straddle a boundary aren't lost.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to break on a sentence boundary ('. ') within last 500 chars of window
        boundary = text.rfind('. ', end - 500, end)
        if boundary != -1:
            end = boundary + 1  # include the period
        chunks.append(text[start:end])
        start = end - overlap  # step back by overlap for next chunk

    return chunks


def dedup_pairs(pairs: List[dict]) -> List[dict]:
    """
    Remove near-duplicate questions across chunks.
    Keeps the first occurrence; compares lowercased stripped question strings.
    """
    seen = set()
    out = []
    for p in pairs:
        key = re.sub(r'\s+', ' ', p.get("question", "").lower().strip())
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out




def process_file(
    filepath: Path,
    remedy_id: str,
    remedy_name: str,
    output_dir: Path,
    client: GeminiClient,
) -> Optional[Dict]:
    """Process one remedy file → returns stats dict or None on failure."""

    print(f"\n{'─'*60}")
    print(f"  Remedy : {remedy_name}")
    print(f"  File   : {filepath.name}")
    print(f"{'─'*60}")

    raw_text = filepath.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r'\s+', ' ', raw_text).strip()
    print(f"  📄 Text length: {len(text):,} chars")

    chunks = chunk_text(text)
    if len(chunks) > 1:
        print(f"  ✂  Split into {len(chunks)} chunks (~{CHUNK_SIZE:,} chars each)")

    all_pairs: List[dict] = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"  📦 Chunk {i}/{len(chunks)} ({len(chunk):,} chars)")
        prompt = build_prompt(remedy_name, chunk)
        pairs = client.call(prompt, temperature=0.4)
        if pairs:
            all_pairs.extend(pairs)
        else:
            print(f"  ⚠  Chunk {i} returned no pairs — continuing with remaining chunks")

    if not all_pairs:
        print(f"  ✗ Zero Q&A pairs extracted for {remedy_name}")
        return None

    # Deduplicate across chunks
    all_pairs = dedup_pairs(all_pairs)
    print(f"  ✓ {len(all_pairs)} unique pairs after dedup")

    training_rows = []
    review_rows = []

    for qa in all_pairs:
        if not isinstance(qa, dict):
            continue
        q = str(qa.get("question", "")).strip()
        a = str(qa.get("answer", "")).strip()
        excerpt = str(qa.get("excerpt", "")).strip()
        if not q or not a:
            continue

        confidence = float(qa.get("confidence", 0.5))
        row = {
            "text": (
                f"<medicine>{remedy_name}</medicine>\n"
                f"<context>\n{excerpt}\n</context>\n\n"
                f"<user>\n{q}\n</user>\n\n"
                f"<assistant>\n{a}\n</assistant>"
            )
        }

        if confidence >= CONFIDENCE_CUTOFF:
            training_rows.append(row)
        else:
            review_rows.append({
                **row,
                "_meta": {
                    "remedy": remedy_id,
                    "confidence": confidence,
                    "topic": qa.get("topic", ""),
                },
            })

    training_path = output_dir / TRAINING_FILE
    review_path = output_dir / REVIEW_FILE

    with open(training_path, "a", encoding="utf-8") as f:
        for row in training_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(review_path, "a", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "remedy_id": remedy_id,
        "remedy_name": remedy_name,
        "total_pairs": len(all_pairs),
        "training": len(training_rows),
        "review": len(review_rows),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"  → Training: {len(training_rows)}  |  Review: {len(review_rows)}")
    return stats

# ── Main ──────────────────────────────────────────────────────────────────────

def load_keys(args) -> List[str]:
    keys = []

    if args.keys:
        keys.extend([k.strip() for k in args.keys.split(",") if k.strip()])

    if args.keys_file:
        path = Path(args.keys_file)
        if path.exists():
            for line in path.read_text().splitlines():
                k = line.strip()
                if k and not k.startswith("#"):
                    keys.append(k)

    env = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
    if env:
        keys.extend([k.strip() for k in env.split(",") if k.strip()])

    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)

    return unique


def normalise_rid(stem: str) -> str:
    """Consistent remedy ID from a filename stem — always lowercase+underscores."""
    return re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_')


def run_pipeline(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = load_keys(args)
    if not keys:
        print("ERROR: No API keys provided.")
        print("  Use --keys KEY1,KEY2  or  --keys-file keys.txt  or  GEMINI_API_KEYS env var")
        sys.exit(1)

    print(f"\n🔑 Loaded {len(keys)} API key(s)")

    pool = KeyPool(keys)
    client = GeminiClient(pool, model=args.model)
    checkpoint = Checkpoint(output_dir / CHECKPOINT_FILE)

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(Path(args.folder).glob("*.txt"))

    if not files:
        print("ERROR: No .txt files found.")
        sys.exit(1)

    todo = []
    skipped = 0
    for fp in files:
        rid = normalise_rid(fp.stem)
        rname = fp.stem.replace("-", " ").replace("_", " ").title()
        if checkpoint.is_done(rid):
            skipped += 1
        else:
            todo.append((fp, rid, rname))

    print(f"\n📋 Files: {len(files)} total | {skipped} already done | {len(todo)} to process")

    if not todo:
        print("✅ All files already processed! Nothing to do.")
        _write_summary(output_dir, checkpoint)
        return

    for idx, (fp, rid, rname) in enumerate(todo, 1):
        if _shutdown_requested:
            print("\n🛑 Shutdown requested. Progress saved. Re-run to resume.")
            break

        print(f"\n[{idx}/{len(todo)}]", end="")
        stats = process_file(fp, rid, rname, output_dir, client)

        if stats:
            checkpoint.mark_done(rid, stats)
        else:
            checkpoint.mark_failed(rid, "zero pairs extracted")
            print(f"  ⚠  Marked as failed — will retry on next run")

        if idx < len(todo) and not _shutdown_requested:
            time.sleep(BATCH_DELAY)

    _write_summary(output_dir, checkpoint)
    print(f"\n✅ Done. Total API calls: {client.call_count}")
    print(f"   Training data : {output_dir / TRAINING_FILE}")
    print(f"   Review data   : {output_dir / REVIEW_FILE}")


def _write_summary(output_dir: Path, checkpoint: Checkpoint):
    stats_list = list(checkpoint.data.get("stats", {}).values())
    summary = {
        "completed": len(checkpoint.data.get("completed", [])),
        "failed": len(checkpoint.data.get("failed", [])),
        "total_training_pairs": sum(s.get("training", 0) for s in stats_list),
        "total_review_pairs":   sum(s.get("review", 0) for s in stats_list),
        "generated_at": datetime.now().isoformat(),
        "per_remedy": stats_list,
        "failed_list": checkpoint.data.get("failed", []),
    }
    path = output_dir / SUMMARY_FILE
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n📊 Summary written to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Medical Q&A Pipeline — robust, resumable, multi-key"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file",   help="Single .txt file")
    src.add_argument("--folder", help="Folder of .txt files")

    parser.add_argument("--output-dir", default="./outputs",    help="Output directory (default: ./outputs)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,  help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--keys",       help="Comma-separated API keys")
    parser.add_argument("--keys-file",  help="Path to file with one API key per line")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()