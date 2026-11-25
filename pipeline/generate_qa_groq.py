#!/usr/bin/env python3
"""
Medical Q&A Pipeline - Groq Version (Folder + File Support)
"""

import argparse
import os
import re
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from groq import Groq


# ==============================
# CONFIG
# ==============================

CONFIDENCE_THRESHOLD = 0.6
MAX_RETRIES = 3
API_DELAY = 3
BATCH_DELAY = 7


# ==============================
# GROQ CLIENT
# ==============================

class GroqClient:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.call_count = 0

    def call_api(self, prompt: str,
                 temperature: float = 0.7,
                 max_tokens: int = 8192) -> Optional[str]:

        for attempt in range(MAX_RETRIES):
            self.call_count += 1
            print(f"  API call #{self.call_count} (attempt {attempt+1})...", end=" ")

            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=False
                )

                text = completion.choices[0].message.content

                if not text:
                    raise ValueError("Empty response")

                print(f"OK ({len(text)} chars)")
                return text

            except Exception as e:
                print(f"Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5 + attempt * 3)
                else:
                    return None

        return None

    def extract_json(self, text: str) -> Optional[dict]:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text).strip()

        start = text.find('{')
        end = text.rfind('}')

        if start == -1 or end == -1:
            return None

        json_text = text[start:end+1]

        try:
            return json.loads(json_text)
        except:
            pattern = r'\{[^{}]*?"question"[^{}]*?"answer"[^{}]*?"excerpt"[^{}]*?"topic"[^{}]*?"confidence"[^{}]*?\}'
            matches = re.findall(pattern, json_text, re.DOTALL)

            salvaged = []
            for m in matches:
                try:
                    obj = json.loads(m)
                    salvaged.append(obj)
                except:
                    continue

            if salvaged:
                return {"qa_pairs": salvaged}

        return None


# ==============================
# QA EXTRACTOR
# ==============================

class QAExtractor:

    def __init__(self, client: GroqClient):
        self.client = client

    def extract_qa_pairs(self, text: str, remedy_name: str) -> List[dict]:

        if len(text) > 20000:
            text = text[:20000]

        prompt = f"""
Extract Q&A pairs from this homeopathic remedy text.

REMEDY: {remedy_name}

TEXT:
{text}

Return STRICT JSON:

{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "...",
      "excerpt": "...",
      "topic": "...",
      "confidence": 0.9
    }}
  ]
}}
"""

        response = self.client.call_api(prompt)

        if not response:
            return []

        data = self.client.extract_json(response)
        if not data:
            return []

        return data.get("qa_pairs", [])


# ==============================
# FILE PROCESSING
# ==============================

def process_file(filepath: Path, remedy_id: str,
                 remedy_name: str, output_dir: Path,
                 extractor: QAExtractor):

    print("\n" + "="*60)
    print(f"Processing: {remedy_name}")
    print("="*60)

    text = filepath.read_text(encoding="utf-8")
    text = re.sub(r'\s+', ' ', text).strip()

    qa_pairs = extractor.extract_qa_pairs(text, remedy_name)

    if not qa_pairs:
        print("No Q&A extracted.")
        return

    triplets = []

    for qa in qa_pairs:
        if qa.get("confidence", 0.5) >= CONFIDENCE_THRESHOLD:
            triplets.append({
                "text":
                    f"<context>\n{qa.get('excerpt','')}\n</context>\n"
                    f"<user>\n{qa['question']}\n</user>\n"
                    f"<assistant>\n<actual response>\n"
                    f"{qa['answer']}\n"
                    f"</actual response>\n</assistant>"
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{remedy_id}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Saved {len(triplets)} triplets → {output_file}")


# ==============================
# FOLDER PROCESSING
# ==============================

def process_folder(folder_path: Path,
                   output_dir: Path,
                   extractor: QAExtractor):

    txt_files = sorted(folder_path.glob("*.txt"))

    if not txt_files:
        print("No .txt files found.")
        return

    print(f"\nFound {len(txt_files)} files.\n")

    for idx, filepath in enumerate(txt_files, 1):

        remedy_id = filepath.stem.replace("-", "_").lower()
        remedy_name = filepath.stem.replace("-", " ").title()

        process_file(filepath, remedy_id, remedy_name,
                     output_dir, extractor)

        if idx < len(txt_files):
            print(f"Waiting {BATCH_DELAY}s...")
            time.sleep(BATCH_DELAY)


# ==============================
# MAIN
# ==============================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Single file")
    parser.add_argument("--folder", help="Folder with txt files")
    parser.add_argument("--remedy-id")
    parser.add_argument("--remedy-name")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--model", default="openai/gpt-oss-120b")

    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY")
        return

    client = GroqClient(api_key, args.model)
    extractor = QAExtractor(client)
    output_dir = Path(args.output_dir)

    if args.folder:
        process_folder(Path(args.folder), output_dir, extractor)

    elif args.file:
        if not args.remedy_id or not args.remedy_name:
            print("Provide --remedy-id and --remedy-name")
            return

        process_file(Path(args.file),
                     args.remedy_id,
                     args.remedy_name,
                     output_dir,
                     extractor)

    else:
        print("Provide --file or --folder")

    print(f"\nTotal API calls: {client.call_count}")


if __name__ == "__main__":
    main()
