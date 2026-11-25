"""Report unparsable lines and token-length statistics for jsonl training files.

    python tools/check_jsonl.py groq_train.jsonl training.jsonl [--tokenizer bharatgenai/Param-1-2.9B-Instruct]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hft.data import normalize_row, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", help="optional: report token-length percentiles")
    args = ap.parse_args()

    texts = []
    for f in args.files:
        good, bad = read_jsonl(f)
        valid = [r for r in good if normalize_row(r)]
        print(f"{f}: {len(good)} parsed, {len(bad)} bad json, {len(good) - len(valid)} missing tags")
        for line_no, err in bad[:5]:
            print(f"  line {line_no}: {err}")
        texts += [r["text"] for r in valid]

    if args.tokenizer and texts:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        n = sorted(len(tok(t)["input_ids"]) for t in texts)
        pct = lambda p: n[min(len(n) - 1, int(p * len(n)))]
        print(f"tokens: max={n[-1]} mean={sum(n)//len(n)} p95={pct(.95)} p99={pct(.99)}")


if __name__ == "__main__":
    main()
