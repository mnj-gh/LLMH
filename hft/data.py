"""Data loading, validation and tokenisation (pure Python apart from the collator)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import SYSTEM_PROMPT


def read_jsonl(path: str | Path) -> tuple[list[dict], list[tuple[int, str]]]:
    """Return (good_rows, [(line_number, error), ...]). Blank lines are ignored."""
    good, bad = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad.append((i, str(e)))
    return good, bad


def extract_tag(text: str, tag: str) -> str | None:
    """Text between <tag> and </tag>, or None when the tag is missing."""
    try:
        return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()
    except IndexError:
        return None


def _squash(s: str) -> str:
    return re.sub(r"\n{2,}", "\n", s).strip()


def normalize_row(row: dict) -> dict | None:
    """Parse a tagged Q&A row into medicine / context / user / assistant fields.

    Returns None when context, user or assistant parts are missing.
    """
    text = row.get("text", "")
    medicine = extract_tag(text, "medicine")
    context = extract_tag(text, "context")
    user = extract_tag(text, "user")
    assistant = extract_tag(text, "actual response") or extract_tag(text, "assistant")
    if not (context and user and assistant):
        return None

    medicine = _squash(medicine) if medicine else ""
    context, user, assistant = _squash(context), _squash(user), _squash(assistant)
    user_content = (f"{medicine}\n" if medicine else "") + f"<context>\n{context}\n</context>\n\n{user}"
    return {"medicine": medicine, "context": context, "user": user,
            "assistant": assistant, "user_content": user_content}


def load_qa_files(paths: Iterable[str | Path]) -> list[dict]:
    """Load, validate and normalise one or more Q&A jsonl files."""
    rows = []
    for p in paths:
        good, bad = read_jsonl(p)
        kept = [r for r in map(normalize_row, good) if r]
        print(f"{p}: {len(kept)} kept, {len(good) - len(kept) + len(bad)} dropped")
        rows.extend(kept)
    return rows


def load_dialogues(path: str | Path) -> list[dict]:
    """Load multi-turn dialogues: {"messages": [{"role", "content"}, ...]} with >= 3 messages."""
    good, bad = read_jsonl(path)
    rows = [r for r in good if len(r.get("messages", [])) >= 3]
    print(f"{path}: {len(rows)} kept, {len(good) - len(rows) + len(bad)} dropped")
    return rows


def qa_conversation(row: dict) -> list[dict]:
    """Chat messages for one normalised Q&A row (answer wrapped in the special tokens)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["user_content"]},
        {"role": "assistant", "content": f"<actual response>\n{row['assistant']}\n</actual response>"},
    ]


def tokenize_qa(row: dict, tokenizer, max_len: int) -> dict:
    """Tokenise a Q&A row; only the assistant answer contributes to the loss."""
    conv = qa_conversation(row)
    full = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(conv[:-1], tokenize=False, add_generation_prompt=True)
    full_enc = tokenizer(full, truncation=True, max_length=max_len, padding=False)
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len, padding=False)["input_ids"])
    ids = full_enc["input_ids"]
    labels = list(ids)
    labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
    return {"input_ids": ids, "attention_mask": full_enc["attention_mask"], "labels": labels}


def tokenize_dialogue(example: dict, tokenizer, max_len: int) -> dict:
    """Tokenise a multi-turn dialogue; every assistant turn is trained, everything else masked."""
    messages = example["messages"]
    ids = tokenizer(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
        truncation=True, max_length=max_len, padding=False,
    )["input_ids"]
    labels = [-100] * len(ids)

    def n_tokens(msgs, generation_prompt):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation_prompt)
        return len(tokenizer(text, truncation=True, max_length=max_len, padding=False)["input_ids"])

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        start = n_tokens(messages[:i], True)       # up to and including the assistant cue
        end = n_tokens(messages[: i + 1], False)   # through the end of this reply
        for pos in range(start, min(end, len(ids))):
            labels[pos] = ids[pos]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


@dataclass
class PadCollator:
    """Pads input_ids / attention_mask / labels to the longest sequence in the batch."""
    tokenizer: Any

    def __call__(self, features):
        import torch  # imported lazily so the data helpers work without torch

        longest = max(len(f["input_ids"]) for f in features)

        def pad(key, value):
            return torch.tensor([f[key] + [value] * (longest - len(f[key])) for f in features])

        return {"input_ids": pad("input_ids", self.tokenizer.pad_token_id),
                "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", -100)}
