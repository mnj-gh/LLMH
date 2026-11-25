import json

from hft.data import (extract_tag, load_dialogues, normalize_row, read_jsonl,
                      tokenize_dialogue, tokenize_qa)


class FakeTokenizer:
    """Whitespace tokenizer with a trivial chat template, enough to test label masking."""
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}> {m['content']} </{m['role']}> " for m in msgs)
        return s + ("<assistant> " if add_generation_prompt else "")

    def __call__(self, text, truncation=True, max_length=None, padding=False):
        ids = list(range(1, len(text.split()) + 1))[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


ROW = {"text": "<medicine> Aconite </medicine> <context> fear, sudden onset </context> "
               "<user> When is Aconite indicated? </user> <actual response> After cold dry wind. </actual response>"}


def test_extract_tag():
    assert extract_tag("<a> x </a>", "a") == "x"
    assert extract_tag("no tags", "a") is None


def test_normalize_row_ok_and_missing():
    n = normalize_row(ROW)
    assert n["medicine"] == "Aconite" and n["assistant"] == "After cold dry wind."
    assert n["user_content"].startswith("Aconite\n<context>")
    assert normalize_row({"text": "<user> hi </user>"}) is None


def test_tokenize_qa_masks_prompt_only():
    n = normalize_row(ROW)
    out = tokenize_qa(n, FakeTokenizer(), max_len=512)
    labels = out["labels"]
    assert labels[0] == -100
    assert labels[-1] != -100
    assert any(l == -100 for l in labels) and any(l != -100 for l in labels)


def test_tokenize_dialogue_trains_every_assistant_turn():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"}]
    out = tokenize_dialogue({"messages": msgs}, FakeTokenizer(), 2048)
    trained = [i for i, l in enumerate(out["labels"]) if l != -100]
    assert trained and trained[0] > 0
    assert out["labels"][0] == -100


def test_jsonl_helpers(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"messages": [1, 2, 3]}) + "\n{bad\n\n" + json.dumps({"messages": [1]}) + "\n")
    good, bad = read_jsonl(p)
    assert len(good) == 2 and len(bad) == 1
    assert len(load_dialogues(p)) == 1
