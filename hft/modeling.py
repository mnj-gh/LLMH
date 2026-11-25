"""Model/tokenizer loading shared by training and chat."""
from __future__ import annotations

from .config import SPECIAL_TOKENS


def load_model_and_tokenizer(model_id: str, device_map="auto"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # Param-1 ships a rope_scaling dict that older transformers versions reject.
    rope = getattr(config, "rope_scaling", None)
    if rope is not None:
        rope.setdefault("type", "linear")
        rope.setdefault("factor", 1.0)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, trust_remote_code=True,
        device_map=device_map, torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def add_special_tokens(model, tokenizer) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
