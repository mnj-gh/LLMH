"""Shared constants for both training stages."""

BASE_MODEL_ID = "bharatgenai/Param-1-2.9B-Instruct"
QA_MODEL_ID = "mnj-hf/homeopathy"          # stage 1 output (knowledge)
CHAT_MODEL_ID = "mnj-hf/homeopathy-chat"   # stage 2 output (conversation)

# Tokens that wrap the assistant answer in stage 1.
SPECIAL_TOKENS = ["<actual response>", "</actual response>"]

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for qualified homeopathic medical practitioners.\n"
    "Your role is to help practitioners think through cases, summarize repertory, compare remedy options,\n"
    "and highlight red-flag symptoms that require conventional medical evaluation.\n"
    "Base your reasoning on classical and widely accepted homeopathic sources and general medical knowledge."
)

# Stage 1: ~100k Q&A pairs, single turn.
STAGE1 = dict(max_len=512, epochs=3, batch_size=8, grad_accum=4, lr=2e-5, warmup_steps=440, save_steps=1000)

# Stage 2: ~200 synthetic multi-turn consultations. Lower LR because it continues stage 1.
STAGE2 = dict(max_len=2048, epochs=2, batch_size=4, grad_accum=8, lr=5e-6, warmup_steps=20)
