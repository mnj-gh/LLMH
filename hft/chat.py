"""Interactive consultation with a trained HFT model (decision support only).

    python -m hft.chat --model mnj-hf/homeopathy-chat
"""
import argparse

from .config import CHAT_MODEL_ID, SYSTEM_PROMPT
from .modeling import load_model_and_tokenizer

DISCLAIMER = "HFT is a decision-support and educational tool. It does not replace clinical judgment."


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=CHAT_MODEL_ID, help="Hub id or local path")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(DISCLAIMER, "\nType 'quit' to exit, 'reset' to start a new case.\n")

    while True:
        user = input("Practitioner> ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        if user.lower() == "reset":
            messages = messages[:1]
            continue
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=args.temperature, pad_token_id=tokenizer.pad_token_id)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\nHFT> {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
