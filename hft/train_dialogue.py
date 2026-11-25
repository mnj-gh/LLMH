"""Stage 2: continue from the stage-1 model on multi-turn consultation dialogues.

    python -m hft.train_dialogue --data final_multiturn_dialogues.jsonl \
        [--init-model mnj-hf/homeopathy] [--hub-id mnj-hf/homeopathy-chat] [--no-push]
"""
import argparse

from datasets import Dataset
from transformers import Trainer, TrainingArguments

from .config import CHAT_MODEL_ID, QA_MODEL_ID, STAGE2
from .data import PadCollator, load_dialogues, tokenize_dialogue
from .modeling import add_special_tokens, load_model_and_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Dialogue jsonl ({'messages': [...]})")
    ap.add_argument("--init-model", default=QA_MODEL_ID, help="Stage-1 model to continue from")
    ap.add_argument("--hub-id", default=CHAT_MODEL_ID, help="Push to a new repo so the stage-1 model stays intact")
    ap.add_argument("--output-dir", default="./hft_chat")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    cfg = STAGE2

    model, tokenizer = load_model_and_tokenizer(args.init_model, device_map={"": 0})
    add_special_tokens(model, tokenizer)  # no-op if already present, keeps vocab consistent

    rows = load_dialogues(args.data)
    dataset = Dataset.from_list(rows).map(
        lambda r: tokenize_dialogue(r, tokenizer, cfg["max_len"]),
        remove_columns=list(rows[0].keys()),
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["warmup_steps"],
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        dataloader_num_workers=2,
        report_to="none",
        push_to_hub=not args.no_push,
        hub_model_id=None if args.no_push else args.hub_id,
        hub_strategy="end",
        hub_private_repo=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=PadCollator(tokenizer))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if not args.no_push:
        trainer.push_to_hub(commit_message="HFT stage 2: multi-turn consultation fine-tune")
        tokenizer.push_to_hub(args.hub_id)


if __name__ == "__main__":
    main()
