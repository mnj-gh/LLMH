"""Stage 1: full fine-tune of Param-1-2.9B-Instruct on the Q&A pairs.

    python -m hft.train_qa --data groq_train.jsonl training.jsonl gemini_train.jsonl \
        [--hub-id mnj-hf/homeopathy] [--no-push]
"""
import argparse

from datasets import Dataset
from transformers import Trainer, TrainingArguments

from .config import BASE_MODEL_ID, QA_MODEL_ID, STAGE1
from .data import PadCollator, load_qa_files, tokenize_qa
from .modeling import add_special_tokens, load_model_and_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="Q&A jsonl files")
    ap.add_argument("--base-model", default=BASE_MODEL_ID)
    ap.add_argument("--hub-id", default=QA_MODEL_ID)
    ap.add_argument("--output-dir", default="./hft_qa")
    ap.add_argument("--no-push", action="store_true", help="Train locally without pushing to the Hub")
    args = ap.parse_args()
    cfg = STAGE1

    model, tokenizer = load_model_and_tokenizer(args.base_model)
    add_special_tokens(model, tokenizer)

    rows = load_qa_files(args.data)
    dataset = Dataset.from_list(rows).map(
        lambda r: tokenize_qa(r, tokenizer, cfg["max_len"]),
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
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        logging_steps=50,
        dataloader_num_workers=4,
        report_to="none",
        push_to_hub=not args.no_push,
        hub_model_id=None if args.no_push else args.hub_id,
        hub_strategy="checkpoint",
        hub_private_repo=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=PadCollator(tokenizer))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if not args.no_push:
        trainer.push_to_hub(commit_message="HFT stage 1: Q&A fine-tune")
        tokenizer.push_to_hub(args.hub_id)


if __name__ == "__main__":
    main()
