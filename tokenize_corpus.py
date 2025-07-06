from tokenizers import Tokenizer
import os

# === Load the trained tokenizer ===
tokenizer_path = "homeo_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

# === Path to cleaned corpus ===
input_path = "homeo_corpus.txt"

# === Output path for token ID sequence ===
output_path = "homeo_corpus_ids.txt"

# Collect token IDs here
token_ids = []

# === Process each line ===
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            encoding = tokenizer.encode(line)
            token_ids.extend(
                encoding.ids + [tokenizer.token_to_id("</s>")]  # add sentence-end token
            )

# === Save token ID sequence to file ===
with open(output_path, "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, token_ids)))

print(f"✅ Tokenized corpus saved to: {output_path}")
print(f"🧮 Total tokens: {len(token_ids)}")
