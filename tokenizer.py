from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
import os

# Path to your preprocessed corpus
corpus_path = "homeo_corpus.txt"

# Output tokenizer file
tokenizer_output = "homeo_tokenizer.json"

# Optional: vocab size (should match your domain, 25k is good for small GPTs)
VOCAB_SIZE = 25600

# === Load text ===
with open(corpus_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# === Initialize tokenizer ===
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

tokenizer.normalizer = normalizers.Sequence([NFKC()])
tokenizer.pre_tokenizer = Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
)

tokenizer.train_from_iterator(lines, trainer=trainer)

# Save it
tokenizer.save(tokenizer_output)
print(f"✅ Tokenizer saved to {tokenizer_output}")
