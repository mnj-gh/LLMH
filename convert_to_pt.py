import torch

# Read the token IDs
with open("homeo_corpus_ids.txt", "r", encoding="utf-8") as f:
    token_ids = list(map(int, f.read().strip().split()))

# Convert to a PyTorch tensor
tensor = torch.tensor(token_ids, dtype=torch.long)

# Save it
torch.save(tensor, "homeo_corpus.pt")

print(f"✅ Saved tensor to homeo_corpus.pt with shape: {tensor.shape}")
