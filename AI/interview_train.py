import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math

# ===== Load Data =====
df = pd.read_csv("../data/Data.csv")

# Convert columns into lists
texts = (
    df["Name"].astype(str) + " | " +
    df["Role acceptance"].astype(str) + " | " +
    df["Call-pitch Elements used during the call Sales Scenario"].astype(str)
).tolist()
labels = df["Whether joined the company or not\n"].astype(str).tolist()

# ===== Tokenizer =====
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ===== Dataset Class =====
class InterviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = f"{self.texts[idx]} -> {self.labels[idx]}"
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        return encoding["input_ids"].squeeze(0), encoding["input_ids"].squeeze(0)

# ===== Full Dataset =====
full_dataset = InterviewDataset(texts, labels, tokenizer)

# ===== Model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ===== Training Loop (10 epochs, each on 20% slice) =====
n = len(full_dataset)
chunk_size = n // 5  # 20% = 1/5 of dataset

for epoch in range(10):
    # Cycle through 5 chunks (20% each)
    chunk_id = epoch % 5
    start = chunk_id * chunk_size
    end = (chunk_id + 1) * chunk_size if chunk_id < 4 else n
    subset = Subset(full_dataset, list(range(start, end)))
    train_loader = DataLoader(subset, batch_size=8, shuffle=True)

    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, labels=y).logits
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = out.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    perplexity = math.exp(avg_loss)

    print(
        f"Epoch {epoch+1}/10 "
        f"(Chunk {chunk_id+1}/5 | Data {start}:{end}) "
        f"| Loss={avg_loss:.4f} | Acc={accuracy:.4f} | PPL={perplexity:.2f}"
    )

# ===== Save Model =====
model.save_pretrained("saved_model_full")
tokenizer.save_pretrained("saved_model_full")
print("✅ Model trained across 20% slices (10 epochs) and saved in 'saved_model_full/'")
