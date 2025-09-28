import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import math
import os
import random

# ===== Load Data =====
df = pd.read_csv("../data/Data.csv")

# Combine relevant columns
texts = (
    df["Name"].astype(str) + " | " +
    df["Role acceptance"].astype(str) + " | " +
    df["Call-pitch Elements used during the call Sales Scenario"].astype(str)
)
labels = (df["Whether joined the company or not\n"].str.strip() == "Yes").astype(int)  # 1=Joined, 0=Not Joined

# ===== Tokenizer =====
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

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
        encoding = self.tokenizer(
            str(self.texts.iloc[idx]),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }

dataset = InterviewDataset(texts, labels, tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Model =====
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# ===== Training =====
epochs = 10
batch_size = 8
subset_fraction = 0.2  # 20% per epoch

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # Sample 20% of dataset each epoch
    indices = random.sample(range(len(dataset)), int(len(dataset) * subset_fraction))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

# ===== Save Model =====
os.makedirs("saved_model_cls", exist_ok=True)
model.save_pretrained("saved_model_cls")
tokenizer.save_pretrained("saved_model_cls")
print("✅ Model trained and saved in 'saved_model_cls/'")

# ===== Test Prediction Example =====
model.eval()
example_text = "John | Accepted Role | Basic Pitch Elements"
encoding = tokenizer(example_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    pred = torch.argmax(outputs.logits, dim=1)
    print("Predicted: Joined" if pred.item() == 1 else "Predicted: Not Joined")
