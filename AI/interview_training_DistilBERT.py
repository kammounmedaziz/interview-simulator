import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import math

# ===== Load Data =====
df = pd.read_csv("../data/Data.csv")

# Input: candidate + role + call pitch
texts = (
    df["Name"].astype(str) + " | " +
    df["Role acceptance"].astype(str) + " | " +
    df["Call-pitch Elements used during the call Sales Scenario"].astype(str)
)

# Target: Whether joined
labels = df["Whether joined the company or not\n"].astype(str)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# ===== Tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

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
        enc = self.tokenizer(
            self.texts.iloc[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label

# ===== Dataset & Split =====
dataset = InterviewDataset(texts, labels_encoded, tokenizer)
train_size = int(0.2 * len(dataset))  # use 20% of data per epoch
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Model =====
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
).to(device)

# ===== Loss & Optimizer =====
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# ===== Training =====
best_val_acc = 0
for epoch in range(10):  # 10 epochs
    model.train()
    total_loss, correct, total = 0, 0, 0

    for input_ids, attention_mask, labels_batch in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    # ===== Validation =====
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels_batch in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            val_correct += (preds == labels_batch).sum().item()
            val_total += labels_batch.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("saved_model_classifier")
        tokenizer.save_pretrained("saved_model_classifier")

print("✅ Training finished and best model saved in 'saved_model_classifier/'")
