import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


df = pd.read_csv("../data/Data.csv")  
print(df.head())
print(df.columns)


# Combine relevant columns for input
df['prompt'] = (
    "Candidate Mode: " + df['Mode of interview given by candidate?'].astype(str) +
    " | Role: " + df['What was the type of Role?\t'].astype(str) +
    " | Call Pitch Elements: " + df['Call-pitch Elements used during the call Sales Scenario'].astype(str)
)

# Combine relevant columns for output/response
df['response'] = (
    "Confidence: " + df['Confidence Score'].astype(str) +
    " | Structured Thinking: " + df['Structured Thinking Score'].astype(str) +
    " | Regional Fluency: " + df['Regional Fluency Score'].astype(str) +
    " | Verdict: " + df['Interview Verdict'].astype(str)
)


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['prompt'].tolist(),
    df['response'].tolist(),
    test_size=0.1,
    random_state=42
)




# Word-level tokenizer
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"])

# Train tokenizer on both prompts and responses
tokenizer.train_from_iterator(train_texts + train_labels, trainer)
tokenizer.save("tokenizer.json")



class InterviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.texts[idx]).ids[:self.max_len]
        y = self.tokenizer.encode(self.labels[idx]).ids[:self.max_len]
        # pad sequences
        x += [0]*(self.max_len - len(x))
        y += [0]*(self.max_len - len(y))
        return torch.tensor(x), torch.tensor(y)

train_dataset = InterviewDataset(train_texts, train_labels, tokenizer)
val_dataset = InterviewDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)



class SimpleSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, y):
        x = self.embedding(x)
        _, (h, c) = self.encoder(x)
        y = self.embedding(y)
        out, _ = self.decoder(y, (h, c))
        return self.fc(out)

vocab_size = len(tokenizer.get_vocab())
model = SimpleSeq2Seq(vocab_size)



device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):  # adjust number of epochs
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, y)
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader)}")


def generate_response(model, prompt, tokenizer, max_len=50):
    model.eval()
    x = tokenizer.encode(prompt).ids
    x = torch.tensor(x).unsqueeze(0).to(device)
    generated = x
    for _ in range(max_len):
        out = model(generated, generated)
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return tokenizer.decode(generated[0].cpu().numpy())

# Test
prompt = "Candidate Mode: Online | Role: Software Engineer | Call Pitch Elements: Basic"
print(generate_response(model, prompt, tokenizer))



