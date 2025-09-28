from tokenizers import Tokenizer
import torch
import torch.nn as nn

# ===== Model Class (same as training) =====
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

# ===== Load Tokenizer & Model =====
tokenizer = Tokenizer.from_file("tokenizer.json")
vocab_size = len(tokenizer.get_vocab())
model = SimpleSeq2Seq(vocab_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load("interview_model.pt", map_location=device))
model.to(device)
model.eval()

# ===== Response Generator =====
def generate_response(model, prompt, tokenizer, max_len=50):
    x = tokenizer.encode(prompt).ids
    x = torch.tensor(x).unsqueeze(0).to(device)
    generated = x
    for _ in range(max_len):
        out = model(generated, generated)
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return tokenizer.decode(generated[0].cpu().numpy())

# ===== Test =====
prompt = "Candidate Mode: Online | Role: Software Engineer | Call Pitch Elements: Basic"
print("Generated Response:", generate_response(model, prompt, tokenizer))
