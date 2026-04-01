import torch
from embed import embed
from retrieve import retrieve
from quantize import quantize, dequantize
from model import MiniGPT

# Simple character-level tokenizer
def encode(text):
    return [ord(c) for c in text if ord(c) < 256]

def decode(tokens):
    return ''.join(chr(t) for t in tokens)

# Model parameters
vocab_size = 256  # ASCII
n_embd = 64
n_head = 4
n_layer = 2
block_size = 128

# load docs
with open("../data/docs.txt") as f:
    docs = [line.strip() for line in f]

# embed docs
doc_vecs = [embed(d) for d in docs]

# quantize embeddings
quantized = [quantize(v) for v in doc_vecs]

# query
query = "How are you?"
q_vec = embed(query)

# retrieve (using original for now)
result = retrieve(q_vec, doc_vecs, docs)

print("Retrieved:", result)

# Initialize model
model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size)

# Load model if exists
import os
if os.path.exists('../model.pth'):
    model.load_state_dict(torch.load('../model.pth'))
    print("Loaded pre-trained model.")
    train_model = False
else:
    print("No pre-trained model found, will train.")
    train_model = True

# Load training data
with open("../data/plots.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% train
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split, batch_size=4):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y

if train_model:
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_steps = 1000  # Small training for demo

    print("Training model...")
    model.train()
    for step in range(num_steps):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    print("Training complete!")

    # Save the trained model
    torch.save(model.state_dict(), '../model.pth')
    print("Model saved to ../model.pth")

# Prepare context: query + retrieved doc
context_text = query + " " + result
context_tokens = encode(context_text)
context_tensor = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0)  # (1, T)

print("Context:", context_text)

# Generate text
generated_tokens = model.generate(context_tensor, max_new_tokens=50)
generated_text = decode(generated_tokens[0].tolist())

print("Generated:", generated_text)