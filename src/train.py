import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, defaultdict
import pandas as pd
from model import Seq2SeqTransformer
import json
import time


# -------- Parameters --------
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
lr = 1e-4
n_epochs = 10
batch_size = 32
grad_clip = 1.0


# load csv
df = pd.read_csv('../datasets/data.csv')
src_sentences = df['english'].tolist()
tgt_sentences = df['spanish'].tolist()

# vocab and tokenization
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

def tokenize(sentence):
    return sentence.lower().strip().split()

class SimpleVocab:
    def __init__(self, tokens, min_freq=1, specials=SPECIAL_TOKENS):
        counter = Counter(tokens)
        self.itos = list(specials)
        self.itos += [tok for tok, freq in counter.items() if freq >= min_freq and tok not in specials]
        self.stoi = defaultdict(lambda: self.itos.index('<unk>'), {tok: idx for idx, tok in enumerate(self.itos)})

    def __getitem__(self, token):
        return self.stoi[token]

    def __len__(self):
        return len(self.itos)

src_tokens = [tok for sent in src_sentences for tok in tokenize(sent)]
tgt_tokens = [tok for sent in tgt_sentences for tok in tokenize(sent)]

src_vocab = SimpleVocab(src_tokens)
tgt_vocab = SimpleVocab(tgt_tokens)

PAD_IDX = tgt_vocab['<pad>']
SOS_IDX = tgt_vocab['<sos>']
EOS_IDX = tgt_vocab['<eos>']


# dataset
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = [self.src_vocab[token] for token in tokenize(self.src_sentences[idx])]
        tgt_tokens = [self.tgt_vocab[token] for token in tokenize(self.tgt_sentences[idx])]

        tgt_in = [SOS_IDX] + tgt_tokens
        tgt_out = tgt_tokens + [EOS_IDX]

        return torch.tensor(src_tokens), torch.tensor(tgt_in), torch.tensor(tgt_out)


# collate
def collate_fn(batch):
    src_batch, tgt_in_batch, tgt_out_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX).transpose(0, 1)
    tgt_in_batch = pad_sequence(tgt_in_batch, padding_value=PAD_IDX).transpose(0, 1)
    tgt_out_batch = pad_sequence(tgt_out_batch, padding_value=PAD_IDX).transpose(0, 1)

    return {
        "src": src_batch,
        "tgt_in": tgt_in_batch,
        "tgt_out": tgt_out_batch
    }


# load data
dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)

# Simple split 80/20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# model
model = Seq2SeqTransformer(len(src_vocab), len(tgt_vocab), d_model, num_heads, d_ff, num_layers, max_len=512)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# make target mask

def make_tgt_mask(tgt):
    # Mask future tokens in target for decoder self-attention
    seq_len = tgt.size(1)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
    return mask


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch['src'].to(device)
            tgt_in = batch['tgt_in'].to(device)
            tgt_out = batch['tgt_out'].to(device)

            tgt_mask = make_tgt_mask(tgt_in)

            output = model(src, tgt_in, tgt_mask=tgt_mask)
            output = output.reshape(-1, output.shape[-1])
            tgt_out = tgt_out.reshape(-1)

            loss = criterion(output, tgt_out)
            total_loss += loss.item()
    return total_loss / len(val_loader)


# Training
best_val_loss = float('inf')
early_stop_counter = 0
early_stop_patience = 2  # Stop if val loss doesn't improve for 2 epochs

num_batches = len(train_loader)

batch_losses = []
epoch_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_start = time.time()
    running_loss = 0
    for i, batch in enumerate(train_loader):
        src = batch['src'].to(device)
        tgt_in = batch['tgt_in'].to(device)
        tgt_out = batch['tgt_out'].to(device)

        tgt_mask = make_tgt_mask(tgt_in)

        optimizer.zero_grad()
        output = model(src, tgt_in, tgt_mask=tgt_mask)
        output = output.reshape(-1, output.shape[-1])
        tgt_out = tgt_out.reshape(-1)

        loss = criterion(output, tgt_out)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        running_loss += loss.item()

        batch_losses.append(loss.item())  # Track every batch loss

        if i % 10 == 0 or i == num_batches - 1:
            elapsed = time.time() - epoch_start
            batches_done = i + 1
            batches_left = num_batches - batches_done
            batch_time = elapsed / batches_done
            eta = batch_time * batches_left
            percent = 100 * batches_done / num_batches
            avg_loss = running_loss / batches_done

            print(f"Epoch {epoch + 1}/{n_epochs} - Batch {batches_done}/{num_batches} "
                  f"({percent:.2f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s - Avg Loss: {avg_loss:.4f}")

    avg_epoch_loss = running_loss / num_batches
    epoch_losses.append(avg_epoch_loss)  # Track loss per epoch

    val_loss = evaluate(model, val_loader)
    val_losses.append(val_loss)  # Track validation loss per epoch

    print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

print("Training complete.")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in the trained model: {total_params}")
torch.save(model.state_dict(), "../models/translation_model.pth")       #save model

# Save loss data to file for plotting or analysis later
with open('losses.json', 'w') as f:
    json.dump({
        'batch_losses': batch_losses,
        'epoch_losses': epoch_losses,
        'val_losses': val_losses
    }, f)
