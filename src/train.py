import torch
import torch.nn as nn
import torch.optim as optim
from model import Seq2SeqTransformer

# Parameters (Change before we do training)
src_vocab_size = 10000
tgt_vocab_size = 10000
seq_len = 20
batch_size = 32
d_model = 512
num_heads = 8 #how many diff qkv do we make?
d_ff = 2048 #expansion dimension for MLP
num_layers = 6
lr = 1e-4 # learning rate
n_epochs = 5 # How many times do we show our data to the model?

# Replace later with datasets
src = None
tgt_in = None
tgt_out = None

model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    logits = model(src, tgt_in)
    loss = criterion(logits.view(-1, tgt_vocab_size), tgt_out.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")


# After one training, it will be saved here. We can do stuff for
# more trainings later
torch.save(model.state_dict(), '../models/translation_model.pth')

