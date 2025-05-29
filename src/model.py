import torch
import numpy as np
import torch.nn as nn
import math

"""
In the self initialization of each class, we just need to transform the input 
the forward function can use the math or logic

in TransformerDecoder and TransformerEncoder, we will just use the previous classes there
to do our computations

in Seq2Seq we will use TransformerEncoder and TransformerDecoder
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Add sinusoidal positional encodings
        # Shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, d_model]
        # Output/return: x + positional encodings of shape [1, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Dot Product between
        # d_model: input/output embedding size.
        # num_heads: number of attention heads.
        # Internal projections for query, key, value: Linear layers projecting d_model -> d_model
        # Final output projection: Linear(d_model, d_model)
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
        B, T, _ = x.size()
        B2, S, _ = context.size()

        assert B == B2, f"Batch size mismatch: {B} vs {B2}"

        Q = self.q_proj(x)
        K = self.k_proj(context)
        V = self.v_proj(context)
        Q = Q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        K = K.view(B, S, self.num_heads, self.d_head).transpose(1, 2)  # (B, nh, S, dh)
        V = V.view(B, S, self.num_heads, self.d_head).transpose(1, 2)  # (B, nh, S, dh)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, T, S)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, T, S)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, -1)  # (B, T, d_model)
        return self.out_proj(out)



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        # Implements position-wise feed-forward network:
        # Two linear layers with ReLU in between.
        # First: Linear(d_model, d_ff)
        # Second: Linear(d_ff, d_model)

    def forward(self, x):
        # Input/Output: [batch_size, seq_len, d_model]
        # Applies FFN to each position independently.
        return self.net(x)
        

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # One layer of Transformer encoder block:
        # - Multi-head self-attention
        # - FeedForward network
        # - LayerNorm and residual connections around both sub-layers
        # - Dropout applied after attention and feed-forward layers

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attn(x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x
        # x: [batch_size, seq_len, d_model]
        # mask: [batch_size, seq_len, seq_len] or [seq_len, seq_len]
        # Output: encoded x of shape [batch_size, seq_len, d_model]


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # One layer of Transformer decoder block:
        # - Masked self-attention (prevents looking ahead)
        # - Encoder-decoder cross-attention
        # - FeedForward network
        # - Residual connections and LayerNorm around each sub-layer

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x
        # x: [batch_size, tgt_seq_len, d_model]
        # enc_out: [batch_size, src_seq_len, d_model]
        # tgt_mask: prevents attending to future tokens
        # src_mask: masks padding in encoder output
        # Output: decoded x of shape [batch_size, tgt_seq_len, d_model]


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # Embedding layer: token embeddings of shape [vocab_size, d_model]
        # Positional encoding: added to token embeddings
        # Stack of `num_layers` TransformerEncoderBlock layers

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
        # x: [batch_size, seq_len] (token indices)
        # mask: optional attention mask for padding
        # Output: encoded representation of shape [batch_size, seq_len, d_model]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)


        # Embedding + positional encoding for target input tokens
        # Stack of `num_layers` TransformerDecoderBlock layers
        # Final projection: Linear(d_model, vocab_size) to generate logits over vocabulary

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.fc_out(x)
        # x: [batch_size, tgt_seq_len] (target token indices)
        # enc_out: [batch_size, src_seq_len, d_model] (from encoder)
        # tgt_mask: prevents peeking ahead
        # src_mask: masks encoder outputs
        # Output: logits of shape [batch_size, tgt_seq_len, vocab_size]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.3):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)

        # Full Transformer model combining:
        # - Encoder (embedding + positional encoding + encoder blocks)
        # - Decoder (embedding + positional encoding + decoder blocks + final projection)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, tgt_mask, src_mask)
        # src: [batch_size, src_seq_len] (source token indices)
        # tgt: [batch_size, tgt_seq_len] (target token indices)
        # src_mask: optional mask for source padding
        # tgt_mask: mask for decoder input (including future positions)
        # Output: logits of shape [batch_size, tgt_seq_len, tgt_vocab_size]

