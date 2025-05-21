import torch
import numpy as np
import torch.nn as nn

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

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, d_model]
        # Output/return: x + positional encodings of shape [1, seq_len, d_model]
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Dot Product between
        # d_model: input/output embedding size.
        # num_heads: number of attention heads.
        # Internal projections for query, key, value: Linear layers projecting d_model -> d_model
        # Final output projection: Linear(d_model, d_model)

    def forward(self, x, context=None, mask=None):
        # x: [batch_size, seq_len, d_model] (used as query matrix)
        # context: [batch_size, context_len, d_model] (used as key and value). Do self attention if None
        # mask: [batch_size, seq_len, context_len] or [seq_len, context_len], used to mask out padding or future positions.
        # Returns: Attention output of shape [batch_size, seq_len, d_model]
        pass

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Implements position-wise feed-forward network:
        # Two linear layers with ReLU in between.
        # First: Linear(d_model, d_ff)
        # Second: Linear(d_ff, d_model)

    def forward(self, x):
        # Input/Output: [batch_size, seq_len, d_model]
        # Applies FFN to each position independently.
        pass

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # One layer of Transformer encoder block:
        # - Multi-head self-attention
        # - FeedForward network
        # - LayerNorm and residual connections around both sub-layers
        # - Dropout applied after attention and feed-forward layers

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        # mask: [batch_size, seq_len, seq_len] or [seq_len, seq_len]
        # Output: encoded x of shape [batch_size, seq_len, d_model]
        pass

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # One layer of Transformer decoder block:
        # - Masked self-attention (prevents looking ahead)
        # - Encoder-decoder cross-attention
        # - FeedForward network
        # - Residual connections and LayerNorm around each sub-layer

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        # x: [batch_size, tgt_seq_len, d_model]
        # enc_out: [batch_size, src_seq_len, d_model]
        # tgt_mask: prevents attending to future tokens
        # src_mask: masks padding in encoder output
        # Output: decoded x of shape [batch_size, tgt_seq_len, d_model]
        pass

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        super().__init__()
        # Embedding layer: token embeddings of shape [vocab_size, d_model]
        # Positional encoding: added to token embeddings
        # Stack of `num_layers` TransformerEncoderBlock layers

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len] (token indices)
        # mask: optional attention mask for padding
        # Output: encoded representation of shape [batch_size, seq_len, d_model]
        pass

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        super().__init__()
        # Embedding + positional encoding for target input tokens
        # Stack of `num_layers` TransformerDecoderBlock layers
        # Final projection: Linear(d_model, vocab_size) to generate logits over vocabulary

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        # x: [batch_size, tgt_seq_len] (target token indices)
        # enc_out: [batch_size, src_seq_len, d_model] (from encoder)
        # tgt_mask: prevents peeking ahead
        # src_mask: masks encoder outputs
        # Output: logits of shape [batch_size, tgt_seq_len, vocab_size]
        pass

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        super().__init__()
        # Full Transformer model combining:
        # - Encoder (embedding + positional encoding + encoder blocks)
        # - Decoder (embedding + positional encoding + decoder blocks + final projection)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [batch_size, src_seq_len] (source token indices)
        # tgt: [batch_size, tgt_seq_len] (target token indices)
        # src_mask: optional mask for source padding
        # tgt_mask: mask for decoder input (including future positions)
        # Output: logits of shape [batch_size, tgt_seq_len, tgt_vocab_size]
        pass
