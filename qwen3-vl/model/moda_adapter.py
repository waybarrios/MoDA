"""
MoDA (Modality-aware Dynamic Attention) Adapter for Qwen3-VL.

Adapted from LLaVA-MORE multimodal_masking/builder.py.
Uses cross-attention to generate a soft modulation mask over visual tokens,
conditioned on language context.

This implementation uses standard nn.Linear layers instead of
nn.TransformerDecoderLayer/nn.MultiheadAttention to ensure full
compatibility with DeepSpeed ZeRO-3 parameter partitioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDADecoderLayer(nn.Module):
    """
    Custom Transformer decoder layer using nn.Linear for all projections.
    Functionally equivalent to nn.TransformerDecoderLayer but ZeRO-3 safe.

    Contains: self-attention -> cross-attention -> FFN
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Self-attention (Q, K, V projections + output)
        self.sa_q = nn.Linear(d_model, d_model)
        self.sa_k = nn.Linear(d_model, d_model)
        self.sa_v = nn.Linear(d_model, d_model)
        self.sa_out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (Q from visual, K/V from language)
        self.ca_q = nn.Linear(d_model, d_model)
        self.ca_k = nn.Linear(d_model, d_model)
        self.ca_v = nn.Linear(d_model, d_model)
        self.ca_out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def _attention(self, q, k, v):
        """Scaled dot-product multi-head attention."""
        B, S, _ = q.shape
        T = k.shape[1]

        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return out

    def forward(self, tgt, memory):
        # Self-attention over visual tokens
        x = tgt
        sa_out = self._attention(self.sa_q(x), self.sa_k(x), self.sa_v(x))
        x = self.norm1(x + self.dropout1(self.sa_out(sa_out)))

        # Cross-attention: visual queries, language keys/values
        ca_out = self._attention(self.ca_q(x), self.ca_k(memory), self.ca_v(memory))
        x = self.norm2(x + self.dropout2(self.ca_out(ca_out)))

        # Feed-forward
        ff_out = self.ff2(F.gelu(self.ff1(x)))
        x = self.norm3(x + self.dropout3(ff_out))

        return x


class MoDAAdapter(nn.Module):
    """
    Generate a modulation mask over visual tokens using cross-attention
    with language embeddings as context.

    Formula: ẽV = V ⊙ σ(W · F(T, V))
    where F is the decoder, W is the mask_head, σ is sigmoid.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        hidden_dim: int = 512,
        num_heads: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        sparsity_lambda: float = 1e-3,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.sparsity_lambda = sparsity_lambda

        self.layers = nn.ModuleList([
            MoDADecoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.mask_head = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_mask = None

        nn.init.xavier_uniform_(self.mask_head.weight)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, N_vis, D) - projected visual features (query/tgt)
            language_tokens: (B, N_lang, D) - language embeddings (memory/key-value)

        Returns:
            Modulated visual tokens: (B, N_vis, D)
        """
        decoded = visual_tokens
        for layer in self.layers:
            decoded = layer(tgt=decoded, memory=language_tokens)

        mask = self.sigmoid(self.mask_head(decoded))
        self.last_mask = mask
        return visual_tokens * mask

    def get_sparsity_loss(self) -> torch.Tensor:
        """Sparsity regularization: encourages mask values toward 0."""
        if self.last_mask is None:
            return torch.tensor(0.0)
        loss = self.sparsity_lambda * torch.mean(torch.abs(self.last_mask))
        self.last_mask = None
        return loss
