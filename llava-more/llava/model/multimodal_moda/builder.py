"""MoDA: Modulation Adapter for fine-grained visual grounding (arXiv:2506.01850).

MoDA cross-attends from pre-aligned visual features (queries) to the
instruction token embeddings (memory) and produces a channel-wise sigmoid
mask that modulates the visual features multiplicatively:

    V_out = V * sigmoid(W . F(T, V))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDAAdapter(nn.Module):
    """Instruction-guided channel-wise modulation of pre-aligned visual features."""

    def __init__(self, embedding_dim=4096, hidden_dim=1024, num_heads=16,
                 num_layers=1, l1_lambda=1e-5, sparsity_lambda=1e-3):
        super().__init__()

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.mask_head = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.l1_lambda = l1_lambda
        self.sparsity_lambda = sparsity_lambda
        self.loss_value = 0
        self.last_mask = None
        nn.init.xavier_uniform_(self.mask_head.weight)

    def forward(self, x, input_embeds, labels=None):
        """
        x: Tensor (B, N, D) - pre-aligned visual features (query)
        input_embeds: Tensor (B, M, D) - instruction token embeddings (key/value)
        """
        decoded = self.decoder(tgt=x, memory=input_embeds)  # [B, N, D]
        mask = self.sigmoid(self.mask_head(decoded))        # [B, N, D], values in (0, 1)
        self.last_mask = mask
        return x * mask

    def l1_loss(self):
        """L1 regularization on the projection head (disabled by default)."""
        return self.loss_value

    def get_sparsity_loss(self):
        """Mean absolute mask value; encourages sparse modulation masks."""
        if self.last_mask is None:
            return 0.0
        loss = torch.mean(torch.abs(self.last_mask))
        self.last_mask = None
        return loss


def build_moda(embedding_dim=4096, hidden_dim=1024, num_heads=16,
               num_layers=1, l1_lambda=1e-5, sparsity_lambda=1e-3):
    return MoDAAdapter(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        l1_lambda=l1_lambda,
        sparsity_lambda=sparsity_lambda,
    )


# Backward-compatible aliases: earlier research code used "masking" naming.
EncoderMask = MoDAAdapter
build_vision_masking = build_moda
