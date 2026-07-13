import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderMask(nn.Module):
    """
    Generate a sparse mask over x using a Transformer decoder with input_embeds as memory.
    The decoder learns context-aware masking by attending over language or other inputs.
    """
    def __init__(self, embedding_dim=4096, hidden_dim=1024, num_heads=16, num_layers=2, l1_lambda=1e-5, sparsity_lambda=1e-3):
        super().__init__()

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.mask_head = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.l1_lambda = l1_lambda
        self.sparsity_lambda = sparsity_lambda
        self.loss_value = 0
        self.last_mask = None

        # Initialize weights
        nn.init.xavier_uniform_(self.mask_head.weight)

    def forward(self, x, input_embeds, labels=None):
        """
        x: Tensor (B, N, D) - visual or input tokens (query)
        input_embeds: Tensor (B, M, D) - memory from another modality (key/value)
        labels: Optional (B, N, D) - targets for supervision
        """
        decoded = self.decoder(tgt=x, memory=input_embeds)  # [B, N, D]

        # Learn mask via projection and activation
        mask = self.mask_head(decoded)
        mask = self.sigmoid(mask)  # [B, N, D], values in (0, 1)
        self.last_mask = mask
        masked_x = x * mask

        return masked_x

    def l1_loss(self):
        """L1 regularization on the projection head to encourage sparse mask weights."""
        return self.l1_lambda * F.l1_loss(self.mask_head.weight, torch.zeros_like(self.mask_head.weight), reduction="sum")

    def get_sparsity_loss(self):
        """Encourages mask values to be close to 0 (sparse)."""
        if self.last_mask is None:
            return 0.0
        loss = self.sparsity_lambda * torch.mean(torch.abs(self.last_mask))
        self.last_mask = None  # clean
        return loss


def build_vision_masking(embedding_dim=4096, hidden_dim=1024, num_heads=16, num_layers=2, l1_lambda=1e-5, sparsity_lambda=1e-3):
    return EncoderMask(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        l1_lambda=l1_lambda,
        sparsity_lambda=sparsity_lambda
    )

# Utility: FLOPs/MACs and parameter counter using thop
try:
    from thop import profile
except ImportError:
    profile = None


def count_model_stats(model, seq_len=256, mem_len=256, device='cuda'):
    """
    Counts MACs (multiply-add ops) and parameters for the model.
    Requires `thop` installed.
    """
    if profile is None:
        raise ImportError("Install thop to count FLOPs/MACs: pip install thop")

    model = model.to(device).eval()
    B, N, M = 1, seq_len, mem_len
    D = model.decoder_layer.self_attn.embed_dim
    x = torch.randn(B, N, D, device=device)
    mem = torch.randn(B, M, D, device=device)

    macs, params = profile(model, inputs=(x, mem), verbose=False)
    print(f"Model parameters: {params/1e6:.2f} M")
    print(f"MACs: {macs/1e9:.2f} G, FLOPs: {2*macs/1e9:.2f} G")
    return macs, params


if __name__ == '__main__':
    # Configura el modelo y datos de ejemplo
    model = build_vision_masking().cuda().eval()
    # Ajusta B, N, M, D según tu caso
    B, N, M, D = 1, 256, 256, 4096
    x = torch.randn(B, N, D, device='cuda')
    mem = torch.randn(B, M, D, device='cuda')

    # Cuenta MACs y parámetros
    if profile is None:
        print("Instala thop para contar flops/mac: pip install thop")
    else:
        macs, params = profile(model, inputs=(x, mem), verbose=False)
        print(f"MACs: {macs/1e9:.1f} G, Params: {params/1e6:.1f} M")
        print(f"MACs: {macs/1e9:.2f} G, FLOPs: {2*macs/1e9:.2f} G")

    # Medir latencia en A100 (FP32)
    with torch.no_grad():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Warm-up
        _ = model(x, mem)
        torch.cuda.synchronize()
        start.record()
        _ = model(x, mem)
        end.record()
        torch.cuda.synchronize()
        print(f"Latency: {start.elapsed_time(end):.2f} ms")

