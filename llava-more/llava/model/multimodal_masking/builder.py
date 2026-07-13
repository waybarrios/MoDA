import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderMask(nn.Module):
    """
    Generate a sparse mask over x using a Transformer decoder with input_embeds as memory.
    The decoder learns context-aware masking by attending over language or other inputs.
    """
    def __init__(self, embedding_dim=4096, hidden_dim=1024, num_heads=16, num_layers=2, l1_lambda=1e-5,sparsity_lambda=1e-3):
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
        #self.activation = nn.SiLU()
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
         
        #if labels is not None:
        #    norm_x = masked_x / masked_x.norm(dim=-1, keepdim=True)
        #    norm_labels = labels / labels.norm(dim=-1, keepdim=True)
        #    self.loss_value = 1 - torch.mean(torch.matmul(norm_x, norm_labels.transpose(-2, -1)))

        return masked_x

    def l1_loss(self):
        """L1 regularization on the projection head to encourage sparse mask weights."""
        l1_penalty = self.l1_lambda * F.l1_loss(self.mask_head.weight, torch.zeros_like(self.mask_head.weight), reduction="sum")
        return self.loss_value  # + l1_penalty if self.loss_value is not None else l1_penalty
    def get_sparsity_loss(self):
        """Encourages mask values to be close to 0 (sparse)."""
        if self.last_mask is None:
            loss =  0.0
        else:
            loss = torch.mean(torch.abs(self.last_mask)) #self.sparsity_lambda * torch.mean(torch.abs(self.last_mask))
            self.last_mask = None # clean 
        return loss #self.sparsity_lambda * torch.mean(torch.abs(self.last_mask))

# Function to build the transformer-based vision masking model
def build_vision_masking(embedding_dim=4096, hidden_dim=1024, num_heads=16, num_layers=2, l1_lambda=1e-5,sparsity_lambda=1e-3):
    return EncoderMask(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=1, #num_layers,
        l1_lambda=l1_lambda,
        sparsity_lambda=sparsity_lambda
    )

