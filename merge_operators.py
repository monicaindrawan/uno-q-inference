import torch.nn as nn
import torch

class FusionHead(nn.Module):
    """
    Per-node fusion head for combining own embedding with peer embeddings.

    Each node has its own fusion head, trained from its perspective.
    Returns a fused embedding (not logits) - the node's classifier is used after.
    """
    def __init__(self, embedding_dim=128, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Attention scoring for weighted pooling
        self.attn_score = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings, padding_mask=None):
        """
        Args:
            embeddings: (batch, N_nodes, embed_dim) - own embedding first, then peers
            padding_mask: (batch, N_nodes) - True for dropped/unavailable nodes

        Returns:
            fused: (batch, embed_dim) - fused embedding
            weights: (batch, N_nodes) - attention weights for interpretability
        """
        # L2-normalize
        normed = nn.functional.normalize(embeddings, p=2, dim=-1)

        # refined embeddings
        refined = self.transformer(normed)

        scores = self.attn_score(refined)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))

        weights = torch.softmax(scores, dim=1)
        fused = (weights * refined).sum(dim=1)

        return fused, weights.squeeze(-1)