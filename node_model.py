import torch
import torch.nn as nn
import torch.nn.functional as F

from merge_operators import FusionHead

EMBEDDING_DIM = 128
NUM_CLASSES = 43
NUM_NODES = 2

class Encoder(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 12 -> 6
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 6 -> 1x1
        )
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x


class Classifier(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embedding):
        return self.head(embedding)


class NodeModel(nn.Module):
    """
    A node in the decentralized network.

    Each node has:
    - encoder: extracts embeddings from images
    - classifier: classifies embeddings (solo mode)
    - fusion_head: combines own + peer embeddings (collaborative mode)
    """
    def __init__(self, include_fusion=True):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.fusion_head = FusionHead() if include_fusion else None

    def forward(self, x):
        """Solo inference - just encode and classify."""
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding

    def fused_forward(self, own_embedding, peer_embeddings, padding_mask=None):
        """
        Collaborative inference - fuse own embedding with peers and classify.

        Args:
            own_embedding: (batch, embed_dim) - this node's embedding
            peer_embeddings: list of (batch, embed_dim) - peer embeddings
            padding_mask: (batch, N_nodes) - optional mask for dropped peers

        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, N_nodes)
        """
        if self.fusion_head is None:
            raise ValueError("This node has no fusion head. Initialize with include_fusion=True")

        # Stack: own embedding first, then peers
        all_embs = [own_embedding] + list(peer_embeddings)
        stacked = torch.stack(all_embs, dim=1)  # (batch, N_nodes, embed_dim)

        fused, attn_weights = self.fusion_head(stacked, padding_mask)
        logits = self.classifier(fused)

        return logits, attn_weights