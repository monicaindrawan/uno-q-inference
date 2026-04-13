"""
Merge Operators for Node Communication Layer
=============================================

Abstract base class and implementations for combining node embeddings
during collaborative inference.

All operators receive:
    - requesting_emb: Embedding from the node requesting collaboration (batch, embed_dim)
    - peer_embeddings: List of embeddings from peer nodes [(batch, embed_dim), ...]
    - peer_contexts: Dict with metadata per peer (confidences, names, etc.)

All operators return:
    - fused_embedding: Combined embedding (batch, embed_dim)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class MergeOperator(ABC):
    """Abstract base class for merge operators."""

    @abstractmethod
    def merge(
        self,
        requesting_emb: torch.Tensor,
        peer_embeddings: List[torch.Tensor],
        peer_contexts: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Merge embeddings from requesting node and peers.

        Args:
            requesting_emb: (batch, embed_dim) embedding from requesting node
            peer_embeddings: List of (batch, embed_dim) tensors from peers
            peer_contexts: Dict containing:
                - "confidences": List of confidence scores per peer
                - "peer_names": List of peer node names
                - "requesting_name": Name of requesting node
                - "requesting_confidence": Confidence of requesting node

        Returns:
            fused_embedding: (batch, embed_dim) merged embedding
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name for logging."""
        pass

    @property
    def trainable(self) -> bool:
        """Return True if this operator has learnable parameters."""
        return False

    def parameters(self):
        """Return trainable parameters (empty by default)."""
        return iter([])


class ConfidenceWeightedMean(MergeOperator):
    """
    Weighted average of embeddings by confidence scores.

    merged = sum(conf_i * emb_i) / sum(conf_i)

    Higher-confidence nodes contribute more to the fused embedding.
    """

    def merge(self, requesting_emb, peer_embeddings, peer_contexts):
        confidences = peer_contexts["confidences"]
        requesting_conf = peer_contexts.get("requesting_confidence", 0.5)

        all_embs = [requesting_emb] + peer_embeddings
        all_confs = [requesting_conf] + list(confidences)

        # Weighted sum
        weighted_sum = torch.zeros_like(requesting_emb)
        weight_total = 0.0

        for emb, conf in zip(all_embs, all_confs):
            weighted_sum = weighted_sum + conf * emb
            weight_total += conf

        return weighted_sum / (weight_total + 1e-8)

    @property
    def name(self):
        return "ConfidenceWeightedMean"


class RobustMedian(MergeOperator):
    """
    Element-wise median across all embeddings.

    Robust to outliers from malfunctioning or adversarial nodes.
    """

    def merge(self, requesting_emb, peer_embeddings, peer_contexts):
        all_embs = [requesting_emb] + peer_embeddings
        stacked = torch.stack(all_embs, dim=1)  # (batch, N, embed_dim)
        return torch.median(stacked, dim=1).values  # (batch, embed_dim)

    @property
    def name(self):
        return "RobustMedian"


class TopKConfident(MergeOperator):
    """
    Average embeddings from K most confident nodes only.

    Ignores low-confidence nodes that might add noise.
    """

    def __init__(self, k: int = 2):
        self.k = k

    def merge(self, requesting_emb, peer_embeddings, peer_contexts):
        confidences = peer_contexts["confidences"]
        requesting_conf = peer_contexts.get("requesting_confidence", 0.5)

        all_embs = [requesting_emb] + peer_embeddings
        all_confs = [requesting_conf] + list(confidences)

        # Sort by confidence (descending)
        sorted_indices = sorted(
            range(len(all_confs)), key=lambda i: all_confs[i], reverse=True
        )
        top_k_indices = sorted_indices[: self.k]

        # Average top-K embeddings
        top_embs = [all_embs[i] for i in top_k_indices]
        return torch.stack(top_embs, dim=1).mean(dim=1)

    @property
    def name(self):
        return f"TopK(k={self.k})"


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


class PerNodeFusion(MergeOperator):
    """
    Placeholder for per-node fusion.

    This operator signals that the requesting node's own fusion_head should be used.
    In collaborative_inference_with_dropout, passing None as merge_operator
    triggers this behavior.
    """

    def merge(self, requesting_emb, peer_embeddings, peer_contexts):
        # This should not be called directly - it's a signal to use node's fusion_head
        raise NotImplementedError(
            "PerNodeFusion.merge() should not be called directly. "
            "Pass None as merge_operator to use each node's built-in fusion head."
        )

    @property
    def name(self):
        return "PerNodeFusion"


def create_all_merge_operators() -> List[Optional[MergeOperator]]:
    """
    Create all available merge operator instances.

    Returns a list including None (for PerNodeFusion) and other merge operators.

    Returns:
        List where None represents PerNodeFusion, plus other MergeOperator instances
    """
    # None = use each node's built-in fusion head (PerNodeFusion)
    # Other operators are alternatives that can be tested
    operators = [
        None,  # PerNodeFusion - uses node's own fusion_head
        ConfidenceWeightedMean(),
        RobustMedian(),
        TopKConfident(k=2),
    ]

    return operators
