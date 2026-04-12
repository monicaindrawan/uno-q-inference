import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 128
NUM_CLASSES = 43
NUM_NODES = 2

# ============================================================================
# Model — CNN encoder + classifier (adapted for 48x48 input)
# ============================================================================
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


class FuseEncode(nn.Module):
    """
    Information fusion encoder for collaborative inference.
    Returns a fused embedding (not logits) - the node's classifier is used after.
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()

        self.emb_pad = 160 # padding for GPU optimisation 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_pad,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.compress = nn.Linear(in_features=NUM_NODES*self.emb_pad, out_features=embedding_dim)
        

    def forward(self, embeddings, padding_mask=None):
        """
        Args:
            embeddings (tagged): (batch, N_nodes, embed_dim+1) - sorted, stacked, normed and tagged. 
            padding_mask: (batch, N_nodes) - True for dropped/unavailable nodes

        Returns:
            fused: (batch, embed_dim) - fused embedding
        """
        embeddings = F.pad(embeddings, (0, self.emb_pad - embeddings.shape[-1]))
        refined = self.transformer(embeddings, src_key_padding_mask=padding_mask)

        refined = torch.flatten(refined, start_dim=1)

        fused = self.compress(refined)

        return fused
    
    
class FuseDecode(nn.Module):
    
    """
    Decodes the fused embedding back to normalised stacked embeddings. 
    This is only used for training purpose (forward differential loss).

    """
    def __init__(self, embedding_dim=EMBEDDING_DIM, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.emb_pad = 160 # padding for GPU optimisation 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_pad,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.project = nn.Linear(in_features=embedding_dim, out_features=NUM_NODES*self.emb_pad)
        

    def forward(self, embedding, padding_mask=None):
        """
        Args:
            embedding: (batch, embed_dim) - fused embeddings by FusedEncode

        Returns:
            decoded: (batch, N_nodes, embed_dim) - decoded stacked embedding
            tags: (batch, N_nodes) - recovered availability tag
        """
        batch_size = embedding.shape[0]
        projected = self.project(embedding)
        shape_recovered = torch.reshape(projected, (batch_size, NUM_NODES, self.emb_pad))
        decoded = self.transformer(shape_recovered)

        tags = decoded[:,:,self.embedding_dim]
        decoded = decoded[:,:,:self.embedding_dim]

        return tags, decoded 


class NodeModel(nn.Module):
    """
    A node in the decentralized network.

    Each node has:
    - encoder: extracts embeddings from images
    - classifier: classifies embeddings (solo mode)
    - fuse_encode: combines own + peer embeddings (collaborative mode)
    """
    def __init__(self, include_fusion=True):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.fuse_encode = FuseEncode() if include_fusion else None

    def forward(self, x):
        """Solo inference - just encode and classify."""
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding

    def fused_forward(self, own_embedding, peer_embeddings, mask):
        """
        Collaborative inference - fuse own embedding with peers and classify.

        Args:
            own_embedding: tuple (name, embedding) with embedding size = (batch, embed_dim) - this node's embedding
            peer_embeddings: list of tuples (name, embedding) with size = (batch, embed_dim)
            mask: (batch, NUM_NODES) - concacenated availablity tag: This is now NOT an optional parameter 

        Returns:
            logits: (batch, num_classes) 
            log_logits: (batch, num_classes), used for forward and backward loss training (symmetric KL-divergence)
        """
        if self.fuse_encode is None: 
            raise ValueError("This node has no fusion head. Initialize with include_fusion=True")

        # Sort the nodes by alphabetical order 
        all_embs_labelled = [own_embedding] + list(peer_embeddings)
        sorted_embs_labelled = sorted(all_embs_labelled, key=lambda x: x[0])
        embs_only = [item[1] for item in sorted_embs_labelled]
        
        # Stack
        stacked = torch.stack(embs_only, dim=1)  # (batch, N_nodes, embed_dim)

        mask = mask.unsqueeze(-1)
        stacked_n = nn.functional.normalize(stacked, p=2, dim=-1)
        stacked_masked_n = stacked_n * mask
        

        stacked_masked_n_tagged = torch.cat([stacked_masked_n, mask], dim=-1)

        # Create padding mask for transformer (True = ignore)
        padding_mask = (mask.squeeze(-1) == 0)

        fused = self.fuse_encode(stacked_masked_n_tagged, padding_mask)
        logits = self.classifier(fused)

        return logits 
