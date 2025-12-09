import torch
import torch.nn as nn
import torch.nn.functional as F

def logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    """
    CORAL の出力（logits）をラベルに変換．
    logits: Tensor of shape (batch_size, num_classes - 1) のシグモイド前の値
    Returns:
        Tensor of shape (batch_size,) with values in [1, num_classes]
    """
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1) + 1

class MtAttentionHubertCornModel(nn.Module):
    """
    Multi-task CORN model with self-attention on HuBERT embeddings.
    """
    def __init__(self,
                 hubert_dim: int = 768,
                 embed_dim: int = 256,
                 num_classes: int = 9,
                 n_heads: int = 4,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.hubert_proj = nn.Linear(hubert_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)

        # Classifiers for intelligibility
        self.classifiers_int = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim // 2, 1)
            )
            for _ in range(num_classes - 1)
        ])

        # Classifiers for naturalness
        self.classifiers_nat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim // 2, 1)
            )
            for _ in range(num_classes - 1)
        ])

    def extract_features(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hubert_proj(hubert_feats))
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        feat = x.mean(dim=1)
        return self.dropout(feat)

    def forward(self, hubert_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(hubert_feats)
        logits_int = torch.stack([clf(feat).squeeze(1) for clf in self.classifiers_int], dim=1)
        logits_nat = torch.stack([clf(feat).squeeze(1) for clf in self.classifiers_nat], dim=1)
        return logits_int, logits_nat

    def predict(self, hubert_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_int, logits_nat = self.forward(hubert_feats)
        probs_int = torch.sigmoid(logits_int)
        preds_int = (probs_int > 0.5).sum(dim=1) + 1
        probs_nat = torch.sigmoid(logits_nat)
        preds_nat = (probs_nat > 0.5).sum(dim=1) + 1
        return preds_int, preds_nat

class MtAttentionHubertOrdinalRegressionModel(nn.Module):
    """
    Multi-task CORAL model with self-attention on HuBERT embeddings.
    """
    def __init__(self,
                 hubert_dim: int = 768,
                 embed_dim: int = 256,
                 num_classes: int = 9,
                 n_heads: int = 4,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.hubert_proj = nn.Linear(hubert_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)

        # FC layers for each task
        self.shared_fc_int = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, 1)
        )
        self.shared_fc_nat = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, 1)
        )

        # Thresholds for each task
        self.thresholds_int = nn.Parameter(torch.zeros(num_classes - 1))
        self.thresholds_nat = nn.Parameter(torch.zeros(num_classes - 1))

    def extract_features(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hubert_proj(hubert_feats))
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        feat = x.mean(dim=1)
        return self.dropout(feat)

    def forward(self, hubert_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(hubert_feats)
        
        g_int = self.shared_fc_int(feat)
        logits_int = g_int.repeat(1, self.num_classes - 1) - self.thresholds_int.view(1, -1)
        
        g_nat = self.shared_fc_nat(feat)
        logits_nat = g_nat.repeat(1, self.num_classes - 1) - self.thresholds_nat.view(1, -1)
        
        return logits_int, logits_nat

    def predict(self, hubert_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_int, logits_nat = self.forward(hubert_feats)
        preds_int = logits_to_label(logits_int)
        preds_nat = logits_to_label(logits_nat)
        return preds_int, preds_nat