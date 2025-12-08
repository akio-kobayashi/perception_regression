import torch
import torch.nn as nn
import torch.nn.functional as F

class MtHubertCornModel(nn.Module):
    """
    Multi-task CORN model for predicting intelligibility and naturalness.
    """
    def __init__(self,
                 hubert_dim: int = 768,
                 proj_dim: int = 256,
                 num_classes: int = 9,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.proj = nn.Linear(hubert_dim, proj_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Classifiers for intelligibility
        self.classifiers_int = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(proj_dim // 2, 1)
            )
            for _ in range(num_classes - 1)
        ])

        # Classifiers for naturalness
        self.classifiers_nat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(proj_dim // 2, 1)
            )
            for _ in range(num_classes - 1)
        ])

    def extract_features(self, hubert_feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(hubert_feats))
        x = x.mean(dim=1)
        return self.dropout(x)

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