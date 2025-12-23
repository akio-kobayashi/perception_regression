import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from typing import List, Tuple
from torch import Tensor
from einops import rearrange
import coral_loss

# ----------------------------------------------------------------
# 1) MtHubertDataset クラス
# ----------------------------------------------------------------
class MtHubertDataset(torch.utils.data.Dataset):
    """
    データフレーム(path, feature, intelligibility, naturalness)から
    precomputed HuBERT 埋め込みを読み込む multi-task Dataset
    """
    def __init__(self, path: str, config: dict) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = pd.read_csv(path)
        if 'eval' in self.df.columns:
            self.df = self.df[self.df['eval'] != 'trial']
        self.data_length = len(self.df)

    @staticmethod
    def score_to_rank(score: float) -> int:
        # 例: スコア 1.0→ rank 1, 1.5→2, … 5.0→9
        return int(round((score - 1.0) / 0.5)) + 1

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int]:
        row = self.df.iloc[idx]
        data = torch.load(row['hubert'], map_location='cpu')
        hubert = data['hubert']
        rank_int = self.score_to_rank(row['intelligibility'])
        rank_nat = self.score_to_rank(row['naturalness'])
        return hubert, rank_int, rank_nat


# ----------------------------------------------------------------
# 2) data_processing 関数を multi-task 用に変更
# ----------------------------------------------------------------
def data_processing(batch: List[Tuple[Tensor, int, int]]):
    """
    batch: List of (hubert_feats, rank_int, rank_nat)
    Returns:
        huberts:    Tensor of shape (B, T_max, D)
        labels_int: Tensor of shape (B, K-1) for intelligibility
        ranks_int:  Tensor of shape (B,) for intelligibility
        labels_nat: Tensor of shape (B, K-1) for naturalness
        ranks_nat:  Tensor of shape (B,) for naturalness
        lengths:    Tensor of shape (B,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huberts, ranks_int, ranks_nat, lengths = [], [], [], []

    for hubert_feats, rank_int, rank_nat in batch:
        lengths.append(hubert_feats.shape[0])
        huberts.append(hubert_feats)
        ranks_int.append(rank_int)
        ranks_nat.append(rank_nat)

    # テンソル化
    ranks_int = torch.tensor(ranks_int, device=device)
    ranks_nat = torch.tensor(ranks_nat, device=device)
    lengths   = torch.tensor(lengths, device=device)
    
    # CORAL用のラベル行列
    labels_int = coral_loss.ordinal_labels(ranks_int, num_classes=9)
    labels_nat = coral_loss.ordinal_labels(ranks_nat, num_classes=9)

    # パディング
    huberts = nn.utils.rnn.pad_sequence(huberts, batch_first=True)

    return huberts, labels_int, ranks_int, labels_nat, ranks_nat, lengths
