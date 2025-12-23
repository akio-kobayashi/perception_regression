import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from typing import List, Tuple
from torch import Tensor
from einops import rearrange
import coral_loss

# ----------------------------------------------------------------
# 1) HubertDataset クラス
# ----------------------------------------------------------------
class HubertDataset(torch.utils.data.Dataset):
    """
    データフレーム(path, feature, intelligibility/naturalness)から
    precomputed HuBERT 埋め込みを読み込む Dataset
    """
    def __init__(self, path: str, config: dict) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = pd.read_csv(path)
        if 'eval' in self.df.columns:
            self.df = self.df[self.df['eval'] != 'trial']
        self.data_length = len(self.df)
        self.target_column = config.get('target_column', 'intelligibility')

    @staticmethod
    def score_to_rank(score: float) -> int:
        # 例: スコア 1.0→ rank 1, 1.5→2, … 5.0→9
        return int(round((score - 1.0) / 0.5)) + 1

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        row = self.df.iloc[idx]
        # torch.save で { 'hubert_feats': Tensor[T, D] } の形式で保存している前提
        data = torch.load(row['hubert'], map_location='cpu')
        #hubert = data['hubert_feats']  # shape=(T, hubert_dim)
        hubert = data['hubert']
        rank = self.score_to_rank(row[self.target_column])
        return hubert, rank


# ----------------------------------------------------------------
# 2) data_processing 関数を Hubert 用に変更
# ----------------------------------------------------------------
def data_processing(batch: List[Tuple[Tensor, int]]):
    """
    batch: List of (hubert_feats, rank)
    Returns:
        huberts:   Tensor of shape (B, T_max, D)
        labels:    Tensor of shape (B, K-1)  (ordinal labels for coral_loss)
        ranks:     Tensor of shape (B,)
        lengths:   Tensor of shape (B,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huberts, ranks, lengths = [], [], []

    for hubert_feats, rank in batch:
        # hubert_feats: (T, D)
        lengths.append(hubert_feats.shape[0])
        huberts.append(hubert_feats)
        ranks.append(rank)

    # テンソル化
    ranks   = torch.tensor(ranks, device=device)
    lengths = torch.tensor(lengths, device=device)
    # CORAL用のラベル行列 (B, num_classes-1)
    labels  = coral_loss.ordinal_labels(ranks, num_classes=9)

    # パディング: (B, T_max, D)
    huberts = nn.utils.rnn.pad_sequence(huberts, batch_first=True)

    return huberts, labels, ranks, lengths


# ----------------------------------------------------------------
# 3) DataLoader 例
# ----------------------------------------------------------------
if __name__ == "__main__":
    # CSV には 'feature' 列に torch.save したファイルパス、
    # 'intelligibility' 列に 1.0-5.0 のスコアがある想定
    dataset = HubertDataset(path="data/metadata.csv")
    loader  = data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=data_processing,
        num_workers=4,
        pin_memory=False
    )

    for huberts, labels, ranks, lengths in loader:
        # huberts: (16, T_max, D)
        # labels:  (16, 8)
        # ranks:   (16,)
        # lengths: (16,)
        print(huberts.shape, labels.shape)
        break
