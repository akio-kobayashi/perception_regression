import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from typing import List, Tuple
from torch import Tensor
from einops import rearrange
import coral_loss

# ----------------------------------------------------------------
# 1) CbsHubertDataset クラス
# ----------------------------------------------------------------
class CbsHubertDataset(torch.utils.data.Dataset):
    """
    データフレームから precomputed HuBERT 埋め込みを読み込む
    multi-task Dataset (int, nat, cbs)
    """
    def __init__(self, path: str, config: dict) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = pd.read_csv(path)
        if 'eval' in self.df.columns:
            self.df = self.df[self.df['eval'] != 'trial']
        self.data_length = len(self.df)
        self.cb_columns = ['cb1', 'cb2', 'cb3', 'cb4']

    @staticmethod
    def score_to_rank(score: float) -> int:
        # 例: スコア 1.0→ rank 1, 1.5→2, … 5.0→9
        return int(round((score - 1.0) / 0.5)) + 1

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int, Tensor]:
        row = self.df.iloc[idx]
        data = torch.load(row['hubert'], map_location='cpu')
        hubert = data['hubert']
        rank_int = self.score_to_rank(row['intelligibility'])
        rank_nat = self.score_to_rank(row['naturalness'])
        cbs = torch.FloatTensor([row[c] for c in self.cb_columns])
        return hubert, rank_int, rank_nat, cbs


# ----------------------------------------------------------------
# 2) data_processing 関数を multi-task (cbs) 用に変更
# ----------------------------------------------------------------
def data_processing(batch: List[Tuple[Tensor, int, int, Tensor]]):
    """
    batch: List of (hubert_feats, rank_int, rank_nat, cbs)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huberts, ranks_int, ranks_nat, cbs_list, lengths = [], [], [], [], []

    for hubert_feats, rank_int, rank_nat, cbs in batch:
        lengths.append(hubert_feats.shape[0])
        huberts.append(hubert_feats)
        ranks_int.append(rank_int)
        ranks_nat.append(rank_nat)
        cbs_list.append(cbs)

    # テンソル化
    ranks_int = torch.tensor(ranks_int, device=device)
    ranks_nat = torch.tensor(ranks_nat, device=device)
    lengths   = torch.tensor(lengths, device=device)
    cbs_batch = torch.stack(cbs_list, dim=0).to(device)
    
    # CORAL用のラベル行列
    labels_int = coral_loss.ordinal_labels(ranks_int, num_classes=9)
    labels_nat = coral_loss.ordinal_labels(ranks_nat, num_classes=9)

    # パディング
    huberts = nn.utils.rnn.pad_sequence(huberts, batch_first=True)

    return huberts, labels_int, ranks_int, labels_nat, ranks_nat, cbs_batch, lengths
