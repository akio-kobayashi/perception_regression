import os
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch.utils.data as data
from cbs_hubert_dataset import CbsHubertDataset, data_processing
from cbs_hubert_solver import LitCbsHubert
import pandas as pd
import yaml
from argparse import ArgumentParser
from string import Template
import warnings
warnings.filterwarnings('ignore')


def load_config(path: str) -> dict:
    raw = open(path, 'r', encoding='utf-8').read()
    rendered = Template(raw).substitute(**os.environ)
    cfg = yaml.safe_load(rendered)
    return cfg.get('config', cfg)


def main(args, config: dict):
    # 1) モデルとデータローダーを準備
    model = LitCbsHubert(config)

    train_dataset = CbsHubertDataset(path=config['train_path'], config=config)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['process']['num_workers'],
        pin_memory=False,
        collate_fn=data_processing
    )

    valid_dataset = CbsHubertDataset(path=config['valid_path'], config=config)
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['process']['num_workers'],
        pin_memory=False,
        collate_fn=data_processing
    )

    # 2) コールバックとロガー
    checkpoint_cb = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    logger = TensorBoardLogger(**config['logger'])

    # 3) Trainer の作成と学習
    trainer = pl.Trainer(
        callbacks=[checkpoint_cb],
        logger=logger,
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=1,
        **config['trainer']
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.checkpoint
    )

    # 4) ベストモデルで検証データを予測し CSV 出力
    best_path = checkpoint_cb.best_model_path
    if best_path:
        best = LitCbsHubert.load_from_checkpoint(best_path, config=config)
        best.eval()

        df = pd.read_csv(config['valid_path'])
        preds_int, preds_nat, preds_cbs = [], [], []
        for batch in valid_loader:
            huberts, _, _, _, _, _, _ = batch
            with torch.no_grad():
                pred_int, pred_nat, pred_cbs = best.model.predict(huberts.to(best.device))
            preds_int.extend(pred_int.cpu().tolist())
            preds_nat.extend(pred_nat.cpu().tolist())
            preds_cbs.extend(pred_cbs.cpu().tolist())

        df['predict_int'] = [1.0 + (r-1)*0.5 for r in preds_int]
        df['predict_nat'] = [1.0 + (r-1)*0.5 for r in preds_nat]
        for i in range(4):
            df[f'predict_cb{i+1}'] = [c[i] for c in preds_cbs]
        df.to_csv(config['output_csv'], index=False)

        correct_int = (df['intelligibility'] == df['predict_int']).sum()
        acc_int = correct_int / len(df) if len(df) > 0 else 0
        print(f"Intelligibility validation accuracy: {acc_int:.4f} ({correct_int}/{len(df)})")

        correct_nat = (df['naturalness'] == df['predict_nat']).sum()
        acc_nat = correct_nat / len(df) if len(df) > 0 else 0
        print(f"Naturalness validation accuracy: {acc_nat:.4f} ({correct_nat}/{len(df)})")

        for i in range(4):
            correct_cb = (df[f'cb{i+1}'] == df[f'predict_cb{i+1}']).sum()
            acc_cb = correct_cb / len(df) if len(df) > 0 else 0
            print(f"CB{i+1} validation accuracy: {acc_cb:.4f} ({correct_cb}/{len(df)})")
    else:
        print("No best checkpoint found.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    config = load_config(args.config)
    main(args, config)
