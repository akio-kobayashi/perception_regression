import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import corn_loss
import coral_loss
from cbs_hubert_model import CbsAttentionHubertCornModel, CbsAttentionHubertOrdinalRegressionModel

class LitCbsHubert(pl.LightningModule):
    """
    PyTorch Lightning solver for multi-task HuBERT model (int, nat, cbs).
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        model_cfg = config['model'].copy()
        class_name = model_cfg.pop('class_name', 'CbsAttentionHubertCornModel')
        
        model_map = {
            'CbsAttentionHubertCornModel': CbsAttentionHubertCornModel,
            'CbsAttentionHubertOrdinalRegressionModel': CbsAttentionHubertOrdinalRegressionModel,
        }
        ModelClass = model_map[class_name]
        
        self.model = ModelClass(**model_cfg)
        self.save_hyperparameters()

        # Metrics
        self.num_correct_int = 0
        self.num_total_int = 0
        self.num_correct_nat = 0
        self.num_total_nat = 0
        self.num_correct_cbs = torch.zeros(4)
        self.num_total_cbs = 0

    def forward(self, hubert_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(hubert_feats)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        huberts, labels_int, ranks_int, labels_nat, ranks_nat, cbs_batch, lengths = batch
        
        logits_int, logits_nat, logits_cbs = self.forward(huberts)
        
        if isinstance(self.model, CbsAttentionHubertCornModel):
            loss_int = corn_loss.corn_loss(logits_int, labels_int)
            loss_nat = corn_loss.corn_loss(logits_nat, labels_nat)
        else:
            loss_int = coral_loss.coral_loss(logits_int, labels_int)
            loss_nat = coral_loss.coral_loss(logits_nat, labels_nat)
            
        loss_cbs = F.binary_cross_entropy_with_logits(logits_cbs, cbs_batch)
        
        loss = loss_int + loss_nat + loss_cbs
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_loss_int', loss_int)
        self.log('train_loss_nat', loss_nat)
        self.log('train_loss_cbs', loss_cbs)
        
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        huberts, labels_int, ranks_int, labels_nat, ranks_nat, cbs_batch, lengths = batch
        
        logits_int, logits_nat, logits_cbs = self.forward(huberts)
        
        if isinstance(self.model, CbsAttentionHubertCornModel):
            loss_int = corn_loss.corn_loss(logits_int, labels_int)
            loss_nat = corn_loss.corn_loss(logits_nat, labels_nat)
        else:
            loss_int = coral_loss.coral_loss(logits_int, labels_int)
            loss_nat = coral_loss.coral_loss(logits_nat, labels_nat)

        loss_cbs = F.binary_cross_entropy_with_logits(logits_cbs, cbs_batch)
        loss = loss_int + loss_nat + loss_cbs
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_loss_int', loss_int)
        self.log('val_loss_nat', loss_nat)
        self.log('val_loss_cbs', loss_cbs)
        
        preds_int, preds_nat, preds_cbs = self.model.predict(huberts)

        self.num_correct_int += (preds_int == ranks_int).sum().item()
        self.num_total_int += ranks_int.size(0)
        
        self.num_correct_nat += (preds_nat == ranks_nat).sum().item()
        self.num_total_nat += ranks_nat.size(0)

        self.num_correct_cbs += (preds_cbs == cbs_batch).sum(dim=0).cpu()
        self.num_total_cbs += cbs_batch.size(0)

    def on_validation_epoch_end(self) -> None:
        acc_int = self.num_correct_int / self.num_total_int if self.num_total_int > 0 else 0.0
        self.log('val_acc_int', acc_int, prog_bar=True)
        
        acc_nat = self.num_correct_nat / self.num_total_nat if self.num_total_nat > 0 else 0.0
        self.log('val_acc_nat', acc_nat, prog_bar=True)
        
        if self.num_total_cbs > 0:
            acc_cbs = self.num_correct_cbs / self.num_total_cbs
            for i, acc in enumerate(acc_cbs):
                self.log(f'val_acc_cb{i+1}', acc, prog_bar=False)
        
        # Reset metrics
        self.num_correct_int = 0
        self.num_total_int = 0
        self.num_correct_nat = 0
        self.num_total_nat = 0
        self.num_correct_cbs = torch.zeros(4)
        self.num_total_cbs = 0

    def configure_optimizers(self):
        opt_cfg = self.config['optimizer'].copy()
        opt_cfg['lr'] = float(opt_cfg['lr'])
        opt_cfg['weight_decay'] = float(opt_cfg['weight_decay'])
        optimizer = torch.optim.Adam(self.model.parameters(), **opt_cfg)
        
        sched_cfg = self.config['scheduler'].copy()
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **sched_cfg),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]