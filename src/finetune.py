import os
import math
import torch
import torch.nn.functional as F
import torchvision
import torchmetrics
import pytorch_lightning as pl
import numpy as np
from torch import nn
from optimizers import build_optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from data_utils.protobuf_dataset import process_data_directory_surgery
from masked_dataset import process_data_directory_surgery

from misc.base_params import parse_arguments
from functools import partial
from torchmetrics import (
    Accuracy,
    MetricCollection,
    Precision,
    Recall,
    F1Score,
    JaccardIndex,
)
from collections import OrderedDict

# Model Imports
import utils
from mvitv2 import MViT,build_cfg_mvitv2
from maskmvit import MaskMViT,build_cfg
from target_features import spatial_hog
from utils import MultipleMSELoss
from config.defaults import get_cfg
from videomae import TubeMaskingGenerator,PretrainVisionTransformer
from swintransformer import SwinTransformer3D,build_cfg_swin

torch.manual_seed(0)
np.random.seed(0)


def get_cosine_schedule_warmup(
    optimizer,
    num_warmup_steps=40,
    num_training_steps=800,
    base_lr=1.5e-4,
    min_lr=1e-5,
    last_epoch=-1,
    objective=None,
):
    """
    Schedule with learning rate that decreases following the values of cosine function between 0 and "pi*cycle" after
    warmup period where it increases between 0 and base_lr
    """

    def lr_lambda(epoch):
        epoch += 1
        if epoch <= num_warmup_steps:
            return float(epoch) / float(max(1, num_warmup_steps))
        progress = min(
            float(epoch - num_warmup_steps)
            / float(max(1, num_training_steps - num_warmup_steps)),
            1,
        )
        if objective == "mim":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return factor * (1 - min_lr / base_lr) + min_lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class FinetuneTrainer(pl.LightningModule):
    def __init__(self,dim=768,mask_model=None,backbone=None,n_class=7,class_names=[],ckpt_dir=None,num_epochs=75):
        super().__init__()
        self.dim = dim
        self.n_class = n_class
        self.class_names = class_names
        default_cfg = get_cfg()

        if mask_model=='videomae' and backbone=='vit_B':
            cfg = build_cfg(pretrain=False,mask_model=mask_model,backbone=backbone,default_cfg=default_cfg)
            self.model = MViT(cfg)
            self.model.class_head = nn.Sequential(
                OrderedDict([
                    ('norm',nn.LayerNorm(768)),
                    ('projection',nn.Linear(768,7,bias=True)),
                    ('act',nn.Softmax(dim=1))
                ])
            )
            nn.init.constant_(self.model.class_head.norm.weight,1.0)
            nn.init.constant_(self.model.class_head.norm.bias,0.02)
            nn.init.trunc_normal_(self.model.class_head.projection.weight,std=0.02)
            nn.init.constant_(self.model.class_head.projection.bias,0.02)
            self.model.act = nn.Softmax(dim=1)
        elif backbone=='mvitv2_S' or backbone=='mvitv2_B':
            cfg = build_cfg(pretrain=False,mask_model=mask_model,backbone=backbone,default_cfg=default_cfg)
            self.model = MViT(cfg)
        else:
            print('swin')
            cfg = build_cfg_swin(size='base')
            self.model = SwinTransformer3D(
                embed_dim=cfg['embed_dim'],
                depths=cfg['depths'],
                num_heads=cfg['num_heads'],
                patch_size=cfg['patch_size'],
                window_size=cfg['window_size'],
                drop_path_rate=cfg['drop_path_rate'],
                patch_norm=cfg['patch_norm'])
        self.valid_acc = Accuracy()
        self.num_epochs = num_epochs
        self.init_weights(ckpt_dir)

    def init_weights(self,ckpt_dir):
        self.apply(self.__init__weights)
        if ckpt_dir is not None:
            ckpt = torch.load(ckpt_dir)
            state_dict = OrderedDict()
            for k,v in ckpt['state_dict'].items():
                if 'model' in k:
                    name = k[6:]
                    state_dict[name] = v
            msg = self.model.load_state_dict(state_dict,strict=False)
            print(msg)

    def __init__weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m,nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            torch.nn.init.constant_(m.bias,0)
            torch.nn.init.constant_(m.weight,1.0)

    def forward_step(self,x):
        if isinstance(self.model,MViT):
            y_hat = self.model(x.transpose(1,2).unsqueeze(0))
        elif isinstance(self.model,PretrainVisionTransformer):
            x = x.transpose(1,2)
            y_hat = self.class_head(self.model(x.transpose(1,2)))
        return y_hat

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # x = batch[0]
        # y = batch[1][:,:,0]
        x = batch["imgs"]
        y = batch["phase_trajectory"]
        loss = 0
        y_hat = self.forward_step(x)
        loss += F.cross_entropy(y_hat, y[:,-1, :].squeeze())
        self.log('train_loss', loss)
        return loss

    def on_validation_start(self) -> None:
        # define cm
        self.cm = torch.zeros(self.n_class, self.n_class)

    def validation_step(self, batch, batch_idx):
        # x = batch[0]
        # y = batch[1][:,:,0]
        x = batch["imgs"]
        y = batch["phase_trajectory"]
        self.seq_len = y.shape[1]
        loss = 0
        y_hat = self.forward_step(x)
        loss += F.cross_entropy(y_hat, y[:,-1, :].squeeze())
        for idx_batch in range(y.shape[0]):
            gt = y[idx_batch, -1].argmax(dim=-1)
            est = y_hat[idx_batch].argmax(dim=-1)
            self.cm[int(gt), int(est)] += 1.0
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        cm = self.cm.detach().cpu().numpy()
        accuracy = cm.diagonal() / cm.sum(axis=0)
        accuracy[np.isnan(accuracy)] = 0.0
        print("confusion matrix:")
        print(cm.astype(int))
        print("Recall:")
        accuracy = cm.diagonal() / cm.sum(axis=-1)
        accuracy[np.isnan(accuracy)] = 0.0
        stats = [self.current_epoch, "Recall"]
        for idx, class_name in  enumerate(self.class_names):
            print(class_name + ':' + str(accuracy[idx]))
            stats.append(accuracy[idx])
        accuracy_mean = accuracy[accuracy != 0].mean()
        print('Overall recall' + ' :' + str(accuracy_mean) + '\n')
        stats.append(accuracy_mean)
        accuracy_overall = cm.diagonal().sum() / cm.sum(axis=-1).sum()
        print('average accuracy' + ' :' + str(accuracy_overall) + '\n')
        stats.append(accuracy_overall)
        # self.stat_writer.writerow(stats)

        print("Precision:")
        stats = [self.current_epoch, "Precision"]
        precision = cm.diagonal() / cm.sum(axis=0)
        precision[np.isnan(precision)] = 0.0
        for idx, class_name in  enumerate(self.class_names):
            print(class_name + ':' + str(precision[idx]))
            stats.append(precision[idx])
        precision_mean = precision[precision != 0].mean()
        print('Overall precision' + ' :' + str(precision_mean)  + '\n')
        stats.append(precision_mean)
        accuracy_overall = cm.diagonal().sum() / cm.sum(axis=-1).sum()
        print('average accuracy' + ' :' + str(accuracy_overall) + '\n')
        stats.append(accuracy_overall)
        
        # self.stat_writer.writerow(stats)

        self.log('val_accuracy', accuracy_overall)


    def configure_optimizers(self):
        opt_params = {
            'opt':'adamw',
            'lr':4.0e-5,
            'min_lr':1.0e-7,
            'weight_decay':0.05,
            'num_warmup_steps':5,
            'num_training_steps':self.num_epochs,
            # 'warmup_start_lr':1.0e-8
        }
        optimizer = build_optimizer(opt_params, self.model, is_pretrain=True)
        lr_scheduler = get_cosine_schedule_warmup(
            optimizer=optimizer,
            num_warmup_steps=opt_params['num_warmup_steps'],
            num_training_steps=opt_params['num_training_steps'],
            base_lr=opt_params['lr'],
            min_lr=opt_params['min_lr'],
        )
        return [optimizer], [lr_scheduler]
        # optimizer = torch.optim.Adam(self.parameters(),lr=1.0e-5)
        # return optimizer


if __name__ == "__main__":

    args = parse_arguments()
    params = vars(args)

    dataset_splits = ["train", "test"]
    training_ratio = {"train": 1.0, "test": 0.0}

    for split in dataset_splits:
        datasets = process_data_directory_surgery(
            data_dir=args.data_dir,
            fractions=args.fractions,
            width=args.image_width,
            height=args.image_height,
            sampling_rate=args.sampling_rate,
            past_length=args.temporal_length,
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            sampler=None,
            verbose=False,
            annotation_folder=args.annotation_filename + "/" + split,
            temporal_len=args.temporal_length,
            train_ratio=training_ratio[split],
            skip_nan=True,
            seed=1234,
            phase_translation_file=args.phase_translation_file,
            cache_dir=args.cache_dir,
            params=params,
            masked=False,
            cfg=None
        )

        if split == "train":
            train = datasets["train"]
        elif split == "test":
            val = datasets["val"]

    dataloader_train = DataLoader(
        train,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_dataloader_workers,
        pin_memory=True
    )
    dataloader_test = DataLoader(
        val,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_dataloader_workers,
        pin_memory=True
    )
    # finetuner = FinetuneTrainer(mask_model=args.mask_model,backbone=args.backbone,class_names=train.class_names,ckpt_dir=args.ckpt_dir)
    finetuner = FinetuneTrainer(mask_model=args.mask_model,backbone=args.backbone,class_names=train.class_names,num_epochs=args.num_epochs)
    '''
    finetuner = FinetuneTrainer.load_from_checkpoint(
            checkpoint_path=args.ckpt_dir,
            strict=False,
            mask_model=args.mask_model,
            backbone=args.backbone,
            class_names=train.class_names
    )
    '''
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy='ddp',
        num_nodes=1,
        check_val_every_n_epoch=1,
        default_root_dir=args.log_dir,
        max_epochs=args.num_epochs,
        accumulate_grad_batches=64,
        # accumulate_grad_batches=128 if args.backbone!='vit_L' else 256,
        precision=16,
    )
    trainer.fit(finetuner, dataloader_train, dataloader_test, ckpt_path=args.finetune_ckpt_dir)
    # trainer.fit(finetuner, dataloader_train, dataloader_test)
