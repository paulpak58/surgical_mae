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

# Model Imports
# from videomae import VideoMAE
# from videomae2 import PretrainVisionTransformer
# from maskfeat import MaskFeat
# from maevit import MAEViT
# from surgmae_mvit import SurgMAE_MViT
# from mask_generator import CubeMaskGenerator,TubeMaskingGenerator,RepresentativeMaskGenerator
from maskmvit import MaskMViT,build_cfg,build_cfg_maskfeat
from target_features import spatial_hog
from utils import MultipleMSELoss
from config.defaults import get_cfg
from videomae import TubeMaskingGenerator,PretrainVisionTransformer
from collections import OrderedDict

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


class PreTrainer(pl.LightningModule):
    def __init__(self,dim=768,mask_model=None,backbone=None,ckpt_dir='',num_epochs=75):
        super().__init__()
        self.dim = dim
        default_cfg = get_cfg()

        if mask_model=='videomae' and backbone=='vit_B':
            self.model = PretrainVisionTransformer(
                img_size=224,
                patch_size=16, 
                encoder_embed_dim=768, 
                encoder_depth=12, 
                encoder_num_heads=12,
                encoder_num_classes=0,
                decoder_num_classes=1536,
                decoder_embed_dim=384,
                decoder_num_heads=6,
                mlp_ratio=4, 
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.mask_generator = TubeMaskingGenerator(input_size=(8,14,14),mask_ratio=0.9)
        else:
            cfg = build_cfg(pretrain=True,mask_model=mask_model,backbone=backbone,default_cfg=default_cfg)
            self.model = MaskMViT(cfg)
            print('Model successfully loaded')

        if mask_model=='maskfeat' or mask_model=='mae':
            self.loss_fn = MultipleMSELoss()
        else:
            self.loss_fn = nn.MSELoss()
        self.valid_acc = Accuracy()
        self.num_epochs = num_epochs

        # Initialize Linear layers and load pretrained weights if exist
        self.init_weights(ckpt_dir)

    def init_weights(self,ckpt_dir):
        self.apply(self.__init__weights)
        if ckpt_dir!='':
            ckpt = torch.load(ckpt_dir)
            state_dict = OrderedDict()
            for k,v in ckpt['state_dict'].items():
                if 'model' in k:
                    name = k[6:]
                    state_dict[name] = v
            msg = self.model.load_state_dict(state_dict,strict=False)
            print('CKPT loaded',msg)

    def __init__weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m,nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            torch.nn.init.constant_(m.bias,0)
            torch.nn.init.constant_(m.weight,1.0)


    def forward_step(self,x,mask):
        if isinstance(self.model,MaskMViT):
            x = x.permute(0,2,1,3,4) 
            pred, labels = self.model(x,mask)
            loss = self.loss_fn(pred,labels)
        elif isinstance(self.model,PretrainVisionTransformer):
            mask = self.mask_generator()
            mask = torch.from_numpy(mask).unsqueeze(0).repeat(x.shape[0],1).type_as(x)
            pred,loss = self.model(x.transpose(1,2),mask)
        return loss

    def training_step(self, batch, batch_idx):
        x = batch["imgs"]
        mask = batch["mask"]
        loss = self.forward_step(x,mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["imgs"]
        mask = batch["mask"]
        loss = self.forward_step(x,mask)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = sum(outputs)/len(outputs)
        print("\n---------------------\nEpoch Loss: ",avg_loss,"\n---------------------")

    def configure_optimizers(self):
        '''
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=1.0e-5,
                betas=(0.9,0.999),
                weight_decay=0.01
            )
        else:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=1.0e-5,
                momentum=0.9,
                weight_decay=0.01
            )
        scheduler = get_cosine_schedule_warmup(
            optimizer,
            num_warmup_steps=30,
            num_training_steps=120,
            base_lr=4.0e-5,
            min_lr=0,
            last_epoch=-1,
            objective=None,
        )
        '''
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


if __name__ == "__main__":

    args = parse_arguments()
    params = vars(args)

    dataset_splits = ["train", "test"]
    training_ratio = {"train": 1.0, "test": 0.0}

    default_cfg = get_cfg()
    cfg = build_cfg(pretrain=True,mask_model=args.mask_model,backbone=args.backbone,default_cfg=default_cfg)

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
            masked=True,
            cfg=cfg
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
    )
    dataloader_test = DataLoader(
        val,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_dataloader_workers,
    )
    pretrainer = PreTrainer(mask_model=args.mask_model,backbone=args.backbone,ckpt_dir=args.ckpt_dir,num_epochs=args.num_epochs)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy='ddp',
        num_nodes=1,
        check_val_every_n_epoch=1,
        default_root_dir=args.log_dir,
        max_epochs=args.num_epochs,
        accumulate_grad_batches=128,
        precision=16,
    )
    trainer.fit(pretrainer, dataloader_train, dataloader_test, ckpt_path=args.ckpt_dir)
