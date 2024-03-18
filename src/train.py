import os
import shutil
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchmetrics
from torchmetrics import (
    Accuracy,
    MetricCollection,
    Precision,
    Recall,
    F1Score,
    JaccardIndex,
)
from torchmetrics.functional import accuracy
import numpy as np
import pytorch_lightning as pl
from data_utils.protobuf_dataset import process_data_directory_surgery
from optimizers import build_optimizer
from misc.base_params import parse_arguments
from collections import OrderedDict

# Import models
from mvitv2 import MViT,build_cfg_mvitv2
from swintransformer import SwinTransformer3D,build_cfg_swin

torch.manual_seed(0)
np.random.seed(0)


def get_cosine_schedule_warmup(
    optimizer,
    num_warmup_steps=5,
    num_training_steps=30,
    base_lr=1.0e-5,
    min_lr=0,
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


class TemporalTrainer(pl.LightningModule):
    def __init__(self, class_names=[], model_name=None, size='base', pretrain=True,ckpt_dir=None):
        super().__init__()

        self.class_names = class_names
        self.n_class = len(self.class_names)
        self.model_name= model_name
        self.size = size
        if(model_name=='mvitv2'):
            if(size=='base'):
                cfg = build_cfg_mvitv2(size='base')
                self.model = MViT(cfg)
            else:   # size=='small'
                cfg = build_cfg_mvitv2(size='small')
                self.model = MViT(cfg)
                # self.embed_dim = list(self.model.children())[-1].normalized_shape[0]
        elif(model_name=='videoswin'):
            if(size=='base'):
                cfg = build_cfg_swin(size='base')
                self.model = SwinTransformer3D(
                    embed_dim=cfg['embed_dim'],
                    depths=cfg['depths'],
                    num_heads=cfg['num_heads'],
                    patch_size=cfg['patch_size'],
                    window_size=cfg['window_size'],
                    drop_path_rate=cfg['drop_path_rate'],
                    patch_norm=cfg['patch_norm'])
            else:   # size=='small'
                self.model = None
        
        metrics = MetricCollection(
            [
                Accuracy(num_classes=self.n_class),
                Precision(num_classes=self.n_class),
                Recall(num_classes=self.n_class),
                F1Score(num_classes=self.n_class),
                JaccardIndex(num_classes=self.n_class),
            ]
        )
        self.test_metrics = metrics.clone(prefix="test_")
        self.init_weights(pretrain,ckpt_dir)

    def init_weights(self,pretrain,ckpt_dir):
        self.apply(self.__init__weights)
        if pretrain:
            print('Loading default pretrain weights')
            home_dir = '/home/paulpak'
            if self.model_name=='mvitv2':
                if self.size=='base':
                    ckpt_path = home_dir + '/.cache/torch/hub/checkpoints/MViTv2_B_32x3_k400_f304025456.pyth'
                    if not os.path.exists(ckpt_path):
                        ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth')
                    else:
                        ckpt = torch.load(ckpt_path)
                    msg = self.model.load_state_dict(ckpt['model_state'],strict=False)
                    # print(msg)
                else:
                    ckpt_path = home_dir + '/.cache/torch/hub/checkpoints/MViTv2_S_16x4_k400_f302660347.pyth'
                    if not os.path.exists(ckpt_path):
                        ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth')
                    else:
                        ckpt = torch.load(ckpt_path)
                    msg = self.model.load_state_dict(ckpt['model_state'],strict=False)
                    # print(msg)
            elif self.model_name=='videoswin':
                if self.size=='base':
                    ckpt_path = home_dir + '/.cache/torch/hub/checkpoints/swin_base_patch244_window1677_sthv2.pth' 
                    ckpt = torch.load(ckpt_path)
                    state_dict = OrderedDict()
                    for k,v in ckpt['state_dict'].items():
                        if 'backbone' in k:
                            name = k[9:]
                            state_dict[name] = v
                    msg = self.model.load_state_dict(state_dict,strict=False)
                    # print(msg)
                else:
                    pass
        else:
            if ckpt_dir is not None:
                ckpt = torch.load(ckpt_dir)
                state_dict = OrderedDict()
                for k,v in ckpt['state_dict'].items():
                    if 'model' in k:
                        name = k[6:]
                        state_dict[name] = v
                msg = self.model.load_state_dict(state_dict,strict=False)
                print('CKPT WEIGHTS', msg)

    def __init__weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m,nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            torch.nn.init.constant_(m.bias,0)
            torch.nn.init.constant_(m.weight,1.0)

    def forward_step(self,x):
        if self.model_name=='mvitv2':
            x = x.transpose(1,2).unsqueeze(0)
            x = self.model(x)
            return x
        else:
            x = x.transpose(1,2)
            x = self.model(x)
            return x


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = batch["imgs"]
        y = batch["phase_trajectory"]
        loss = 0
        y_hat = self.forward_step(x)
        loss += F.cross_entropy(
            y_hat, y[:, -1, :].squeeze().argmax(dim=-1).type(torch.cuda.LongTensor)
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def on_validation_start(self) -> None:
        # define cm
        self.cm = torch.zeros(self.n_class, self.n_class)

    def validation_step(self, batch, batch_idx):
        x = batch["imgs"]
        y = batch["phase_trajectory"]
        loss = 0
        y_hat = self.forward_step(x)
        loss += F.cross_entropy(
            y_hat, y[:, -1, :].squeeze().argmax(dim=-1).type(torch.cuda.LongTensor)
        )
        for idx_batch in range(y.shape[0]):
            gt = y[idx_batch, -1].argmax(dim=-1)
            est = y_hat[idx_batch].argmax(dim=-1)
            self.cm[int(gt), int(est)] += 1.0
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def on_validation_end(self) -> None:
        cm = self.cm.detach().cpu().numpy()
        accuracy = np.divide(
            cm.diagonal(),
            cm.sum(axis=0),
            out=np.zeros_like(cm.diagonal()),
            where=cm.sum(axis=0) != 0,
        )
        accuracy[np.isnan(accuracy)] = 0.0
        for idx, class_name in enumerate(self.class_names):
            print(class_name + " :" + str(accuracy[idx]))
        accuracy_mean = cm.diagonal().sum() / cm.flatten().sum()
        print("Overall" + " :" + str(accuracy_mean))

    def configure_optimizers(self,optimizer='adam'):
        optimizer = build_optimizer(
            hparams={'opt':'adamw','lr':1.0e-5,'weight_decay':0.05},
            model=self.model,
            is_pretrain=True
        )
        lr_scheduler = get_cosine_schedule_warmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=50,
            base_lr=1e-5,
            min_lr=1e-7,
        )
        return [optimizer], [lr_scheduler]
        '''
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=1.0e-5,
                betas=(0.9,0.999),
                weight_decay=0.01
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=1.0e-5,
                momentum=0.9,
                weight_decay=0.01
            )
        scheduler = get_cosine_schedule_warmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=30,
            base_lr=1.0e-4,
            min_lr=1.0e-6,
            last_epoch=-1,
            objective=None,
        )
        return [optimizer],[scheduler]
        '''

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
    # autoencoder = TemporalTrainer(class_names=train.class_names,model_name=args.model,size=args.model_size,pretrain=True,ckpt_dir=None)
    autoencoder = TemporalTrainer(class_names=train.class_names,model_name=args.model,size=args.model_size,pretrain=False,ckpt_dir=args.ckpt_dir)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        strategy='ddp',
        num_nodes=1,
        check_val_every_n_epoch=1,
        max_epochs=args.num_epochs,
        default_root_dir=args.log_dir,
        # accumulate_grad_batches=128,
        accumulate_grad_batches=64,	# MViT setting
        precision=16,
    )
    trainer.fit(autoencoder, dataloader_train, dataloader_test, ckpt_path=args.ckpt_dir)
    # trainer.fit(autoencoder, dataloader_train, dataloader_test)
