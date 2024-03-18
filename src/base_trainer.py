import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import csv
import os
import numpy as np
import pytorch_lightning as pl
# from surgical_rules.model import CNN_LSTM

from data_utils.protobuf_dataset import process_data_directory_surgery
from misc.base_params import parse_arguments

# Import Models
from mvitv2 import MViT,build_cfg_mvitv2
from swintransformer import SwinTransformer3D,build_cfg_swin

class TemporalTrainerEval(pl.LightningModule):

    def __init__(self, class_names = [], model_name = None, size = 'base', log_dir = './eval_stats', write_results_to_file = True):
        super().__init__()
        if model_name=='mvitv2':
            if size=='base':
                cfg = build_cfg_mvitv2(size='base')
                self.model = MViT(cfg)
            else:
                cfg = build_cfg_mvitv2(size='small')
                self.model = MViT(cfg)
        elif model_name=='videoswin':
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
                # self.embed_dim = list(self.model.children())[-1].normalized_shape[0]
            else:   # size=='small'
                pass
        else:
            self.model = CNN_LSTM(n_class=7)
        self.class_names = class_names
        self.n_class = len(self.class_names)
        self.model_name = model_name
        self.log_dir = log_dir
        # create stat csv file to save all the intermidiate stats
        self.stat_file = open(os.path.join(log_dir, 'stats.csv'), 'w')
        self.stat_writer = csv.writer(self.stat_file)
        self.write_results_to_file = write_results_to_file

    def forward_step(self,x):
        if self.model_name=='mvitv2':
            x = x.transpose(1,2).unsqueeze(0)
            x = self.model(x)
            return x
        else:
            x = x.transpose(1,2)
            x = self.model(x)
            return x


    def on_train_start(self) -> None:
        stat_header = ['iter', 'metric']
        stat_header.extend(self.class_names) 
        stat_header.append('mean')
        stat_header.append('accuracy')
        self.stat_writer.writerow(stat_header)
        
        
        return super().on_train_start()


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
        self.log('val_loss', loss)
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
        self.stat_writer.writerow(stats)

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
        
        self.stat_writer.writerow(stats)

        self.log('val_accuracy', accuracy_overall)


    def on_test_start(self):
        self.gt = dict()
        self.est = dict()
        self.cm = torch.zeros(self.n_class, self.n_class)
        # create stat csv file to save all the inference stats
        self.test_stat_file = open(os.path.join(self.log_dir, 'test_stats.csv'), 'w')
        self.test_stat_writer = csv.writer(self.test_stat_file)
        stat_header = ['iter', 'metric']
        stat_header.extend(self.class_names) 
        stat_header.append('mean')
        stat_header.append('accuracy')
        self.test_stat_writer.writerow(stat_header)


    def test_step(self, batch, batch_idx):
        # x = batch[0]
        # y = batch[1][:,:,0]
        x = batch["imgs"]
        y = batch["phase_trajectory"]
        self.seq_len = y.shape[1]
        video_ids = batch['video_name']

        loss = 0
        y_hat = self.forward_step(x)

        loss += F.cross_entropy(y_hat, y[:,-1, :])
        for idx_batch in range(y.shape[0]):
            video_id = video_ids[idx_batch]
            if video_id not in self.est.keys():
                self.est[video_id] = []
                self.gt[video_id] = []
            gt = y[idx_batch, -1].argmax(dim=-1)
            est = y_hat[idx_batch,:].argmax(dim=-1)
            self.cm[int(gt), int(est)] += 1.0
            self.gt[video_id].append(int(gt))
            self.est[video_id].append(int(est))

        self.log('test_loss', loss)
        self.log('test_cm', self.cm)        
        return loss

    def test_epoch_end(self, test_step_outputs):
        cm = self.cm.detach().cpu().numpy()
        print("confusion matrix:")
        print(cm.astype(int))
        print("Recall:")
        stats = [self.current_epoch, "Recall"]
        accuracy = cm.diagonal() / cm.sum(axis=-1)
        accuracy[np.isnan(accuracy)] = 0.0
        for idx, class_name in  enumerate(self.class_names):
            print(class_name + ':' + str(accuracy[idx]))
            stats.append(accuracy[idx])
        accuracy_mean = accuracy[accuracy != 0].mean()
        print('average recall' + ' :' + str(accuracy_mean) + '\n')
        stats.append(accuracy_mean)
        accuracy_overall = cm.diagonal().sum() / cm.sum(axis=-1).sum()
        print('average accuracy' + ' :' + str(accuracy_overall) + '\n')
        stats.append(accuracy_overall)
        self.test_stat_writer.writerow(stats)

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
        self.test_stat_writer.writerow(stats)

        if self.write_results_to_file:
            self.write_results(estimates=self.est,
                               class_names=self.class_names,
                               res_dir=os.path.join(self.log_dir, 'results_txt'))

        self.log("test_accuracy", accuracy_overall)
        return {'gt':self.gt, 'est':self.est}

    def write_results(self,
                      estimates=None,
                      class_names=None,
                      res_dir='./results_txt/camma/',
                      estimate_fps=1,
                      write_fps=25):
        '''
        Write the formatted estimation result into txt files
        :param estimate_list:
        :param class_names:
        :param res_dir:
        :param video_name_list_filename:
        :param estimate_fps: the estimation frame rate
        :param write_fps: the output frame rate
        :return:
        '''
        time_ratio = int(write_fps / estimate_fps)
        os.makedirs(res_dir, exist_ok=True)

        for video_idx in estimates.keys():
            video_name = video_idx
            est = estimates[video_idx]
            write_file_name = os.path.join(res_dir, video_name + '.txt')
            f_out = open(write_file_name, "w")
            f_out.write('Frame\tPhase\n')
            # copy the first few samples in sequence 
            pad_len = self.seq_len -1
            for t in range(pad_len):
                for idx1 in range(time_ratio):
                    t_idx = t * time_ratio + idx1
                    f_out.write(str(t_idx) + '\t' + class_names[est[0]] + '\n')

            # write to file
            for t in range(len(est)):
                for idx1 in range(time_ratio):
                    t_idx = (t + pad_len) * time_ratio + idx1
                    f_out.write(str(t_idx) + '\t' + class_names[est[t]] + '\n')

            f_out.close()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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
    autoencoder = TemporalTrainerEval(class_names=train.class_names,model_name=args.model,size=args.model_size,log_dir=args.log_dir)
    '''
    if os.path.exists(args.ckpt_dir):
        autoencoder = autoencoder.load_from_checkpoint(args.ckpt_dir)
        print('Weights loaded')
    else:
        print('Weights not found')
    '''
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        strategy='ddp',
        num_nodes=1,
        check_val_every_n_epoch=1,
        max_epochs=args.num_epochs,
        default_root_dir=args.log_dir,
        accumulate_grad_batches=128,
        # accumulate_grad_batches=64,	# MViT setting
        precision=16,
        resume_from_checkpoint=args.ckpt_dir
    )
    trainer.test(autoencoder,dataloaders=dataloader_test,ckpt_path=args.ckpt_dir)
