import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from misc.base_params import parse_arguments
from data_utils.protobuf_dataset import process_data_directory_surgery
np.random.seed(0)

def inverse_eigen(imgs, val_ceil=16,nrow=8,ncol=1,viz=True):
    val_floors = [4,8,16,32,64]
    for i in range(len(val_floors)):
        (U,S,V) = torch.linalg.svd(imgs,full_matrices=False)
        print('U,S,V shapes: ',U.shape,S.shape,V.shape)
        Ai = U[:,:val_ceil]@torch.diag(S[:val_ceil])@V[:val_ceil]
        inverse_Ai = U[:,val_floors[i]:]@torch.diag(S[val_floors[i]:])@V[val_floors[i]:]

        # Set to True if you want to visualize inverse-eigen decomposition frames
        if viz:
            fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=False, sharey=False)
            img_grid = torchvision.utils.make_grid(
                (imgs.view(args.temporal_length,3,224,224))[(i*nrow):(i+1)*nrow,:,:,:], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )
            img_rec_grid = torchvision.utils.make_grid(
                (Ai.view(args.temporal_length,3,224,224))[(i*nrow):(i+1)*nrow,:,:,:], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )
            img_rev_rec_grid = torchvision.utils.make_grid(
                (inverse_Ai.view(args.temporal_length,3,224,224))[(i*nrow):(i+1)*nrow,:,:,:], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )
            ax[0].axis('off')
            ax[0].imshow(img_grid.permute(1,2,0))
            ax[1].axis('off')
            ax[1].imshow(img_rec_grid.permute(1,2,0))
            ax[2].axis('off')
            ax[2].imshow(img_rev_rec_grid.permute(1,2,0))
            plt.savefig(f"photos/eigen_test_val{val_floors[i]}.png")
            # plt.show()

def eigen_mask(inverse_Ai):
    seq = ((inverse_Ai.view(args.temporal_length,3,224,224))[0]).flatten()     # Take first image in test video

    # threshold = 0.50*torch.max(seq)
    # x = sum(1 for i in seq if i >= threshold)
    # print(x,len(seq)-x)

    mean_seq = torch.mean(seq,dim=-1)
    print(mean_seq.shape,'mean seq')

    keep = math.floor(0.50*len(seq))
    indices = torch.argsort(seq)
    recover_indices = torch.argsort(indices)
    sorted_seq = seq[indices]
    mask = sorted_seq.clone()
    mask[keep:] = 0
    print(mask[:keep])

    mask = mask[recover_indices].reshape(3,224,224)
    
    # mask = (seq>=threshold).float().reshape(3,224,224)
    seq = seq.reshape(3,224,224)
    seq_grid = torchvision.utils.make_grid(
        (inverse_Ai.view(args.temporal_length,3,224,224))[0:nrow], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
    )
    mask_grid = torchvision.utils.make_grid(
        mask.unsqueeze(0),nrow=1,ncol=1,normalize=True,pad_value=0
    )
    plt.figure()
    plt.imshow(seq_grid.permute(1,2,0))
    plt.figure()
    plt.imshow(mask_grid.permute(1,2,0))
    plt.savefig("eigen_mask.png")

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
    data_iter = iter(dataloader_test)
    imgs = next(data_iter)["imgs"]
    imgs = imgs[0].flatten(1)
    print("Imgs shape", imgs.shape)
    inverse_eigen(imgs)