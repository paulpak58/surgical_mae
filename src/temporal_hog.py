import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
from skimage.feature import hog
from skimage import exposure
from einops import rearrange

np.random.seed(0)

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from misc.base_params import parse_arguments
from data_utils.protobuf_dataset import process_data_directory_surgery


def spatial_hog(img):
    hog_features_r = hog(
        img[:, :, 0],
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        block_norm="L2",
        feature_vector=False,
        )
    hog_features_g = hog(
        img[:, :, 1],
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        block_norm="L2",
        feature_vector=False,
    )
    hog_features_b, hog_img = hog(
        img[:, :, 2],
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        block_norm="L2",
        feature_vector=False,
        visualize=True,
    )
    hog_features = np.concatenate(
        [hog_features_r, hog_features_g, hog_features_b], axis=-1
    )
    hog_features = rearrange(
        hog_features,
        "(ph dh) (pw dw) ch cw c -> ph pw (dh dw ch cw c)",
        ph=14,
        pw=14,
    )  # Patches

    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    return hog_features, hog_img


def temporal_hog(imgs,temporal_scale=False):
    spatial_hog_features = []
    hog_imgs = []

    # Compute each spatial hog across all frames and concatenate them
    images = torch.permute(imgs, (0, 2, 3, 1))
    for img in images:
        hogs_per_img, hog_img = spatial_hog(img.clone().cpu())
        spatial_hog_features.append(hogs_per_img)
        hog_imgs.append(hog_img)
    hog_features = torch.from_numpy(np.array(spatial_hog_features))
    hog_imgs = torch.from_numpy(np.array(hog_imgs))
    temporal_hog_img = hog_imgs.sum(0)  # Accumulated hog image from the video sequences

    if temporal_scale:
        # Reduce sequence in half by taking average with the adjacent frame
        hog_features_avg = list()
        for i in range(0,hog_features.shape[0]-1,2):
            hog_features_avg.append(((hog_features[i,:]+hog_features[i+1,:])/2).unsqueeze(0))
        hog_features_avg = torch.cat(hog_features_avg,dim=0)
        return hog_imgs, temporal_hog_img,hog_features_avg
    else:
        return hog_imgs, temporal_hog_img, hog_features


def retrieve_spatiotemporal_geysers(
    hog_features, num_frames=128, num_representatives=16, iterative=False
):
    """
    Return the indices of 16 frames deemed the spatiotemporal hot spots
    Iterative, Global Differences
    """

    if iterative:
        representatives = torch.tensor([0,num_frames-1])
        # Iteratively accumulate frames to represent the whole temporal sequence
        num_iterations = int(math.log2(num_frames)) - int(
            math.log2(num_representatives)
        )
        for i in range(1, num_iterations + 1):

            # Add in 2 frames in the initial sequence, then 4 in the subsequences, then 8 in the subsequences, etc.
            num_frames_to_add = 2**i
            sequence_hogs = torch.tensor([])
            representatives = np.sort(representatives)
            print("Iteration ", i, ": ", representatives)
            for j in range(len(representatives) - 1):
                start_frame = representatives[j]
                end_frame = representatives[j + 1]
                subsequence = hog_features[start_frame : end_frame + 1]

                # Return sorted indices with greatest absolute HOG feature difference
                hog_abs = extract_temporal_hot_spots(subsequence)
                sequence_hogs = torch.cat((sequence_hogs, hog_abs))
            indices = torch.argsort(sequence_hogs, descending=True)
            representatives = np.concatenate(
                (representatives, indices[:num_frames_to_add].numpy())
            )
    else:
        # One shot accumulation of frames
        num_frames_to_add = num_representatives - 2
        reps = torch.tensor([0,hog_features.shape[0]-1])
        hog_abs = extract_temporal_hot_spots(hog_features)
        indices = torch.argsort(hog_abs, descending=True)
        reps = torch.cat((reps,indices[:num_frames_to_add]),dim=0)
        return torch.sort(reps)[0]

def extract_temporal_hot_spots(hog_features,global_diff=True):
    """
    hog_features: concatenated 2D spatial hog features from a temporal sequence
    """
    if not global_diff:
        # Local Temporal HOG Metric Evaluations
        hog_features = hog_features.reshape(hog_features.shape[0], -1)
        local_hog_diffs = hog_features.clone().detach()
        for i in range(0, local_hog_diffs.shape[0] - 1):
            local_hog_diffs[i] = local_hog_diffs[i + 1] - local_hog_diffs[i]
        local_hog_diffs[-1] = torch.zeros(1, local_hog_diffs.shape[-1])
        local_hog_abs = torch.mean(torch.abs(local_hog_diffs), dim=-1)
        # print('Mean of local absolute diffs:',local_hog_abs)
        return local_hog_abs
    else:
        # Global Temporal HOG Metric Evaluations
        hog_features = hog_features.reshape(hog_features.shape[0], -1)
        hog_diffs_from_start = hog_features - hog_features[0]
        hog_diffs_from_end = hog_features[-1] - hog_features
        hog_sums = torch.abs(torch.mean(hog_diffs_from_start, dim=-1)) + torch.abs(
            torch.mean(hog_diffs_from_end, dim=-1)
        )
        hog_abs = torch.mean(
            torch.abs(hog_diffs_from_start) + torch.abs(hog_diffs_from_end), dim=-1
        )
        hog_abs[0]=0
        hog_abs[-1]=0
        return hog_abs

def random_sampling(x,num=8,T=128,avg=True):
    # B,T,C,H,W = x.shape
    # pooled_len = int(T/2)-1 if num==8 else T-1
    if avg:     # pooled together
        x_avg = [((x[:,i,:,:,:]+x[:,i+1,:,:,:])/2).unsqueeze(0) for i in range(T-1)]
        x_avg = torch.cat(x_avg,dim=0)
    max_frame = T//2 if avg else T
    context_frames = []
    context_indices = [0,max_frame-1]
    gen = np.random.default_rng()
    samples = gen.choice(np.arange(1,max_frame-1,dtype=int),size=num-2,replace=False)
    context_indices = np.concatenate((context_indices,samples),axis=0)
    context_indices = torch.from_numpy(np.sort(context_indices))

    context_frames = x[:,context_indices.long(),:,:,:]
    # context_frames = x[:,context_indices.long(),:,:,:]
    # context_frames = [ x[:,i,:,:,:].unsqueeze(0) for i in context_indices ]
    # context_frames = torch.cat(context_frames,dim=0)
    return context_indices


def visualize_hog_features(imgs, hog_imgs, temporal_hogs=None, nrow=4, ncol=2):
    """
    Visualize cholec80 hog samples
    """
    hog_imgs = hog_imgs.unsqueeze(2)
    img_grid1 = torchvision.utils.make_grid(
        imgs[0,:nrow,:,:,:], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
    )
    hog_grid1 = torchvision.utils.make_grid(
        hog_imgs[0,:nrow,:,:,:], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
    )

    test_white = torch.full((8,3,224,224),255,dtype=torch.float)

    img_grid2 = torchvision.utils.make_grid(
        test_white, nrow=nrow, ncol=ncol, normalize=True, pad_value=0
    )
    img_grid1 = img_grid1.permute(1, 2, 0)
    hog_grid1 = hog_grid1.permute(1, 2, 0)
    img_grid2 = img_grid2.permute(1, 2, 0)

    # Visualize two batches of cholec80 videos with HOG features of each frame
    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(15, 15))
    axs[0].imshow(img_grid1)
    axs[0].set_axis_off()
    axs[1].imshow(hog_grid1, cmap=plt.cm.gray)
    axs[1].set_axis_off()
    axs[2].imshow(img_grid2)
    axs[2].set_axis_off()
    # plt.show()

    # Can visualize summation of all HOG features
    if temporal_hogs is not None:    
        # Visualize summation of all HG Features
        fig2, axs2 = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))
        axs2[0].imshow(temporal_hogs[0], cmap=plt.cm.gray)
        axs2[0].set_axis_off()
        axs2[1].imshow(temporal_hogs[1], cmap=plt.cm.gray)
        axs2[1].set_axis_off()

        # Plot first and last HOG differences
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 8))
        axes[0].imshow(hog_imgs[0], cmap=plt.cm.gray)
        axes[0].set_axis_off()
        axes[1].imshow(hog_imgs[-1], cmap=plt.cm.gray)
        axes[1].set_axis_off()
        axes[2].imshow(hog_imgs[-1] - hog_imgs[0], cmap=plt.cm.gray)
        axes[2].set_axis_off()
        axes[3].imshow(hog_imgs[-1] + hog_imgs[0], cmap=plt.cm.gray)
        axes[3].set_axis_off()
        plt.show()


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
   
    hogs = []
    hog_features = []
    hogs1,_,hog_features1 = temporal_hog(imgs[0])
    hogs2,_,hog_features2 = temporal_hog(imgs[1])
    hog_features.append(hog_features1.unsqueeze(0))
    hog_features.append(hog_features2.unsqueeze(0))
    hogs.append(hogs1.unsqueeze(0))
    hogs.append(hogs2.unsqueeze(0))
    hogs = torch.cat(hogs,dim=0)
    hog_features = torch.cat(hog_features,dim=0)

    # Set to True if you want to visualize the hog features
    if False:
        visualize_hog_features(imgs,hogs,nrow=8,ncol=2)
    
    ### Video Eigendecomposition ###
    hogs = hogs[0].flatten(1)
    imgs = imgs[0].flatten(1) 
    # svd = torch.linalg.svdvals(hogs)
    # (U,S,V) = torch.pca_lowrank(hogs,niter=2)
    # (Ui,Si,Vi) = torch.pca_lowrank(imgs,niter=2)


    val_ceil = 16
    val_floor = 16
    nrow=8
    ncol=1
    num_iterations=4
    viz = True

    (Usvd,Ssvd,Vsvd) = torch.linalg.svd(imgs,full_matrices=False)
    (Uhog,Shog,Vhog) = torch.linalg.svd(hogs,full_matrices=False)
    Ahog = Uhog[:,:val_ceil]@torch.diag(Shog[:val_ceil])@Vhog[:val_ceil]
    inverse_Ahog = Uhog[:,val_floor:]@torch.diag(Shog[val_floor:])@Vhog[val_floor:]
    Ai = Usvd[:,:val_ceil]@torch.diag(Ssvd[:val_ceil])@Vsvd[:val_ceil]
    inverse_Ai = Usvd[:,val_floor:]@torch.diag(Ssvd[val_floor:])@Vsvd[val_floor:]

    # Set to True if you want to visualize inverse-eigen decomposition frames
    if viz:
        for i in range(num_iterations):
            fig, ax = plt.subplots(6, 1, figsize=(8, 8), sharex=False, sharey=False)
            img_grid = torchvision.utils.make_grid(
                (imgs.view(args.temporal_length,3,224,224))[(i*nrow):(i+1)*nrow,:,:,:], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )
            img_rec_grid = torchvision.utils.make_grid(
                (Ai.view(args.temporal_length,3,224,224))[(i*nrow):(i+1)*nrow,:,:,:], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )
            img_rev_rec_grid = torchvision.utils.make_grid(
                (inverse_Ai.view(args.temporal_length,3,224,224))[(i*nrow):(i+1)*nrow,:,:,:], nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )
            hog_grid = torchvision.utils.make_grid(
                ((hogs.view(args.temporal_length,224,224))[(i*nrow):(i+1)*nrow,:,:]).unsqueeze(1), nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )
            hog_rec_grid = torchvision.utils.make_grid(
                ((Ahog.view(args.temporal_length,224,224))[(i*nrow):(i+1)*nrow,:,:]).unsqueeze(1), nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )
            hog_reverse_rec_grid = torchvision.utils.make_grid(
                ((inverse_Ahog.view(args.temporal_length,224,224))[(i*nrow):(i+1)*nrow,:,:].unsqueeze(1)), nrow=nrow, ncol=ncol, normalize=True, pad_value=0
            )

            ax[0].axis('off')
            ax[0].imshow(img_grid.permute(1,2,0))
            ax[1].axis('off')
            ax[1].imshow(img_rec_grid.permute(1,2,0))
            ax[2].axis('off')
            ax[2].imshow(img_rev_rec_grid.permute(1,2,0))
            ax[3].axis('off')
            ax[3].imshow(hog_grid.permute(1,2,0),cmap=plt.cm.gray)
            ax[4].axis('off')
            ax[4].imshow(hog_rec_grid.permute(1,2,0),cmap=plt.cm.gray)
            ax[5].axis('off')
            ax[5].imshow(hog_reverse_rec_grid.permute(1,2,0),cmap=plt.cm.gray)
        plt.savefig("eigen_photo.png")
        # plt.show()

    ###### Masking Eigen Reconstructions ######
    '''
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
    plt.show()
    '''
