# Some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet
# Revied by: Paul Pak 

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse

# Model Imports
from transforms import RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Cholec80Dataset


class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
    

def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_useful_start_idx_LFB(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_long_feature(start_index_list, dict_start_idx_LFB, lfb):
    long_feature = []
    for j in range(len(start_index_list)):
        long_feature_each = []

        # 上一个存在feature的index
        last_LFB_index_no_empty = dict_start_idx_LFB[int(start_index_list[j])]

        for k in range(LFB_length):
            LFB_index = (start_index_list[j] - k - 1)
            if int(LFB_index) in dict_start_idx_LFB:
                LFB_index = dict_start_idx_LFB[int(LFB_index)]
                long_feature_each.append(lfb[LFB_index])
                last_LFB_index_no_empty = LFB_index
            else:
                long_feature_each.append(lfb[last_LFB_index_no_empty])

        long_feature.append(long_feature_each)
    return long_feature


def get_data(data_path, use_flip, crop_type, sequence_length):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    test_paths_80 = train_test_paths_labels[6]
    test_labels_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]


    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))

    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.int64)

    train_transforms = None
    test_transforms = None


    # Train Data Augmentations
    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(size=224, sequence_length=sequence_length),
            RandomHorizontalFlip(sequence_length=sequence_length),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(size=224, sequence_length=sequence_length),
            ColorJitter(sequence_length=sequence_length, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(sequence_length=sequence_length),
            RandomRotation(degrees=5, sequence_length=sequence_length),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])

    # Test Data Augmentations
    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])

    # Given data, label, and transforms, create our datasets
    # This assumes that the train_val_paths_labels1.pkl from the Google Drive is being used
    # Since they use relative paths, we replace with user data paths
    parent_path = '/'.join(data_path.split('/')[:-1])
    for i in range(len(train_paths_80)):
        train_paths_80[i] = train_paths_80[i].replace('../..', parent_path)
    for i in range(len(val_paths_80)):
        val_paths_80[i] = val_paths_80[i].replace('../..', parent_path)
    for i in range(len(test_paths_80)):
        test_paths_80[i] = test_paths_80[i].replace('../..', parent_path)

    train_dataset_80 = Cholec80Dataset(train_paths_80, train_labels_80, sequence_length, train_transforms)
    train_dataset_80_LFB = Cholec80Dataset(train_paths_80, train_labels_80, sequence_length, test_transforms)
    val_dataset_80 = Cholec80Dataset(val_paths_80, val_labels_80, sequence_length, test_transforms)
    test_dataset_80 = Cholec80Dataset(test_paths_80, test_labels_80, sequence_length, test_transforms)

    return (train_dataset_80, train_dataset_80_LFB), train_num_each_80, \
           val_dataset_80, val_num_each_80, test_dataset_80, test_num_each_80


def retrieve_dataloaders(
        train_dataset,
        train_num_each,
        val_dataset,
        val_num_each, 
        train_batch_size, 
        val_batch_size,
        sequence_length,
        workers,
):
    # TensorBoard
    # writer = SummaryWriter('runs/non-local/pretrained_lr5e-7_L40_2fc_copy/')

    (train_num_each_80), \
    (val_dataset, test_dataset), \
    (val_num_each, test_num_each) = train_num_each, val_dataset, val_num_each

    (train_dataset_80, train_dataset_80_LFB) = train_dataset

    train_useful_start_idx_80 = get_useful_start_idx(sequence_length, train_num_each_80)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)
    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    train_useful_start_idx_80_LFB = get_useful_start_idx_LFB(sequence_length, train_num_each_80)
    val_useful_start_idx_LFB = get_useful_start_idx_LFB(sequence_length, val_num_each)
    test_useful_start_idx_LFB = get_useful_start_idx_LFB(sequence_length, test_num_each)

    num_train_we_use_80 = len(train_useful_start_idx_80)
    num_val_we_use = len(val_useful_start_idx)
    num_test_we_use = len(test_useful_start_idx)

    num_train_we_use_80_LFB = len(train_useful_start_idx_80_LFB)
    num_val_we_use_LFB = len(val_useful_start_idx_LFB)
    num_test_we_use_LFB = len(test_useful_start_idx_LFB)

    train_we_use_start_idx_80 = train_useful_start_idx_80
    val_we_use_start_idx = val_useful_start_idx
    test_we_use_start_idx = test_useful_start_idx

    train_we_use_start_idx_80_LFB = train_useful_start_idx_80_LFB
    val_we_use_start_idx_LFB = val_useful_start_idx_LFB
    test_we_use_start_idx_LFB = test_useful_start_idx_LFB

    train_idx = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx_80[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    train_idx_LFB = []
    for i in range(num_train_we_use_80_LFB):
        for j in range(sequence_length):
            train_idx_LFB.append(train_we_use_start_idx_80_LFB[i] + j)

    val_idx_LFB = []
    for i in range(num_val_we_use_LFB):
        for j in range(sequence_length):
            val_idx_LFB.append(val_we_use_start_idx_LFB[i] + j)

    test_idx_LFB = []
    for i in range(num_test_we_use_LFB):
        for j in range(sequence_length):
            test_idx_LFB.append(test_we_use_start_idx_LFB[i] + j)


    train_feature_loader = DataLoader(
        train_dataset_80_LFB,
        batch_size=train_batch_size,
        sampler=SeqSampler(train_dataset_80_LFB, train_idx_LFB),
        num_workers=workers,
        pin_memory=False
    )
    val_feature_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx_LFB),
        num_workers=workers,
        pin_memory=False
    )

    test_feature_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(test_dataset, test_idx_LFB),
        num_workers=workers,
        pin_memory=False
    )
    return (train_feature_loader, val_feature_loader, test_feature_loader,
            train_dataset_80_LFB, val_dataset, test_dataset)

def retrieve_datasets(
    sequence_length = 1,
    train_batch_size = 400,
    val_batch_size = 400,
    epochs = 25,
    workers = 8,
    use_flip = 1,
    crop_type = 1,
    LFB_length = 40,
    gpu=True,
    label_path = '/saiil2/paulpak/surgical_ncp/train_val_paths_labels1.pkl',
    return_datasets=True
):
    num_gpu = torch.cuda.device_count()
    use_gpu = (torch.cuda.is_available() and gpu)
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print('############## CONFIGS ##############')
    print('number of gpu   : {:6d}'.format(num_gpu))
    print('sequence length : {:6d}'.format(sequence_length))
    print('train batch size: {:6d}'.format(train_batch_size))
    print('valid batch size: {:6d}'.format(val_batch_size))
    print('num of epochs   : {:6d}'.format(epochs))
    print('num of workers  : {:6d}'.format(workers))
    print('test crop type  : {:6d}'.format(crop_type))
    print('whether to flip : {:6d}'.format(use_flip))
    print('#####################################')


    train_dataset_80, train_num_each_80, \
    val_dataset, val_num_each, test_dataset, test_num_each = get_data(
        label_path,
        use_flip=use_flip,
        crop_type=crop_type,
        sequence_length=sequence_length)
    # val_dataset, val_num_each, test_dataset, test_num_each = get_data('./train_val_paths_labels1.pkl')
    (train_feature_loader, val_feature_loader, test_feature_loader,
     train_dataset, val_dataset, test_dataset) = retrieve_dataloaders(
        (train_dataset_80),
        (train_num_each_80),
        (val_dataset, test_dataset),
        (val_num_each, test_num_each),
        train_batch_size,
        val_batch_size,
        sequence_length,
        workers
    )
    if return_datasets:
        print(f'Length of train dataset: {len(train_dataset)}')
        print(f'Length of val dataset: {len(val_dataset)}')
        print(f'Length of test dataset: {len(test_dataset)}')
        return (train_dataset, val_dataset, test_dataset)
    else:
        print(f'Length of train_feature_loader: {len(train_feature_loader)}')
        print(f'Length of val_feature_loader: {len(val_feature_loader)}')
        print(f'Length of test_feature_loader: {len(test_feature_loader)}')
        return (train_feature_loader, val_feature_loader, test_feature_loader)


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = retrieve_datasets()

    '''
    if run_loaders:
        for data in train_feature_loader:
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            # print(f'Inputs shape: {inputs.shape}')
            # outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()
            # g_LFB_train = np.concatenate((g_LFB_train, outputs_feature), axis=0)
            # print("train feature length:", len(g_LFB_train))

        # model.eval()
        with torch.no_grad():
            for data in val_feature_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                # outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()
                # g_LFB_val = np.concatenate((g_LFB_val, outputs_feature), axis=0)
                # print("val feature length:", len(g_LFB_val))
            for data in test_feature_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                # outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()
                # g_LFB_test = np.concatenate((g_LFB_test, outputs_feature), axis=0)
                # print("test feature length:", len(g_LFB_test))
        print("finish!")
        '''