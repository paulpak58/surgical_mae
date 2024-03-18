import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import DataParallel
import torch.nn.functional as F
import os
from PIL import Image
import time
import pickle
import numpy as np
import argparse
from torchvision.transforms import Lambda
from NLBlock_MutiConv6_3 import NLBlock
from NLBlock_MutiConv6_3 import TimeConv

import random as random
from transformer2_3_1 import Transformer2_3_1
import mstcn
# import ..mstcn


out_features = 7
# num_workers = 4
workers = 4
# batch_size = 1
test_batch_size = 30
mstcn_causal_conv = True
learning_rate = 1e-3
min_epochs = 12
max_epochs = 25
mstcn_layers = 8
mstcn_f_maps = 32
mstcn_f_dim= 2048
mstcn_stages = 2
sequence_length = 30
# sequence_length = 1
load_exist_LFB = False
crop_type = 1
model_name = 'trans_svnet'

seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

print('gpu             : ', device)
print('sequence length : {:6d}'.format(sequence_length))
print('test batch size : {:6d}'.format(test_batch_size))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('name of this model: {:s}'.format(model_name))  # so we can store all result in the same file
# print('Result store path: {:s}'.format(model_pure_name))



def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels[:, 0]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels = self.file_labels[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels, index

    def __len__(self):
        return len(self.file_paths)


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc_c = nn.Linear(512, 7)
        self.fc_h_c = nn.Linear(1024, 512)
        self.nl_block = NLBlock()
        self.time_conv = TimeConv()

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc_c.weight)
        init.xavier_uniform_(self.fc_h_c.weight)

    def forward(self, x, long_feature=None):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = y[sequence_length - 1::sequence_length]

        Lt = self.time_conv(long_feature)

        y_1 = self.nl_block(y, Lt)
        y = torch.cat([y, y_1], dim=1)
        y = self.fc_h_c(y)
        y = F.relu(y)
        y = self.fc_c(y)
        return y


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        #self.lstm = nn.Linear(2048, 7)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, 7))
        #self.dropout = nn.Dropout(p=0.2)
        #self.relu = nn.ReLU()

        #init.xavier_normal_(self.lstm.weight)
        #init.xavier_normal_(self.lstm.all_weights[0][1])
        #init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        # print(f'shape: {x.shape}')
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        #self.lstm.flatten_parameters()
        #y = self.relu(self.lstm(x))
        #y = y.contiguous().view(-1, 256)
        #y = self.dropout(y)
        #x = self.fc(x)
        return x

    
class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q

        self.transformer = Transformer2_3_1(d_model=out_features, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                        d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q = sequence_length)
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)


    def forward(self, x, long_feature):
        out_features = x.transpose(1,2)
        inputs = []
        for i in range(out_features.size(1)):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.num_classes)).cuda()
                input = torch.cat([input, out_features[:, 0:i+1]], dim=1)
            else:
                input = out_features[:, i-self.len_q+1:i+1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(long_feature).transpose(0,1))
        output = self.transformer(inputs, feas)
        #output = output.transpose(1,2)
        #output = self.fc(output)
        return output


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
        
        # for k in range(LFB_length):
        for k in range(sequence_length):
            LFB_index = (start_index_list[j] - k - 1)
            if int(LFB_index) in dict_start_idx_LFB:                
                LFB_index = dict_start_idx_LFB[int(LFB_index)]
                long_feature_each.append(lfb[LFB_index])
                last_LFB_index_no_empty = LFB_index
            else:
                long_feature_each.append(lfb[last_LFB_index_no_empty])
                
        long_feature.append(long_feature_each)
    return long_feature

def get_test_data(data_path):
    with open(data_path, 'rb') as f:
        test_paths_labels = pickle.load(f)

    test_paths = test_paths_labels[0]
    test_labels = test_paths_labels[1]
    test_num_each = test_paths_labels[2]

    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    test_labels = np.asarray(test_labels, dtype=np.int64)

    test_transforms = None
    global crop_type
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
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return test_dataset, test_num_each


# TODO
# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)




# Long Term Feature bank
# g_LFB_test = np.zeros(shape=(0, 512))
g_LFB_test = np.zeros(shape=(0, 2048))

def test_model(test_dataset, test_num_each):
    global sequence_length
    num_test = len(test_dataset)
    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)
    test_useful_start_idx_LFB = get_useful_start_idx_LFB(sequence_length, test_num_each)

    num_test_we_use = len(test_useful_start_idx)
    num_test_we_use_LFB = len(test_useful_start_idx_LFB)

    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]
    test_we_use_start_idx_LFB = test_useful_start_idx_LFB[0:num_test_we_use_LFB]

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_test_all = len(test_idx)

    test_idx_LFB = []
    for i in range(num_test_we_use_LFB):
        for j in range(sequence_length):
            test_idx_LFB.append(test_we_use_start_idx_LFB[i] + j)

    dict_index, dict_value = zip(*list(enumerate(test_we_use_start_idx_LFB)))
    dict_test_start_idx_LFB = dict(zip(dict_value, dict_index))

    print('num test start idx : {:6d}'.format(len(test_useful_start_idx)))
    print('last idx test start: {:6d}'.format(test_useful_start_idx[-1]))
    print('num of test dataset: {:6d}'.format(num_test))
    print('num of test we use : {:6d}'.format(num_test_we_use))
    print('num of all test use: {:6d}'.format(num_test_all))
# TODO sampler

    global g_LFB_test
    global test_batch_size
    global workers
    global out_features
    global load_exist_LFB

    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             sampler=SeqSampler(test_dataset, test_idx),
                             num_workers=workers)

    print("loading features!>.........")

    if not load_exist_LFB:

        test_feature_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            sampler=SeqSampler(test_dataset, test_idx_LFB),
            num_workers=workers,
            pin_memory=False
        )

        # model_LFB = resnet_lstm_LFB()
        model_LFB = resnet_lstm()

        # model_LFB.load_state_dict(torch.load("./LFB/FBmodel/lstm_epoch_12_length_10_opt_0_mulopt_1_flip_1_crop_1_batch_400_train_9989_val_8839.pth"), strict=False)
        model_LFB.load_state_dict(torch.load("/home/ppak/surgical_adventure/src/Trans-SVNet/best_model/emd_lr5e-4/resnetfc_ce_epoch_15_length_1_opt_0_mulopt_1_flip_1_crop_1_batch_100_train_9946_val_8404_test_7961.pth"), strict=False)

        global use_gpu
        global device
        if use_gpu:
            model_LFB = DataParallel(model_LFB)
            model_LFB.to(device)

        for params in model_LFB.parameters():
            params.requires_grad = False

        model_LFB.eval()

        with torch.no_grad():

            for data in test_feature_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                # print("inputs shape:", inputs.shape)
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()

                g_LFB_test = np.concatenate((g_LFB_test, outputs_feature), axis=0)

                print("test feature length:", len(g_LFB_test))

        print("finish!")
        g_LFB_test = np.array(g_LFB_test)

        # with open("./LFB/g_LFB_test.pkl", 'wb') as f:
        with open("/home/ppak/surgical_adventure/src/Trans-SVNet/eval/python/LFB/g_LFB_test_2048_length30.pkl", 'wb') as f:
            pickle.dump(g_LFB_test, f)

    else:
        # with open("./LFB/g_LFB_test.pkl", 'rb') as f:
        with open("/home/ppak/surgical_adventure/src/Trans-SVNet/eval/python/LFB/g_LFB_test_2048_length30.pkl", 'rb') as f:
            g_LFB_test = pickle.load(f)

        print("load completed")

    print("g_LFB_test shape:", g_LFB_test.shape)

    torch.cuda.empty_cache()


    model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
    model_path = "/home/ppak/surgical_adventure/src/Trans-SVNet/best_model/TeCNO/"
    model_saved_name = 'TeCNO50_epoch_6_train_9935_val_8924_test_8603'
    model1 = Transformer(mstcn_f_maps, mstcn_f_dim, out_features, sequence_length)
    model_path1 = "/home/ppak/surgical_adventure/src/Trans-SVNet/best_model/TeCNO/"
    model_saved_name1 = 'TeCNO50_trans1_3_5_1_length_30_epoch_0_train_8769_val_9054'

    model.load_state_dict(torch.load(model_path+model_saved_name+'.pth'))
    model1.load_state_dict(torch.load(model_path1+model_saved_name1+'.pth'))
    model.cuda()
    model.eval()
    model1.cuda()
    model1.eval()

    '''
    model = resnet_lstm()
    print(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")  
    model.load_state_dict(torch.load(model_name))
    model = DataParallel(model)
    '''

    if use_gpu:
        model.to(device)
    # 应该可以直接多gpu计算
    # model = model.module            #要测试一下
    criterion = nn.CrossEntropyLoss(size_average=False)

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []
    all_preds_score = []


    with torch.no_grad():

        for data in test_loader:
            
            # 释放显存
            torch.cuda.empty_cache()            
            
            if use_gpu:
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels[(sequence_length - 1)::sequence_length]
            else:
                inputs, labels = data
                labels = labels[(sequence_length - 1)::sequence_length]

            start_index_list = data[2]
            start_index_list = start_index_list[0::sequence_length]
            long_feature = get_long_feature(start_index_list=start_index_list,
                                            dict_start_idx_LFB=dict_test_start_idx_LFB,
                                            lfb=g_LFB_test)
            long_feature = torch.Tensor(long_feature).to(device)

            # inputs = inputs.view(-1, sequence_length, 3, 224, 224)

            # outputs = model.forward(inputs, long_feature=long_feature)
            video_fe = long_feature.transpose(2,1)
            print(video_fe.shape)
            out_features = model.forward(video_fe)[-1]
            out_features = out_features.squeeze(1)
            p_classes1 = model1(out_features.detach(), long_feature)
            outputs = p_classes1.squeeze(1)

            # outputs = outputs[sequence_length - 1::sequence_length]
            # Sm = nn.Softmax()
            # outputs = Sm(outputs)
            possibility, preds = torch.max(outputs.data, 1)
            print("possibility:",possibility)

            for i in range(len(preds)):
                all_preds.append(preds[i].data.cpu())
            for i in range(len(possibility)):
                all_preds_score.append(possibility[i].data.cpu())
            print("all_preds length:",len(all_preds))
            print("all_preds_score length:",len(all_preds_score)) 
            loss = criterion(outputs, labels)
            # TODO 和batchsize相关
            # test_loss += loss.data[0]/test_loss += loss.data.item()
            print("preds:",preds.data.cpu())
            print("labels:",labels.data.cpu())

            test_loss += loss.data.item()
            test_corrects += torch.sum(preds == labels.data)
            print("test_corrects:",test_corrects)

    test_elapsed_time = time.time() - test_start_time
    test_accuracy = float(test_corrects) / float(num_test_we_use)
    test_average_loss = test_loss / num_test_we_use

    print('type of all_preds:', type(all_preds))
    print('leng of all preds:', len(all_preds))
    save_test = int("{:4.0f}".format(test_accuracy * 10000))
    pred_name = model_name + '_test_' + str(save_test) + '_crop_' + str(crop_type) + '_length30' + '.pkl'
    pred_score_name = model_name + '_test_' + str(save_test) + '_crop_' + str(crop_type) +'_score'+ '_length30' + '.pkl'

    with open(pred_name, 'wb') as f:
        pickle.dump(all_preds, f)
    with open(pred_score_name, 'wb') as f:
        pickle.dump(all_preds_score, f)
    print('test elapsed: {:2.0f}m{:2.0f}s'
          ' test loss: {:4.4f}'
          ' test accu: {:.4f}'
          .format(test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_average_loss, test_accuracy))


print()


def main():
    test_dataset, test_num_each = get_test_data(
        '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/python/test_paths_labels.pkl')

    test_model(test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
