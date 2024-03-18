import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
import numpy as np
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    ViTMAEConfig,
    ViTMAEModel,
    ViTForImageClassification
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import pickle
from transforms import RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Cholec80Dataset

sequence_length = 1
train_batch_size = val_batch_size = 1
epochs = 25
workers = 4
use_flip = 1
crop_type = 1
load_exist_LFB = False
LFB_length=40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def get_data(data_path):
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

sig_f = nn.Sigmoid()


def train_model(train_dataset, train_num_each, val_dataset, val_num_each, save_preds, output_dir=None):
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

    #    np.random.seed(0)
    # np.random.shuffle(train_we_use_start_idx)
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

    dict_index, dict_value = zip(*list(enumerate(train_we_use_start_idx_80_LFB)))
    dict_train_start_idx_LFB = dict(zip(dict_value, dict_index))

    dict_index, dict_value = zip(*list(enumerate(val_we_use_start_idx_LFB)))
    dict_val_start_idx_LFB = dict(zip(dict_value, dict_index))

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)
    num_test_all = len(test_idx)

    print('num train start idx 80: {:6d}'.format(len(train_useful_start_idx_80)))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num of all valid use: {:6d}'.format(num_val_all))
    print('num of all test use: {:6d}'.format(num_test_all))
    print('num of all train LFB use: {:6d}'.format(len(train_idx_LFB)))
    print('num of all valid LFB use: {:6d}'.format(len(val_idx_LFB)))

    if not load_exist_LFB:

        train_feature_loader = DataLoader(
            train_dataset_80_LFB,
            batch_size=val_batch_size,
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
        print(f'Length of train_feature_loader: {len(train_feature_loader)}')
        print(f'Length of val_feature_loader: {len(val_feature_loader)}')
        print(f'Length of test_feature_loader: {len(test_feature_loader)}')

        model_mae = get_model()
        if torch.cuda.is_available():
            # model_mae = DataParallel(model_mae)
            model_mae.to(device)
        for params in model_mae.parameters():
            params.requires_grad = False
        # print(model_mae)
        model_mae.eval()
        print(f'MAE {model_mae}')

        import time
        test_progress = 0
        test_corrects_phase = 0
        test_all_preds_phase = []
        test_all_labels_phase = []
        test_acc_each_video = []
        test_start_time = time.time()
        num_labels = 0
        with torch.no_grad():
            for data in test_feature_loader:
                if torch.cuda.is_available():
                    inputs, labels_phase = data["imgs"].to(device), data["label"].to(device)
                else:
                    inputs, labels_phase = data["imgs"], data["label"]
                out_features = model_mae.forward(inputs)[-1]
                print(f'out_features: {out_features.shape}')
                
                # out_features = out_features.squeeze()
                # loss = criterion_phase1(out_features, labels_phase[0])
                _, preds_phase = torch.max(out_features, 1)
                test_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                test_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / len(labels_phase))
                num_labels += len(labels_phase)

                for j in range(len(preds_phase)):
                    test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
                for j in range(len(labels_phase)):
                    test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

                test_progress += 1
                if test_progress * val_batch_size >= len(test_feature_loader):
                    percent = 100.0
                    print('Test progress: %s [%d/%d]' % (str(percent) + '%', len(test_feature_loader),
                                                        len(test_feature_loader)), end='\n')
                else:
                    percent = round(test_progress * val_batch_size / len(test_feature_loader) * 100, 2)
                    print('Test progress: %s [%d/%d]' % (
                    str(percent) + '%', test_progress * val_batch_size, len(test_feature_loader)),)

        test_accuracy_phase2 = float(test_corrects_phase) / num_labels
        test_acc_video = np.mean(test_acc_each_video)
        test_elapsed_time = time.time() - test_start_time
        print('Test accuracy video: {:.4f}'.format(test_acc_video))
        print('Test elapsed time: {:.4f}'.format(test_elapsed_time))
        print("finish!")
        if save_preds:
            if not os.path.exists("./eval/vitmae-cholec80-predictions"):
                os.makedirs("./eval/vitmae-cholec80-predictions")
            with open("./eval/vitmae-cholec80-predictions/mae_test_labels.pkl", 'wb') as f:
                pickle.dump(test_all_preds_phase, f)


logger = logging.getLogger(__name__)
check_min_version("4.28.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(default="cifar10", metadata={"help": "Name of a dataset from the datasets package"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    image_column_name: Optional[str] = field(default=None, metadata={"help": "The column name of the images in the files."})
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(default=0.15, metadata={"help": "Percent to split off of train for validation."})
    max_train_samples: Optional[int] = field(default=None,metadata={"help": ("For debugging purposes or quicker training, truncate the number of training examples to this ""value if set.")})
    max_eval_samples: Optional[int] = field(default=None,metadata={"help": ("For debugging purposes or quicker training, truncate the number of evaluation examples to this ""value if set.")})
    # remove_unusued_columns: Optional[bool] = field(default=False, metadata={"help": "Remove columns not required by the model."})
    def __post_init__(self):
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """
    model_name_or_path: str = field(default=None,
        metadata={"help": ("The model checkpoint for weights initialization.Don't set if you want to train a model from scratch.")})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"})
    config_overrides: Optional[str] = field(default=None,metadata={"help": ("Override some existing default config settings when a model is trained from scratch. Example: ""n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})
    model_revision: str = field(default="main",metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(default=False,metadata={"help": ("Will use the token generated when running `huggingface-cli login` (necessary to use this script ""with private models).")})
    mask_ratio: float = field(default=0.75, metadata={"help": "The ratio of the number of masked tokens in the input sequence."})
    norm_pix_loss: bool = field(default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(default=1e-3, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."})
    label_path: str = field(default=None, metadata={"help": "Path to the label file."})
    save_preds: bool = field(default=True, metadata={"help": "Save train, val, and test predictions in pkl fil."})

def get_model():

    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "num_labels": 7
    }
    if model_args.config_name:
        config = ViTMAEConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ViTMAEConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise Exception('Load a pretrained model for finetuning')
    config.update( {"norm_pix_loss": model_args.norm_pix_loss})

    # create model
    if model_args.model_name_or_path:
        model = ViTForImageClassification.from_pretrained(
        # model = ViTMAEModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = ViTMAEModel(config)
    return model

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        _, _, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        _, _, training_args = parser.parse_args_into_dataclasses()
    train_dataset_80, train_num_each_80, \
    val_dataset, val_num_each, test_dataset, test_num_each = get_data(training_args.label_path)
    # val_dataset, val_num_each, test_dataset, test_num_each = get_data(args.label_path)
    train_model((train_dataset_80),
                (train_num_each_80),
                (val_dataset, test_dataset),
                (val_num_each, test_num_each),
                training_args.save_preds,
                training_args.output_dir
                )


if __name__ == "__main__":
    main()

print('Done')
print()