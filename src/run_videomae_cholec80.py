#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import logging
import os
import sys
import pickle
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np
# from datasets import load_dataset
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    VideoMAEImageProcessor,
    ViTMAEForPreTraining,
    VideoMAEConfig,
    VideoMAEForPreTraining,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torchvision
from transforms import RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Cholec80DatasetSequence
from accelerate import Accelerator


logger = logging.getLogger(__name__)
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")


@dataclass
class DataTrainingArguments:
    image_column_name: Optional[str] = field(default=None, metadata={"help": "The column name of the images in the files."})
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(default=0.15, metadata={"help": "Percent to split off of train for validation."})
    max_train_samples: Optional[int] = field(default=None,metadata={"help": ("For debugging purposes or quicker training, truncate the number of training examples to this ""value if set.")})
    max_eval_samples: Optional[int] = field(default=None,metadata={"help": ("For debugging purposes or quicker training, truncate the number of evaluation examples to this ""value if set.")})
    def __post_init__(self):
        data_files = {}
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None,
        metadata={"help": ("The model checkpoint for weights initialization.Don't set if you want to train a model from scratch.")})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"})
    config_overrides: Optional[str] = field(default=None,metadata={"help": ("Override some existing default config settings when a model is trained from scratch. Example: ""n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})
    model_revision: str = field(default="main",metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(default=False,metadata={"help": ("Will use the token generated when running `huggingface-cli login` (necessary to use this script ""with private models).")})
    # mask_ratio: float = field(default=0.75, metadata={"help": "The ratio of the number of masked tokens in the input sequence."})
    # norm_pix_loss: bool = field(default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."})


class TubeMaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.
    """

    def __init__(self, num_frames=16, img_size=224, patch_size=16, tubelet_size=2, mask_ratio=0.9):
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.mask_ratio = mask_ratio

    def __call__(self):
        num_patches_per_frame = (self.img_size // self.patch_size) ** 2
        seq_length = (self.num_frames // self.tubelet_size) * num_patches_per_frame
        num_masked = int(self.mask_ratio*seq_length)
        bool_masked_pos = torch.hstack([
            torch.zeros(seq_length - num_masked, dtype=int),
            torch.ones(num_masked, dtype=int)
        ])
        bool_masked_pos = bool_masked_pos[torch.randperm(seq_length)].unsqueeze(0)
        return bool_masked_pos


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(default=1e-3, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."})
    label_path: str = field(default=None, metadata={"help": "Path to the label file."})


image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
def collate_fn(examples):
    pixel_values = torch.stack([example["imgs"] for example in examples])
    # pixel_values = torch.stack([image_processor(example["imgs"]) for example in examples])
    mask = torch.concatenate([example["mask"] for example in examples], dim=0)
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}


def main():

    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Retrieve the indices for paths and labels
    with open(training_args.label_path, 'rb') as f:
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
    parent_path = '/'.join(training_args.label_path.split('/')[:-1])
    for i in range(len(train_paths_80)):
        train_paths_80[i] = train_paths_80[i].replace('../..', parent_path)
    for i in range(len(val_paths_80)):
        val_paths_80[i] = val_paths_80[i].replace('../..', parent_path)
    for i in range(len(test_paths_80)):
        test_paths_80[i] = test_paths_80[i].replace('../..', parent_path)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.int64)

    # Create train and test transforms
    sequence_length = 16
    tube_mask_generator = TubeMaskGenerator()
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((250, 250)),
        RandomCrop(size=224, sequence_length=sequence_length),
        ColorJitter(sequence_length=sequence_length, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        RandomHorizontalFlip(sequence_length=sequence_length),
        RandomRotation(degrees=5, sequence_length=sequence_length),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((250, 250)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
    ])
    train_dataset = Cholec80DatasetSequence(train_paths_80, train_labels_80, sequence_length, tube_mask_generator, train_transforms)
    val_dataset = Cholec80DatasetSequence(val_paths_80, val_labels_80, sequence_length, tube_mask_generator, test_transforms)
    test_dataset = Cholec80DatasetSequence(test_paths_80, test_labels_80, sequence_length, tube_mask_generator, test_transforms)
    print(f'Length of train_dataset: {len(train_dataset)}')
    print(f'Length of val_dataset: {len(val_dataset)}')
    print(f'Length of test_dataset: {len(test_dataset)}')s

    test = train_dataset[0]
    print(f'shape of imgs: {test["imgs"].shape}')
    raise Exception('ckpt')

    # print(f'x image shape {x["imgs"].shape}')
    # print(f'x label shape {x["label"].shape}')
    # print(f'x mask shape {x["mask"].shape}')


    # Setup logging & telemetry
    send_example_telemetry("run_video_mae", model_args, data_args)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Initialize our dataset.
    ds = {'train': train_dataset,'validation': val_dataset,'test': test_dataset}

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Load pretrained model and image processor
    config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision, "use_auth_token": True if model_args.use_auth_token else None}
    if model_args.config_name:
        config = VideoMAEConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = VideoMAEConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = VideoMAEConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    # config.update( {"mask_ratio": model_args.mask_ratio, "norm_pix_loss": model_args.norm_pix_loss})


    # create model
    if model_args.model_name_or_path:
        model = VideoMAEForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = VideoMAEForPreTraining(config)

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))

    # Compute absolute learning rate
    total_train_batch_size = (training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # Half precision    
    training_args.fp16 = True
    
    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        # tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    main()
    print('Done')