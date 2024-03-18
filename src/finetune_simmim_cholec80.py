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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset, load_metric
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor

import transformers
from transformers import (
    CONFIG_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForMaskedImageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoModelForImageClassification,
    ViTForImageClassification
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import evaluate
import pickle
import torchvision
from transforms import RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Cholec80Dataset


""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM).
Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """
    dataset_name: Optional[str] = field(default="cifar10", metadata={"help": "Name of a dataset from the datasets package"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    image_column_name: Optional[str] = field(default=None,metadata={"help": "The column name of the images in the files. If not set, will try to use 'image' or 'img'."},)
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(default=0.15, metadata={"help": "Percent to split off of train for validation."})
    mask_patch_size: int = field(default=32, metadata={"help": "The size of the square patches to use for masking."})
    mask_ratio: float = field(default=0.6,metadata={"help": "Percentage of patches to mask."},)
    max_train_samples: Optional[int] = field(default=None,metadata={"help": ("For debugging purposes or quicker training, truncate the number of training examples to this value if set.")},)
    max_eval_samples: Optional[int] = field(default=None,metadata={"help": ("For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.")},)

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
    model_name_or_path: str = field(default=None,metadata={"help": ("The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a checkpoint identifier on the hub. Don't set if you want to train a model from scratch.")})
    model_type: Optional[str] = field(default=None,metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)})
    config_name_or_path: Optional[str] = field(default=None,metadata={"help": "Pretrained config name or path if not the same as model_name"})
    config_overrides: Optional[str] = field(default=None,metadata={"help": ("Override some existing default config settings when a model is trained from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")})
    cache_dir: Optional[str] = field(default=None,metadata={"help": "Where do you want to store (cache) the pretrained models/datasets downloaded from the hub"},)
    model_revision: str = field(default="main",metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},)
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(default=False,metadata={"help": ("Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models).")},)
    image_size: Optional[int] = field(default=None,metadata={"help": ("The size (resolution) of each image. If not specified, will use `image_size` of the configuration.")})
    patch_size: Optional[int] = field(default=None,metadata={"help": ("The size (resolution) of each patch. If not specified, will use `patch_size` of the configuration.")})
    encoder_stride: Optional[int] = field(default=None,metadata={"help": "Stride to use for the encoder."})


@dataclass    
class CustomTrainingArguments(TrainingArguments):
    label_path: str = field(default=None, metadata={"help": "Path to the label file."})


# Forming Batches
def collate_fn(examples):
    #pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = torch.stack([example["imgs"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
    # return {"pixel_values": pixel_values}

# Define Evaluation metric
def compute_metrics(p):
    logits, labels = p
    metric = evaluate.load('accuracy')
    # metric = load_metric('accuracy')
    return metric.compute(predictions=np.argmax(logits, axis=1), references=labels)


def main():
    # See all possible arguments in src/transformers/training_args.py or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create config
    # Distributed training: The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision, "use_auth_token": True if model_args.use_auth_token else None, "num_labels": 7}
    if model_args.config_name_or_path:
        config = AutoConfig.from_pretrained(model_args.config_name_or_path, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    config.update({"label_names": None})

    # Update configurations and make sure decoder type is simmim (only relevant for BEiT) 
    if hasattr(config, "decoder_type"):
        config.decoder_type = "simmim"
    model_args.image_size = model_args.image_size if model_args.image_size is not None else config.image_size
    model_args.patch_size = model_args.patch_size if model_args.patch_size is not None else config.patch_size
    model_args.encoder_stride = (model_args.encoder_stride if model_args.encoder_stride is not None else config.encoder_stride)
    config.update({"image_size": model_args.image_size, "patch_size": model_args.patch_size, "encoder_stride": model_args.encoder_stride,})

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
    sequence_length = 1
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
    train_dataset = Cholec80Dataset(train_paths_80, train_labels_80, sequence_length, train_transforms)
    val_dataset = Cholec80Dataset(val_paths_80, val_labels_80, sequence_length, test_transforms)
    test_dataset = Cholec80Dataset(test_paths_80, test_labels_80, sequence_length, test_transforms)
    print(f'Length of train_dataset: {len(train_dataset)}')
    print(f'Length of val_dataset: {len(val_dataset)}')
    print(f'Length of test_dataset: {len(test_dataset)}')

    # Initialize our dataset.
    ds = {'train': train_dataset,'validation': val_dataset,'test': test_dataset}

    # Send telemetry and set up logging
    send_example_telemetry("run_mim", model_args, data_args)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",handlers=[logging.StreamHandler(sys.stdout)])
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Create model
    if model_args.model_name_or_path:
        # model = AutoModelForMaskedImageModeling.from_pretrained(
        model = AutoModelForImageClassification.from_pretrained(
        # model = ViTForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForImageClassification.from_config(config)
        # model = ViTForImageClassification.from_config(config)

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
    # print(f'SimMIM model: {model}')
    # print(f'Training args {training_args}')

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        # tokenizer=image_processor,
        data_collator=collate_fn,
        # compute_metrics=compute_metrics
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

    # Testing
    test_results = trainer.predict(test_dataset = ds["test"])
    trainer.log_metrics("test", test_results.metrics)
    trainer.save_metrics("test", test_results.metrics)


    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "masked-image-modeling",
        "dataset": data_args.dataset_name,
        "tags": ["masked-image-modeling"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()