import numpy as np
import os
import random
import pandas
import collections
import copy
import csv
import math
import json
import cv2
import multidict
import glob
from PIL import Image
import tqdm

# from tqdm import tqdm
import torch
import torch.utils.data
from torchvision import transforms

from data_utils.surgical_dataset import SurgicalDataset
from data_utils.protobuf_dataset import load_protobuf_dir
from misc.base_params import parse_arguments

DATASET_KEY_IMAGES = "imgs"
DATASET_KEY_PHASE_TRAJECTORY = "phase_trajectory"
DATASET_KEY_VIDEO_NAME = "video_name"
DATASET_KEY_SURGEON_NAME = "surgeon_id"
DATASET_KEY_VALID_LIST = "valid_list"

"""
This implementation is based on
https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py
Licensed under The MIT License
"""


class MaskingGenerator:
    def __init__(
        self,
        mask_window_size,
        num_masking_patches,
        min_num_patches=16,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(
            mask_window_size,
            (list, tuple),
        ):
            mask_window_size = (mask_window_size,) * 2
        self.height, self.width = mask_window_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] == 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=(self.height, self.width), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask


class MaskingGenerator3D:
    def __init__(
        self,
        mask_window_size,
        num_masking_patches,
        min_num_patches=16,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        self.temporal, self.height, self.width = mask_window_size
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(100):
            target_area = random.uniform(self.min_num_patches, self.max_num_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            t = random.randint(1, self.temporal)  # !
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                front = random.randint(0, self.temporal - t)

                num_masked = mask[
                    front : front + t, top : top + h, left : left + w
                ].sum()
                # Overlap
                if 0 < h * w * t - num_masked <= max_mask_patches:
                    for i in range(front, front + t):
                        for j in range(top, top + h):
                            for k in range(left, left + w):
                                if mask[i, j, k] == 0:
                                    mask[i, j, k] = 1
                                    delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=(self.temporal, self.height, self.width), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask


class MaskedSurgicalDataset(SurgicalDataset):
    def __init__(
        self,
        images_list=[],
        labels_list=[],
        video_idx_list={},
        width=None,
        height=None,
        transform=None,
        time_step=1,
        past_length=10,
        class_names=None,
        track_name=None,
        cache_dir=None,
        video_data_type="video",
        fps=30,
        params=None,
        cfg=None
    ):
        super().__init__(
            images_list,
            labels_list,
            video_idx_list,
            width,
            height,
            transform,
            time_step,
            past_length,
            class_names,
            track_name,
            cache_dir,
            video_data_type,
            fps,
            params,
        )
        self.cfg = cfg

    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * sefl.cfg.AUG_MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            mask_gen = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = mask_gen()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(
                shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=np.int
            )
            n_mask = round(
                self.cfg.AUG_MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO
            )
            mask_t_indices = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_indices, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            mask_gen = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = mask_gen()
        return mask

    def __getitem__(self, idx):

        """
        Surgical Dataset sample contains:
            "idx": time_idx,
            "sequence_length": sample_sequence_length,
            DATASET_KEY_IMAGES: sample_img_list,
            DATASET_KEY_VALID_LIST: sample_validity_list,
            DATASET_KEY_PHASE_TRAJECTORY: sample_phase_trajectory,
            DATASET_KEY_VIDEO_NAME: sample_video_name,
        """
        sample = super().__getitem__(idx)
        mask = self._gen_mask()
        mask = torch.from_numpy(mask)
        sample["mask"] = mask
        return sample


def process_data_directory_surgery(
    data_dir,
    fractions=[],
    width=224,
    height=224,
    max_video=80,
    batch_size=32,
    num_workers=4,
    train_transform=None,
    shuffle=True,
    segment_ratio=1.0,
    patient_factor_list=[],
    past_length=10,
    train_ratio=0.75,
    default_fps=25,
    sampler=None,
    verbose=True,
    annotation_folder=None,
    temporal_len=None,
    sampling_rate=1,
    avoid_annotations=False,
    seed=1234,
    skip_nan=True,
    phase_translation_file=None,
    cache_dir="",
    params={},
    masked=False,
    cfg=None
):
    """
    Read a data directory, and can handle multiple annotators.
    :param data_dir: the root dir for the data
    :param fractions:
    :param width:
    :param height:
    :param max_video:
    :param batch_size:
    :param num_workers:
    :param train_transform:
    :param shuffle:
    :param segment_ratio:
    :param train_ratio:
    :param default_fps:
    :param sampler:
    :param verbose:
    :param annotation_folder: - the folder w/ annotations files.
    :param temporal_len:
    :param avoid_annotations:
    :param seed:
    :param sampling_rate: the sampling rate of creating the dataset from videos, unit: fps
    :param avoid_annotation #TODO - complete this one
    :param skip_nan if add nan label into the dataset
    :param params - a dictionary of additional parameters:
    'track_name' - the track name to generate datasets for.
    :param params: a dictionary for new parameters
    #TODO: move arguments into params dictionary
    :return:
    """
    print("sampling rate:  " + str(sampling_rate))
    train_images = []
    train_labels = []
    train_video_idx = multidict.MultiDict()
    test_images = []
    test_labels = []
    test_video_idx = multidict.MultiDict()
    track_name = params.get("track_name", None)
    # make sure there's a trailing separator for consistency
    data_dir = os.path.join(data_dir, "")
    class_names, annotations = load_protobuf_dir(
        annotation_dir=annotation_folder,
        verbose=verbose,
        phase_translation_file=phase_translation_file,
        allowed_track_names=[track_name],
    )

    if track_name is None:
        track_name = list(class_names.keys())[0]
    training_per_phase = False
    training_frames = 0
    test_frames = 0
    video_surgeon_translation_file = params.get("video_surgeon_translation_file", "")
    if not os.path.exists(data_dir):
        raise MissingVideoDirError(data_dir)
    np.random.seed(seed)
    all_video_files = (
        glob.glob(os.path.join(data_dir, "**/*.mp4"))
        + glob.glob(os.path.join(data_dir, "*.mp4"))
        + glob.glob(os.path.join(data_dir, "**/*.avi"))
        + glob.glob(os.path.join(data_dir, "*.avi"))
    )
    for filename in tqdm.tqdm(sorted(all_video_files), desc="reading videos"):
        video_filename = filename[len(data_dir) :].split(".")[0]
        video_pathname = filename
        try:
            phases_info = annotations[video_filename]
        except:
            continue

        video = cv2.VideoCapture(video_pathname)
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0.0:
            if default_fps is not None:
                fps = default_fps
            else:
                raise "fps missing " + video_pathname
        if avoid_annotations:
            phases_info = multidict.MultiDict()
            phases_info.add("all", {"start": 0, "end": int(num_frames / fps) - 1})

        if not (training_per_phase):
            if np.random.uniform(0.0, 1.0) < train_ratio:
                training_data = True
            else:
                training_data = False

            if verbose:
                print(video_pathname + ", training_data=" + str(training_data))

        if training_data:
            train_video_idx[video_pathname] = multidict.MultiDict()
            train_video_idx[video_pathname]["start_idx"] = len(train_images)
        else:
            test_video_idx[video_pathname] = multidict.MultiDict()
            test_video_idx[video_pathname]["start_idx"] = len(test_images)

        time_step = round(fps / sampling_rate)
        current_phase = 0
        phases_info = list(phases_info.items())
        start_pre = -1
        end_pre = -1
        fraction = 1.0
        if fractions is not None:
            if type(fractions) == float:
                fraction = fractions
            elif isinstance(fractions, list):
                raise Exception(
                    "Need to fix fractions based on the old dataset reading"
                )
        else:
            fraction = 1.0
        fraction_draw = np.random.uniform()
        if fraction < fraction_draw:
            continue

        for frame_idx in range(0, num_frames, time_step):
            add_frame_to_dataset = True
            current_time_idx = frame_idx / fps
            phase_step = phases_info[current_phase]
            start = float(phase_step[1]["start"])
            end = float(phase_step[1]["end"])
            label = None

            if training_data:
                training_frames += 1
            else:
                test_frames += 1
            while label is None:
                if current_time_idx >= start and current_time_idx <= end:
                    # if the frame_idx fall in the segment
                    # add the image to the train and test dataset
                    label = class_names[track_name].index(phase_step[0])
                elif current_time_idx < start and current_time_idx > end_pre:
                    if end_pre == -1:
                        # skip the frames before the first annotation
                        add_frame_to_dataset = False
                        label = float("NaN")
                    else:
                        # add empty label (nans) to the dataset
                        label = float("NaN")
                elif current_time_idx > end:
                    if current_phase < (len(phases_info) - 1):
                        # move to the next phase segment
                        current_phase += 1
                        start_pre = start
                        end_pre = end
                        phase_step = phases_info[current_phase]
                        start = float(phase_step[1]["start"])
                        end = float(phase_step[1]["end"])
                    elif current_phase >= (len(phases_info) - 1):
                        # skip the frames after the last annotation segment
                        add_frame_to_dataset = False
                        label = float("NaN")

            if np.isnan(label) and skip_nan:
                add_frame_to_dataset = False

            if add_frame_to_dataset is False:
                continue

            img = (video_pathname, frame_idx)

            try:
                if training_data:
                    train_images.append(img)
                    train_labels.append(label)
                    for patient_factor in patient_factor_list:
                        train_patient_factor[patient_factor].append(
                            patient_factor_sample[patient_factor]
                        )
                else:
                    test_images.append(img)
                    test_labels.append(label)
                    for patient_factor in patient_factor_list:
                        test_patient_factor[patient_factor].append(
                            patient_factor_sample[patient_factor]
                        )
            except:
                pass

        if training_data:
            train_video_idx[video_pathname]["end_idx"] = len(train_images) - 1
        else:
            test_video_idx[video_pathname]["end_idx"] = len(test_images) - 1

    if verbose:
        print(
            "Collected: %d training, %d test examples, segments: %d,%d "
            % (len(train_images), len(test_images), training_frames, test_frames)
        )

    if any(np.isnan(train_labels)):
        print("there is nan in labels")

    if not masked:
        train_dataset = SurgicalDataset(
            train_images,
            train_labels,
            train_video_idx,
            past_length=past_length,
            fps=fps,
            width=width,
            height=height,
            transform=train_transform,
            class_names=class_names,
            cache_dir=cache_dir,
            params=params,
        )

        val_dataset = SurgicalDataset(
            test_images,
            test_labels,
            test_video_idx,
            past_length=past_length,
            fps=fps,
            width=width,
            height=height,
            class_names=class_names,
            cache_dir=cache_dir,
            params=params,
        )
    else:
        train_dataset = MaskedSurgicalDataset(
            train_images,
            train_labels,
            train_video_idx,
            past_length=past_length,
            fps=fps,
            width=width,
            height=height,
            transform=train_transform,
            class_names=class_names,
            cache_dir=cache_dir,
            params=params,
            cfg=cfg
        )

        val_dataset = MaskedSurgicalDataset(
            test_images,
            test_labels,
            test_video_idx,
            past_length=past_length,
            fps=fps,
            width=width,
            height=height,
            class_names=class_names,
            cache_dir=cache_dir,
            params=params,
            cfg=cfg
        )

    dataloaders = {"train": train_dataset, "val": val_dataset}
    return dataloaders
