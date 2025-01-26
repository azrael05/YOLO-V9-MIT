import json
import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from yolo.tools.data_conversion import discretize_categories
from yolo.utils.logger import logger

def have_differences(list1, list2):
    return set(list1) != set(list2)

def locate_label_paths_cls(dataset_path: Path, converter_dict) -> Tuple[Path, Path]:
    """
    Find the path to label files for a specified dataset and phase(e.g. training).

    Args:
        dataset_path (Path): The path to the root directory of the dataset.
        phase_name (Path): The name of the phase for which labels are being searched (e.g., "train", "val", "test").

    Returns:
        Tuple[Path, Path]: A tuple containing the path to the labels file and the file format ("json" or "txt").
    """
    dataset_files = []
    if not os.path.exists(dataset_path):
        raise LookupError(f"The provided path does not exist {dataset_path}") 
    labels = os.listdir(dataset_path)
    if converter_dict == {}:
        converter_dict = {label:index for index, label in enumerate(labels)}
    else:
        converter_labels = list(converter_dict.keys())
        if have_differences(converter_labels, labels):
            raise ValueError("MISMATCHING LABELS PRESENT IN DATASETS or Missing label in one set")

    for label in labels:
        current_label_path = os.path.join(dataset_path, label)
        for image in os.listdir(current_label_path):
            current_image_path = os.path.join(current_label_path, image)
            target = encode_target_vector(converter_dict, label)
            dataset_files.append([current_image_path, target])

    return dataset_files, converter_dict

def encode_target_vector(converter_dict, label):
    target = np.zeros(len(converter_dict))
    target[converter_dict[label]] = 1
    return target