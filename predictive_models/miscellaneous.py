"""
    @author: María Andrea Cruz Blandón
    @date: 24.05.2023
    This script contains some miscellaneous functions to be used while in this project.
"""
import logging
from typing import Tuple, Union
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

__docformat__ = ['reStructuredText']
__all__ = ['setup_logger', 'set_gpu_memory_growth', 'get_input_features']


def setup_logger(config: dict) -> None:
    """
    Setup the logger for the project
    :param config: the configuration dictionary
    :return: None
    """
    logging.basicConfig(filename=config['log_path'], level=logging.DEBUG,
                        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')


def set_gpu_memory_growth():
    """
        Set the GPU memory growth to True
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, enable=True)


def get_input_features(input_file: Union[str, Path], config: dict) -> Tuple[np.ndarray, int]:
    """
        It reads the input features from the input_file and returns them in the format required for the model
        (sample_size x num_features). For the input features that are not divisible by sample_size, it pads them with
        zeros. It also returns the number of original frames of the input features.
    """
    # Read file
    with h5py.File(input_file, 'r') as f:
        features = f['data'][:]
        frames = features.shape[0]

        samples_size = config['model']['parameters']['sample_size']

        # Pad with zeros if necessary
        if frames % samples_size != 0:
            pad = samples_size - (frames % samples_size)
            features = np.pad(features, ((0, pad), (0, 0)), 'constant', constant_values=0)
        # reshape to (num_samples, sample_size, num_features)
        features = features.reshape(-1, samples_size, features.shape[1])

        return features, frames
