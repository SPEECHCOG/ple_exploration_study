"""
@author María Andrea Cruz Blandón
@date 27.06.2023

It creates the data generators for the APC and CPC models, given the configuration parameters and the csv files with the
dataset (train and validation).
"""

import logging
import pathlib

import h5py
import numpy as np
import pandas as pd
from typing import Union, Tuple, List

from tensorflow.keras.utils import Sequence

__docformat__ = ['reStructuredText']
__all__ = ['configure_args_ple_data_generator', 'PLEDataGenerator']


def configure_args_ple_data_generator(configuration: dict, validation: bool = False, filtered_val: bool = False) -> \
        Tuple[List[str], List[int]]:
    """
    Configure the arguments for the data generator. It sets the correct file paths.
    It outputs, the list of filenames and the list of indices for the data generator.
    """
    filtered_folder_path = configuration['input_features']['folder_filtered']
    original_folder_path = configuration['input_features']['folder_original']
    profile = configuration['model']['profile']
    if validation:
        csv_path = configuration['input_features']['val_dataset']
    else:
        csv_path = configuration['input_features']['train_dataset']

    # load data
    data = pd.read_csv(csv_path)
    if validation:
        if filtered_val:
            data['filename'] = data['filename'].apply(lambda x: pathlib.Path(filtered_folder_path).joinpath(x))
        else:
            data['filename'] = data['filename'].apply(lambda x: pathlib.Path(original_folder_path).joinpath(x))
    else:
        if profile in ['sanity_check_filtered', 'ple']:
            data['filename'] = data['filename'].apply(lambda x: pathlib.Path(filtered_folder_path).joinpath(x))
        else:  # profile in ['sanity_check_original', 'ale']
            data['filename'] = data['filename'].apply(lambda x: pathlib.Path(original_folder_path).joinpath(x))

    # Preprocess data to work with samples
    # Crop recordings to chunk data into sample size exactly
    if configuration['input_features']['crop']:
        data['total_frames'] = data['total_frames'] - (data['total_frames'] % configuration['model']['parameters'][
            'sample_size'])
        # remove recordings with less than sample_size frames
        data = data[data['total_frames'] > 0]
        data = data.reset_index(drop=True)

    # Create two lists, one for indices of the samples and another for the filenames
    indices = []
    filenames = []
    for i in range(len(data)):
        indices += list(range(0, data['total_frames'][i]))
        filenames += [data['filename'][i]] * data['total_frames'][i]

    return filenames, indices


class PLEDataGenerator(Sequence):
    """
    Data generator class
    """

    def __init__(self, filenames: List, indices: List, batch_size: int, num_features: int, sample_size: int,
                 model_type: str, steps_shift: int):

        self.batch_size = batch_size
        self.model_type = model_type
        self.sample_size = sample_size
        self.num_features = num_features
        self.steps_shift = steps_shift

        self.indices = indices
        self.filenames = filenames

        if len(self.indices) % (self.sample_size * self.batch_size) != 0:
            self.indices = self.indices[:-(len(self.indices) % (self.sample_size * self.batch_size))]
            self.filenames = self.filenames[:-(len(self.filenames) % (self.sample_size * self.batch_size))]

        self.prev_batch = 0

        # logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Loaded configuration for data generator. Total frames: {len(self.indices)}')

    def __len__(self):
        return int((len(self.indices) / self.sample_size) / self.batch_size)

    def __getitem__(self, idx):
        # print(f'\nidx: {idx}\n')
        idx = idx + self.prev_batch
        # print(f'idx + prev_batch: {idx}\n')
        tmp_indices = self.indices[
                      idx * self.sample_size * self.batch_size:(idx + 1) * self.sample_size * self.batch_size]
        tmp_filenames = self.filenames[
                        idx * self.sample_size * self.batch_size:(idx + 1) * self.sample_size * self.batch_size]

        final_indices = []
        tmp_file_indices = []
        filename_prev = None

        for filename, idx in zip(tmp_filenames, tmp_indices):
            if filename != filename_prev:
                if tmp_file_indices:
                    final_indices.append((filename_prev, tmp_file_indices))

                filename_prev = filename
                tmp_file_indices = [idx]
            else:
                tmp_file_indices.append(idx)

        # Last group of indices
        if tmp_file_indices:
            final_indices.append((filename_prev, tmp_file_indices))

        x, y = self._get_batch(final_indices)
        return x, y

    def _get_batch(self, indices: List[Tuple[Union[str, pathlib.Path], List[int]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of data
        :param tmp_indices: dictionary of indices for the batch
        :return: x, y
        """

        x = []
        for filename, file_indices in indices:
            with h5py.File(filename, 'r') as f:
                data = f['data'][file_indices, :]
            x.append(data)

        x = np.concatenate(x, axis=0)  # concatenate all the data from the batch
        del data

        if self.model_type == 'cpc':
            y = np.zeros((self.batch_size, self.sample_size, 1))
        else:
            # For APC model, the target is the next steps_shift frames, so we shift the input x
            y = np.roll(x, -self.steps_shift, axis=0)
            y[-self.steps_shift:, :] = 0  # the last steps_shift frames of the batch are set to zero
            y = y.reshape((self.batch_size, self.sample_size, self.num_features))

        x = x.reshape((self.batch_size, self.sample_size, self.num_features))
        return x, y

    def set_prev_batch_number(self, prev_batch: int):
        self.prev_batch = prev_batch
