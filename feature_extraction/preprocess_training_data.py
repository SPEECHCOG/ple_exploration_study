"""
    @date 03.03.2021
    It extracts the acoustic features from audio files and it creates h5py files for training predictive coding models.
"""
import argparse
import json
import os
import pathlib
import random
import sys
import warnings
import multiprocessing as mp
from itertools import repeat
from typing import Optional, List, Union, Tuple

import h5py
import librosa
import numpy as np
import numpy.matlib as mb

__docformat__ = ['reStructuredText']
__all__ = ['extract_acoustic_features']


def extract_acoustic_features(file_path: Union[pathlib.Path, str],
                              output_path: Union[pathlib.Path, str], source_path: Union[pathlib.Path, str],
                              sample_length: Optional[int] = 200,
                              window_length: Optional[float] = 0.025, window_shift: Optional[float] = 0.01,
                              num_features: Optional[int] = 13, deltas: Optional[bool] = True,
                              deltas_deltas: Optional[bool] = True, cmvn: Optional[bool] = True,
                              name: Optional[str] = 'mfcc',
                              target_sampling_freq: Optional[int] = 16000,
                              replacement: Optional[bool] = True) -> int:
    """
    Extracts the acoustic features from audio files and it creates the h5py file with them.
    """

    window_length_sample = int(target_sampling_freq * window_length)
    window_shift_sample = int(target_sampling_freq * window_shift)
    output_file = pathlib.Path(output_path)

    output_file_tmp = pathlib.Path(output_file).joinpath(file_path.relative_to(source_path)).with_suffix('.h5')

    if not replacement and output_file_tmp.exists():
        return np.zeros(1)

    signal, sampling_freq = librosa.load(file_path, sr=target_sampling_freq)

    if name == 'mfcc':
        tmp_feats = librosa.feature.mfcc(signal, target_sampling_freq, n_mfcc=num_features,
                                         n_fft=window_length_sample, hop_length=window_shift_sample)
        if deltas:
            mfcc_tmp = tmp_feats
            mfcc_deltas = librosa.feature.delta(mfcc_tmp)
            tmp_feats = np.concatenate([tmp_feats, mfcc_deltas])
            if deltas_deltas:
                mfcc_deltas_deltas = librosa.feature.delta(mfcc_tmp, order=2)
                tmp_feats = np.concatenate([tmp_feats, mfcc_deltas_deltas])
    else:
        tmp_feats = librosa.feature.melspectrogram(y=signal, sr=target_sampling_freq, n_fft=window_length_sample,
                                                   hop_length=window_shift_sample, n_mels=num_features)
        tmp_feats = librosa.power_to_db(tmp_feats)

    tmp_feats = np.transpose(tmp_feats)  # (time, features)

    # Normalisation
    if cmvn:
        mean = mb.repmat(np.mean(tmp_feats, axis=0), tmp_feats.shape[0], 1)
        std = mb.repmat(np.std(tmp_feats, axis=0), tmp_feats.shape[0], 1)
        tmp_feats = np.divide((tmp_feats - mean), std)

    if (replacement and output_file_tmp.exists()) or not output_file_tmp.exists():
        _create_single_h5py_file(output_file_tmp, tmp_feats)

    return 0


def _create_single_h5py_file(output_file_path: Union[pathlib.Path, str], features: np.ndarray) -> None:
    """
    Creates a single h5py file with the acoustic features.
    """
    os.makedirs(pathlib.Path(output_file_path).parent, exist_ok=True)
    with h5py.File(pathlib.Path(output_file_path), 'w') as out_file:
        out_file.create_dataset('data', data=features)


def _read_config_file(config_path: Union[pathlib.Path, str]) -> Tuple[List[pathlib.Path], dict]:
    """
    Reads zipfile/tarfile or folder with audio files to extract their paths and shuffle the list if stated in the
    configuration file.
    """
    with open(config_path) as config_file:
        config = json.load(config_file)
        source_folder = config["audios_path"]
        audio_extensions = config["extensions"]
        file_paths = []

        if pathlib.Path(source_folder).exists():
            if pathlib.Path(source_folder).is_dir():
                file_paths += [audio_file for audio_file in pathlib.Path(source_folder).rglob("*") if
                               audio_file.suffix in audio_extensions]
            else:
                warnings.warn(f"The audios path is not a directory: "
                              f"{source_folder}.")
        else:
            warnings.warn(f"audios_path: {source_folder} does not exist")

        if config["shuffle_files"]:
            random.shuffle(file_paths)

        return file_paths, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to preprocess audio files, extract acoustic features and '
                                                 'create h5py files. '
                                                 '\nUsage: python preprocess_training_data.py '
                                                 '--config path_to_json_file')

    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    file_paths_list, params = _read_config_file(args.config)

    try:
        with mp.Pool(mp.cpu_count()) as pool:
            print('starting processing files')
            all_features = pool.starmap(extract_acoustic_features, zip(file_paths_list, repeat(params['output_path']),
                                                                       repeat(params['audios_path']),
                                                                       repeat(params['sample_length']),
                                                                       repeat(params['window_length']),
                                                                       repeat(params['window_shift']),
                                                                       repeat(params['num_features']),
                                                                       repeat(params['deltas']),
                                                                       repeat(params['deltas_deltas']),
                                                                       repeat(params['cmvn']),
                                                                       repeat(params['name']),
                                                                       repeat(params['target_sampling_freq']),
                                                                       repeat(params['replacement'])
                                                                       ))
    except Exception as e:
        print(e)
        print('error processing files')
    print('all file processed')
    sys.exit(0)
