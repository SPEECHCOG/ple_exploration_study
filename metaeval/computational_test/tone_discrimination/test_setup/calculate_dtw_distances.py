"""
    @author María Andrea Cruz Blandón
    @date 02.11.2023

    This script is used to calculate the DTW distances between the file pairs given in the same and different
    conditions. The distances per file pair are saved in a csv file.
"""

__docformat__ = ['reStructuredText']
__all__ = ['calculate_dtw_distances']

from pathlib import Path
from itertools import repeat
import pandas as pd
import numpy as np
from dtw import dtw
from multiprocessing import Pool
from typing import List, Tuple, Dict


def _calculate_dtw(matrix1: str, matrix2: str) -> Tuple[float, Tuple[str, str]]:
    # Avoid non-definition of distance for zero vectors, replace those vectors with random representation
    zero_frames = np.where(np.sum(matrix1, axis=1) == 0)[0]
    matrix1[zero_frames, :] = np.random.rand(len(zero_frames), matrix1.shape[-1])

    zero_frames2 = np.where(np.sum(matrix2, axis=1) == 0)[0]
    matrix2[zero_frames2, :] = np.random.rand(len(zero_frames2), matrix2.shape[-1])

    alignment = dtw(matrix1, matrix2, keep_internals=True, distance_only=True, dist_method='cosine')
    return alignment.normalizedDistance


def _get_segment(segments_list: List[np.ndarray], indices: dict, trial: str) -> np.ndarray:
    index = indices[trial]
    return segments_list[index]


def _create_df_condition(distances: List[float], pairs: List[Tuple[str, str]], condition_type: str) -> pd.DataFrame:
    df = pd.DataFrame({'distance': distances, 'file_pair': pairs})

    df['syllable'] = df['file_pair'].apply(lambda x: x[0].split('_')[0][:-1])
    df['condition'] = condition_type
    df['file1'] = df['file_pair'].apply(lambda x: x[0])
    df['file2'] = df['file_pair'].apply(lambda x: x[1])
    df = df.drop(columns=['file_pair'])

    df = df[['syllable', 'condition', 'file1', 'file2', 'distance']]
    return df


def calculate_dtw_distances(segments: List[np.ndarray], index_info: dict, same_condition: List[Tuple[str, str]],
                            different_condition: List[Tuple[str, str]], output_distances_path: Path) -> None:
    with Pool() as pool:
        same_distances = list(pool.starmap(
            _calculate_dtw, zip([_get_segment(segments, index_info, x1) for (x1, x2) in same_condition],
                                [_get_segment(segments, index_info, x2) for (x1, x2) in same_condition])))
        different_distances = list(pool.starmap(
            _calculate_dtw, zip([_get_segment(segments, index_info, x1) for (x1, x2) in different_condition],
                                [_get_segment(segments, index_info, x2) for (x1, x2) in different_condition])))

    # Write results
    df_same = _create_df_condition(same_distances, same_condition, 'same')
    df_different = _create_df_condition(different_distances, different_condition, 'different')
    df = pd.concat([df_same, df_different])

    df.to_csv(output_distances_path, index=False)
