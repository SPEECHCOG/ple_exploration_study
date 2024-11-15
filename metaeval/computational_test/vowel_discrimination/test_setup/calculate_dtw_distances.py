"""
    This script calculates DTW distances for the different vowel segments (those specified in the test conditions
    lists). It also creates a csv file with the distances calculated for each contrast.

    @date 27.05.2021
    @update 11.10.2023
"""

__docformat__ = ['reStructuredText']
__all__ = ['calculate_dtw_distances']

from itertools import repeat
from pathlib import Path
from typing import List, Tuple
import pandas as pd

import numpy as np
from dtw import dtw

from multiprocessing import Pool


def _get_unique_pairs_trials(condition_list: List[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    # calculate unique pairs ( [a, b] and [b, a] are the same)
    unique_pairs_set = set()
    for contrast in condition_list:
        for trial1, trial2 in contrast:
            if (trial1, trial2) not in unique_pairs_set:
                if (trial2, trial1) not in unique_pairs_set:
                    unique_pairs_set.add((trial1, trial2))
    return list(unique_pairs_set)  # to preserve order


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


def _create_df_condition(distances: List[float], unique_pairs: List[Tuple[str, str]],
                         conditions: List[List[Tuple[str, str]]], contrasts: List[Tuple[str, str]],
                         languages: List[Tuple[str, str]], corpus: str, contrast_type: str,
                         condition_type: str) -> pd.DataFrame:
    df = pd.DataFrame({'distance': distances, 'file_pair': unique_pairs})
    # add rows with same disntances but reversed order of the file pair
    df = pd.concat([df, pd.DataFrame(
        {'distance': distances, 'file_pair': [(x2, x1) for (x1, x2) in unique_pairs]})])
    df = df.drop_duplicates(subset=['file_pair'])

    all_same_conditons = [pair for pair_list in conditions for pair in pair_list]
    df1 = pd.DataFrame({'file_pair': all_same_conditons})
    df1 = df1.merge(df, on='file_pair', how='left')
    del df
    # expand contrast and language columns
    df1['contrast'] = [item for sublist in
                       [repeat(contrasts[idx], len(conditions[idx])) for idx in range(len(conditions))] for
                       item in sublist]
    df1['language'] = [item for sublist in [repeat(languages[idx], len(conditions[idx])) for idx in
                                            range(len(conditions))] for
                       item in sublist]
    df1['corpus'] = corpus
    df1['type'] = contrast_type
    df1['condition'] = condition_type
    df1['file1'] = df1['file_pair'].apply(lambda x: x[0])
    df1['file2'] = df1['file_pair'].apply(lambda x: x[1])
    df1 = df1.drop(columns=['file_pair'])
    df1 = df1[['contrast', 'language', 'corpus', 'type', 'condition', 'file1', 'file2', 'distance']]
    return df1


def calculate_dtw_distances(segments: List[np.ndarray], file_mapping: List[Path],
                            contrasts: List[Tuple[str, str]], contrasts_languages: List[Tuple[str, str]],
                            same_conditions: List[List[Tuple[str, str]]],
                            different_conditions: List[List[Tuple[str, str]]],
                            output_distances_path: Path, contrast_type: str, corpus: str) -> None:

    indices = {file_path.stem: idx for idx, file_path in enumerate(file_mapping)}
    same_unique_pairs = _get_unique_pairs_trials(same_conditions)
    different_unique_pairs = _get_unique_pairs_trials(different_conditions)

    # calculate distances
    with Pool() as pool:
        same_unique_distances = list(pool.starmap(_calculate_dtw,
                                                  zip([_get_segment(segments, indices, x1) for (x1, x2) in same_unique_pairs],
                                                      [_get_segment(segments, indices, x2) for (x1, x2) in same_unique_pairs])))
        different_unique_distances = list(pool.starmap(_calculate_dtw,
                                                       zip([_get_segment(segments, indices, x1) for (x1, x2) in different_unique_pairs],
                                                           [_get_segment(segments, indices, x2) for (x1, x2) in different_unique_pairs])))

    # write results
    df_same = _create_df_condition(same_unique_distances, same_unique_pairs, same_conditions, contrasts,
                                   contrasts_languages, corpus, contrast_type, 'same')
    df_different = _create_df_condition(different_unique_distances, different_unique_pairs, different_conditions,
                                        contrasts, contrasts_languages, corpus, contrast_type, 'different')
    df = pd.concat([df_same, df_different])

    if output_distances_path.exists():
        df.to_csv(output_distances_path, mode='a', index=False, header=False)
    else:
        df.to_csv(output_distances_path, index=False)

