"""
    @author María Andrea Cruz Blandón
    @date 02.11.2023

    This script calculate the DTW distances and save them in a csv file for the different files.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from multiprocessing import Pool

from test_setup.create_test_conditions import generate_test_conditions
from test_setup.calculate_dtw_distances import calculate_dtw_distances

__docformat__ = ['reStructuredText']
__all__ = ['obtain_distances']


def _read_prediction(prediction_path: Path) -> Tuple[np.ndarray, Path]:
    prediction = np.load(prediction_path)
    return prediction, prediction_path


def _get_segments(predictions_path: Path) -> Tuple[List[np.ndarray], dict]:
    """
        It gets the latents in numpy arrays and the mapping from the file name to the index in the list.
    """
    with Pool() as pool:
        predictions, file_mapping = zip(*pool.map(_read_prediction, predictions_path.rglob('**/*.npy')))
        index_info = {file_path.stem: idx for idx, file_path in enumerate(file_mapping)}
    return predictions, index_info


def obtain_distances(predictions_path: Path, output_folder: Path, test_type: str) -> Path:
    """
        It obtains conditions to calculate the distances and them calculates them. The results are saved in a csv
        file in the output folder. The conditions are calculated based on the test type.
    """

    output_distances_path = output_folder.joinpath('dtw_distances.csv')
    output_distances_path.parent.mkdir(parents=True, exist_ok=True)

    predictions, index_info = _get_segments(predictions_path)

    # get the conditions to be tested
    same_condition, different_condition = generate_test_conditions(index_info, test_type)

    calculate_dtw_distances(predictions, index_info, same_condition, different_condition, output_distances_path)

    return output_distances_path
