"""
    @author María Andrea Cruz Blandón
    @date 11.10.2020

    This script obtains the segments to be used for the dtw distance calculations.
"""
from typing import List, Tuple

import numpy as np
from pathlib import Path
from multiprocessing import Pool
from itertools import repeat


__docformat__ = ['reStructuredText']
__all__ = ['get_segments']


def _read_prediction(prediction_path: Path, corpus_info: dict, window_shift: int,
                     context: str) -> Tuple[np.ndarray, Path]:
    """
        It reads the h5py file with the prediction and extract the vowel if needed from the context using the window
        shift and corpus information (vowel boundaries from the annotation).
    """

    prediction = np.load(prediction_path)
    if context == 'v':
        trial_name = prediction_path.stem
        onset_time = corpus_info[trial_name]['vowel_onset']
        offset_time = corpus_info[trial_name]['vowel_offset']
        # Time in msec (window_shift)
        timestamps_trial = np.arange(window_shift / 2, prediction.shape[0] * window_shift, window_shift)
        index_mask = (timestamps_trial >= onset_time) & (timestamps_trial <= offset_time)
        prediction = prediction[index_mask]
    return prediction, prediction_path


def get_segments(predictions_path: Path, corpus_info: dict, corpus: str, window_shift: int,
                 context: str) -> Tuple[List[np.ndarray], List[Path]]:
    """
        It calculates the segments to be used. If context is cvc the full context is used otherwise (v context), the
        vowel is extracted as used as segment. The function returns the segments and the paired file name of each
        segment.
    """
    if context == 'cvc' or (corpus == 'ivc'):
        context = 'cvc'   # in the case of ivc (isolated vowels) cvc and v are the same, and so cvc is used
        # regardless of what was initially selected
    with Pool() as pool:
        predictions_list, file_mapping = zip(*pool.starmap(_read_prediction,
                                                           zip(predictions_path.joinpath(corpus).rglob('**/*.npy'),
                                                               repeat(corpus_info),
                                                               repeat(window_shift),
                                                               repeat(context))))
    return predictions_list, file_mapping

