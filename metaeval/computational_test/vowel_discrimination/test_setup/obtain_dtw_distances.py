"""
    @author María Andrea Cruz Blandón
    @date 11.10.2023

    This script corresponds to the main function to calculate the dtw distances for the different contrasts, corpus, and
    test conditions.
"""
import pickle
from pathlib import Path
from typing import List, Tuple


__docformat__ = ['reStructuredText']
__all__ = ['obtain_distances']

from test_setup.calculate_dtw_distances import calculate_dtw_distances
from test_setup.create_tests_conditions import generate_tests_conditions
from test_setup.extract_segments import get_segments

BASIC_FILTERS = {'repetitions': ['N1'], 'failed_listeners_test': False}


def obtain_distances(predictions_path: Path, output_distances_path: Path, contrasts: List[List[Tuple[str, str]]],
                     contrasts_languages: List[List[Tuple[str, str]]], contrasts_corpus: List[str],
                     contrasts_type: List[str], window_shift: int, context: str) -> Path:
    """
        It goes corpus by corpus and obtain the segments to be tested according to the selected context and window
        shift. Then it calculates the different test conditions (same/different contrast) to be tested. Finally,
        it calculates the dtw distances per each file pair and records the result in a csv file.
    """

    output_distances_path = output_distances_path.joinpath('dtw_distances.csv')
    output_distances_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, corpus in enumerate(contrasts_corpus):
        with open(f'./test_data/{corpus}_corpus_info.pickle', 'rb') as f:
            corpus_info = pickle.load(f)
        predictions_list, file_mapping = get_segments(predictions_path, corpus_info, corpus, window_shift, context)

        # List of the list of file pairs to be compared per contrast
        same_condition, different_conditions = generate_tests_conditions(corpus_info, contrasts[idx],
                                                                         BASIC_FILTERS, corpus,
                                                                         contrasts_languages[idx])
        # calculate DTW distances
        calculate_dtw_distances(predictions_list, file_mapping, contrasts[idx], contrasts_languages[idx],
                                same_condition, different_conditions, output_distances_path, contrasts_type[idx],
                                corpus)

    return output_distances_path
