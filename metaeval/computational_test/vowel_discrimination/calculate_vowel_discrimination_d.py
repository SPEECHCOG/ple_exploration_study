"""
    This script runs the test for vowel discrimination task, calculating meta-analysis statistics.

    @date 28.05.2021
    @updated 10.10.2023
"""

__docformat__ = ['reStructuredText']
__all__ = ['run_test']

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from test_setup.calculate_effect_size import obtain_effect_size
from test_setup.obtain_dtw_distances import obtain_distances

BASIC_OC_CONTRASTS = [('a', 'a:')]
BASIC_HC_CONTRASTS = [('A', 'i'), ('i', 'I'), ('A', 'E'), ('e', 'E'), ('A', '{')]
BASIC_IVC_CONTRASTS = [('A', 'i'), ('i', 'I'), ('A', 'E'), ('A', '{')]
BASIC_IVC_NON_NATIVE_CONTRASTS = [('a', 'a~'), ('a:', 'a'), ('u:', 'y:')]
BASIC_OC_CONTRASTS_LANGUAGES = [('de', 'de')]
BASIC_HC_CONTRASTS_LANGUAGES = [('en', 'en')] * 5
BASIC_IVC_CONTRASTS_LANGUAGES = [('en', 'en')] * 4
BASIC_IVC_NON_NATIVE_CONTRASTS_LANGUAGES = [('fr', 'fr'), ('jp', 'jp'), ('de', 'de')]


def run_test(predictions_path: Path, output_path: Path, contrasts: List[List[Tuple[str, str]]],
             contrasts_languages: List[List[Tuple[str, str]]], contrasts_corpus: List[str],  contrasts_type: List[str],
             window_shift: int, context: str) -> None:
    """
        It calculates the distances between file pairs for the list of corpora and contrasts to be tested. The segments
        to be compared are extracting depending on the context (v: only the vowel segment; cvc: the whole CVC context).
        When using only the v context, then window_shift is used to extract the correct segment from the predictions.
    """
    # One block for getting correct segments and distances
    output_distances_path = obtain_distances(predictions_path, output_path, contrasts, contrasts_languages,
                                             contrasts_corpus, contrasts_type, window_shift, context)

    # Another block for calculating the effect size and statistics
    obtain_effect_size(output_distances_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run vowel discrimination test (calculation of effect size, '
                                                 'and standard error). '
                                                 '\nUsage: python test_vowel_discrimination.py '
                                                 '--predictions predictions_path --output output_folder_csv_file '
                                                 '--type all|native|non_native --ivc bool_include_synthetic_data '
                                                 '--window_shift milliseconds_window_shift_use_for_predictions '
                                                 '--context cvc|v ')

    parser.add_argument('--predictions', type=Path, required=True,
                        help='Path to the latent representations (predictions)')
    parser.add_argument('--output', type=Path, required=True, help='Path to the output folder where to '
                                                                   'save the csv file')
    parser.add_argument('--type', type=str, required=True, choices=['all', 'native', 'non_native'],
                        help='Type of test to run: native (only native vowel contrasts), non_native, or all (both'
                             'type of contrasts)')
    parser.add_argument('--ivc', action='store_true', help='Whether to include synthetic data or not')
    parser.add_argument('--window_shift', type=int, default=10,
                        help='Window shift (in milliseconds) used for predictions. This is required to align frames '
                             'with phone boundaries in cases where the CVC context is not used.')
    parser.add_argument('--context', type=str, default='cvc', choices=['cvc', 'v'],
                        help='Type of context to be used in the test: cvc (by default), or v (only the vowel segment).')

    args = parser.parse_args()

    contrasts_test = []
    contrasts_languages_test = []
    contrasts_corpus_test = []
    contrasts_type_test = []

    if args.type == 'native' or args.type == 'all':
        contrasts_test = [BASIC_HC_CONTRASTS]
        contrasts_languages_test = [BASIC_HC_CONTRASTS_LANGUAGES]
        contrasts_corpus_test = ['hc']
        contrasts_type_test = ['native']
        if args.ivc:
            contrasts_test.append(BASIC_IVC_CONTRASTS)
            contrasts_languages_test.append(BASIC_IVC_CONTRASTS_LANGUAGES)
            contrasts_corpus_test.append('ivc')
            contrasts_type_test.append('native')
    if args.type == 'non_native' or args.type == 'all':
        contrasts_test.append(BASIC_OC_CONTRASTS)
        contrasts_languages_test.append(BASIC_OC_CONTRASTS_LANGUAGES)
        contrasts_corpus_test.append('oc')
        contrasts_type_test.append('non_native')
        if args.ivc:
            contrasts_test.append(BASIC_IVC_NON_NATIVE_CONTRASTS)
            contrasts_languages_test.append(BASIC_IVC_NON_NATIVE_CONTRASTS_LANGUAGES)
            contrasts_corpus_test.append('ivc')
            contrasts_type_test.append('non_native')

    run_test(args.predictions, args.output, contrasts_test, contrasts_languages_test,
             contrasts_corpus_test, contrasts_type_test, args.window_shift, args.context)
    sys.exit(0)
