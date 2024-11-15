"""
    @author María Andrea Cruz Blandón
    @date 02.11.2023

    This script calculate the correct segments to be compared and get the DTW distances between them. Then, it
    calculates the effect size and its statistics.
"""

import argparse
import sys
from pathlib import Path

from test_setup.obtain_dtw_distances import obtain_distances
from test_setup.calculate_effect_size import obtain_effect_size

__docformat__ = ['reStructuredText']
__all__ = ['run_test']


def run_test(predictions: Path, output_folder: Path, test_type: str) -> None:
    """
        It calculates the DTW distances between files of same syllable pair. The segments to be compared are obtained
        depending on their syllable, speaker and the type of the test. Then the effect size is calculated comparing
        same (same tone by different speakers) vs different (different tone by different speakers) distances.
    """
    output_distances_path = obtain_distances(predictions, output_folder, test_type)
    obtain_effect_size(output_distances_path, output_folder, test_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrip to run the tone discriminatoin test (calculation of the effect'
                                                 'size and its statistics).'
                                                 '\nUsage: python calculate_tone_discrimination_d.py '
                                                 '--predictions predictions_path --output output_folder_csv_file '
                                                 '--type basic|extended ')
    parser.add_argument('--predictions', type=Path, required=True,
                        help='Path to the latent representations (predictions)')
    parser.add_argument('--output', type=Path, required=True, help='Path to the output folder where to '
                                                                   'save the csv file')
    parser.add_argument('--type', type=str, required=True, choices=['basic', 'extended'],
                        help='Type of test to run: basic (only the ci2 and ci3 syllable), extended (includes all the'
                             'minimal syllable pairs for the two tones)')
    args = parser.parse_args()

    run_test(args.predictions, args.output, args.type)
    sys.exit(0)



