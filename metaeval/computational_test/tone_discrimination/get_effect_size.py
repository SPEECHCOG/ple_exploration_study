"""
    @author María Andrea Cruz Blandón
    @date 02.11.2023

    Calculates the effect size from a csv with the dtw distances per file pair.
"""

import sys
import argparse
from pathlib import Path

from test_setup.calculate_effect_size import obtain_effect_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the effect size from the intermediate csv file.\nUsage: '
                                                 'python get_effect_size.py '
                                                 '--csv path_to_csv '
                                                 '--output path_to_output_folder '
                                                 '--type basic|extended')
    parser.add_argument('--csv', required=True, type=Path, help='Path to the intermediate csv file')
    parser.add_argument('--output', required=True, type=Path, help='Path to the output directory')
    parser.add_argument('--type', type=str, required=True, choices=['basic', 'extended'],
                        help='Type of test to run: basic (only the ci2 and ci3 syllable), extended (includes all the'
                             'minimal syllable pairs for the two tones)')
    args = parser.parse_args()
    obtain_effect_size(args.csv, args.output, args.type)
    sys.exit(0)

