"""
    @author María Andrea Cruz Blandón
    @date 07.11.2023

    This script calculates the effect size from the intermediate csv file (containing the attentional preference scores
    per frame per file).
"""

import argparse
import sys
from pathlib import Path
from calculate_phonotactics_preference_d import calculate_effect_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the effect size from the intermediate csv file.\nUsage: '
                                                 'python get_effect_size.py '
                                                 '--csv path_to_csv '
                                                 '--output path_to_output_folder ')
    parser.add_argument('--csv', required=True, type=Path, help='Path to the intermediate csv file')
    parser.add_argument('--output', required=True, type=Path, help='Path to the output directory')

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    calculate_effect_size(args.csv, args.output)
    sys.exit(0)
