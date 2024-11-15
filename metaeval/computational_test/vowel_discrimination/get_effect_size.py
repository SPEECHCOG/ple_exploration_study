"""
    @author María Andrea Cruz Blandón
    @date 16.10.2023

    Calculates the effect size from the intermediate file of distances.
"""

import sys
import argparse
from pathlib import Path

from test_setup.calculate_effect_size import obtain_effect_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the effect size from the intermediate csv file.\nUsage: '
                                                 'python get_effect_size.py '
                                                 '--csv path_to_csv '
                                                 '--output path_to_output ')
    parser.add_argument('--csv', required=True, type=Path, help='Path to the intermediate csv file')
    parser.add_argument('--output', required=True, type=Path, help='Path to the output directory')
    args = parser.parse_args()
    obtain_effect_size(args.csv, args.output)
    sys.exit(0)
