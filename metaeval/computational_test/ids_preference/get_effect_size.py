"""
    @author María Andrea Cruz Blandón
    @date 16.10.2023

    This script calculates the effect size from the intermediate csv file.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

from calculate_ids_preference_d import calculate_statistics, write_final_csv


def get_effect_size(intermediate_csv: Path, output_folder: Path):
    df = pd.read_csv(intermediate_csv)
    metrics = calculate_statistics(df)
    output_folder.mkdir(parents=True, exist_ok=True)
    write_final_csv(metrics, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the effect size from the intermediate csv file.\nUsage: '
                                                 'python get_effect_size.py '
                                                 '--csv path_to_csv '
                                                 '--output path_to_output ')
    parser.add_argument('--csv', required=True, type=Path, help='Path to the intermediate csv file')
    parser.add_argument('--output', required=True, type=Path, help='Path to the output directory')
    args = parser.parse_args()
    get_effect_size(args.csv, args.output)
    sys.exit(0)
