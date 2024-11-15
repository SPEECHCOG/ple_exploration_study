"""
    @author: María Andrea Cruz Blandón
    @date: 05.03.2024

    This script is used to create a summary of the dataset by calculating the number for frames each file has and saving
    a csv file with that information.
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def get_metadata(folder_features: Path, output_csv: Path) -> None:
    """
    It calculates the number of frames for each file in the folder_features and saves the information in a csv file.
    :param folder_features: the folder with the feature files
    :param output_csv: the output csv file
    :return: None
    """
    all_files = list(folder_features.rglob('*.h5'))
    metadata = []
    for file in all_files:
        with h5py.File(file, 'r') as f:
            data = f['data'][:]
            frames = data.shape[0]
        metadata.append({'filename': file.relative_to(folder_features), 'total_frames': frames})

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates a summary of the dataset by calculating the number for frames '
                    'each file has and saving a csv file with that information.\nUsage: '
                    'python create_dataset_summary.py --input path_to_folder --output path_to_output_file')
    parser.add_argument('--input', type=Path, help='Path to the folder with the feature files')
    parser.add_argument('--output', type=Path, help='Path to the csv output file with the summary of the dataset')
    args = parser.parse_args()

    get_metadata(args.input, args.output)
    print('Summary of the dataset created successfully')
    exit(0)
