"""
    @author: María Andrea Cruz Blandón
    @date: 06.03.2024

    This script goes iteratively through the folder containing non-normalised features and calculates the mean and
    standard deviation of the features. Then, it normalises the features and saves them in a new folder.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path

__docformat__ = ['reStructuredText']
__all__ = ['normalise_features']


def normalise_features(input_folder: Path, output_folder: Path, vectors_output: Path) -> None:
    files = list(input_folder.rglob('**/*.h5'))

    n = 0
    s1 = 0
    s2 = 0

    if vectors_output.exists():
        with h5py.File(vectors_output, 'r') as f:
            mean_vec = f['mean'][:]
            std_vec = f['std'][:]
    else:  # Calculate mean and std
        for feat_path in files:
            with h5py.File(feat_path, 'r') as f:
                data = f['data'][:]
                n += data.shape[0]
                s1 += np.sum(data, axis=0)
                s2 += np.sum(data ** 2, axis=0)

        mean_vec = s1 / n
        std_vec = np.sqrt((s2 / n) - mean_vec ** 2)

        # Save mean and std
        vectors_output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(vectors_output, 'w') as f:
            f.create_dataset('mean', data=mean_vec)
            f.create_dataset('std', data=std_vec)

    # Normalise and save data
    for feat_path in files:
        with h5py.File(feat_path, 'r') as f:
            data = f['data'][:]
            normalised_data = (data - mean_vec) / std_vec

        output_path = output_folder.joinpath(feat_path.relative_to(input_folder))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('data', data=normalised_data)

    # Save normalised data
    output_path = output_folder.joinpath(feat_path.relative_to(input_folder))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=normalised_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Normalises the features in the input folder and saves them in the output folder. It also saves the '
                    'mean and standard deviation of the features in a h5py file.\nUsage: python global_normalisation.py '
                    '--input path_to_folder --output path_to_output_folder --vectors path_to_output_vectors')
    parser.add_argument('--input', type=Path, help='Path to the folder with the non-normalised feature files')
    parser.add_argument('--output', type=Path,
                        help='Path to the folder where the normalised feature files will be saved')
    parser.add_argument('--vectors', type=Path,
                        help='Path to the output h5py file with the mean and standard deviation of the features')
    args = parser.parse_args()

    normalise_features(args.input, args.output, args.vectors)
    print('Features normalised successfully')
    exit(0)
