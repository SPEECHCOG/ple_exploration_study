import pandas as pd
import pathlib
from typing import Union


def get_validation_and_training_data(csv: Union[str, pathlib.Path], total_hours: float,
                                     output_val_csv: Union[str, pathlib.Path],
                                     output_train_csv: Union[str, pathlib.Path]) -> None:
    """
    Get the validation data from the csv file by randomly sampling files until the total hours is reached.
    :param csv: path to the csv file with the metadata of the dataset
    :param total_hours: total hours of the validation dataset
    :param output_val_csv: path to the output csv file for validation set
    :param output_train_csv: path to the output csv file for training set
    :return: None
    """
    csv = pathlib.Path(csv)
    output_val_csv = pathlib.Path(output_val_csv)
    output_train_csv = pathlib.Path(output_train_csv)

    df = pd.read_csv(csv, names=['filename', 'duration', 'bites'])  # duration in seconds
    # filter out recordings shorter than 2 seconds
    df = df[df['duration'] >= 2]
    df['filename'] = df['filename'].apply(lambda x: pathlib.Path(x))
    df['filename'] = df['filename'].apply(lambda x: x.relative_to(*x.parts[:1]))
    df['filename'] = df['filename'].apply(lambda x: x.with_suffix('.h5'))
    df['total_frames'] = df['duration'].apply(lambda x: int(x*100) + 1)
    reordered_df = df.sample(frac=1).reset_index(drop=True)
    lim = reordered_df[reordered_df.duration.cumsum() <= total_hours * 3600].shape[0]
    reordered_df.iloc[:lim].to_csv(output_val_csv, index=False)
    reordered_df.iloc[lim:].to_csv(output_train_csv, index=False)
    return None


get_validation_and_training_data('./EN.csv', 2.5, './EN_validation_set_1.csv', './EN_train_set_1.csv')
