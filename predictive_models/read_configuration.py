"""
@author María Andrea Cruz Blandón
@date 24.05.2023

It reads the configuration from the YAML file and validates it against the required fields.
"""
import os
import pathlib
import datetime
from typing import Union
import yaml
import logging

from schema import Schema, Optional, And, Use, Or, SchemaError
from miscellaneous import setup_logger

__docformat__ = ['reStructuredText']
__all__ = ['read_configuration_yaml']


def _create_logger_file(log_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Create a logger file in the given path
    :param log_path: path where the file will be saved
    :return: a pathlib.Path object with the path to the file
    """
    log_path = pathlib.Path(log_path)
    ext = log_path.suffix
    log_path = log_path.parent.joinpath(f'{log_path.stem}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}{ext}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    return log_path


def _create_folder(folder_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Create a folder in the given path
    :param folder_path: path where the folder will be saved
    :return: a pathlib.Path object with the path to the folder
    """
    folder_path = pathlib.Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


configuration_schema = Schema({
    'log_path': And(str, And(Use(pathlib.Path)), Use(_create_logger_file)),
    'input_features': {
        'folder_original': And(
            And(str, lambda s: os.path.exists(s), error='The original features folder does not exist'),
            Use(pathlib.Path)),
        'folder_filtered': And(
            And(str, lambda s: os.path.exists(s), error='The filtered features folder does not exist'),
            Use(pathlib.Path)),
        'val_dataset': And(
            And(str, lambda s: os.path.exists(s), error='The test dataset file does not exist'),
            Use(pathlib.Path)),
        'train_dataset': And(
            And(str, lambda s: os.path.exists(s), error='The train dataset file does not exist'),
            Use(pathlib.Path)),
        'num_features': int,
        Optional('crop', default=False): bool,
    },
    'model': {
        'output_path': And(str, Use(_create_folder)),
        'checkpoint_path': Or(None, And(
            And(str, lambda s: os.path.exists(s), error='The checkpoint path does not exist'),
            Use(pathlib.Path))),
        'profile': And(str, lambda s: s in ['sanity_check_filtered', 'sanity_check_original', 'ple', 'ale'],
                       error='The profile is not supported'),
        'checkpoints': {
            Optional('save_initial_checkpoint', default=True): bool,
            Optional('save_last_checkpoint', default=True): bool,
            Optional('custom_checkpoints', default=[]): list,
            Optional('frequency_high_resolution', default=200): Use(float),
            Optional('frequency_low_resolution', default=10): Use(float),
            Optional('overall_frequency', default=400): Use(float),
            Optional('max_low_resolution', default=100): Use(float)
        },
        'backup': {
            'resume_checkpoint_name': Or(None, str),
            # Default setting is 32 samples each of 2 seconds then 50 hours will be 2500 steps
            # (each step is 1 full batch, in this case 0.02 hours)
            Optional('save_freq', default=2500): int,
        },
        'tensorboard': {
            Optional('update_freq', default=5000): int,
        },
        'parameters': {
            'type': And(str, lambda s: s in ['apc', 'cpc'], error='The model type is not supported'),
            Optional('batch_size', default=32): int,
            Optional('latent_dim', default=512): int,
            Optional('sample_size', default=200): int,
            'learning_rate': float,
            'specific': {
                # APC
                Optional('prenet', default=True): bool,
                Optional('prenet_layers', default=3): int,
                Optional('prenet_units', default=128): int,
                Optional('prenet_dropout', default=0.2): float,
                Optional('rnn_layers', default=3): int,
                Optional('rnn_units', default=512): int,
                Optional('rnn_dropout', default=0.0): float,
                Optional('residual', default=True): bool,
                Optional('steps_shift', default=5): int,
                # CPC
                Optional('encoder_layers', default=5): int,
                Optional('encoder_units', default=512): int,
                Optional('encoder_dropout', default=0.2): float,
                Optional('gru_units', default=256): int,
                Optional('dropout', default=0.2): float,
                Optional('negative_samples', default=10): int,
                Optional('steps', default=12): int
            }
        }
    }
}, ignore_extra_keys=True)


def read_configuration_yaml(config_path: Union[str, pathlib.Path]) -> dict:
    """
    Read the configuration from a YAML file and validates it. The configuration to be used
    is saved in the log folder with the date and name of the log file but with yml extension.
    :param config_path: path to the YAML configuration file
    :return: a dictionary with the configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        try:
            config = configuration_schema.validate(config)
            if 'sanity_check' in config['model']['profile']:
                # Sanity check does not start from checkpoint_path
                if config['model']['backup']['resume_checkpoint_name'] and config['model']['checkpoint_path']:
                    raise ValueError('The resume checkpoint and the checkpoint path cannot be used at the same time')
            if config['model']['backup']['resume_checkpoint_name'] is not None:
                if not config['model']['output_path'].joinpath(
                        config['model']['backup']['resume_checkpoint_name']).is_dir():
                    raise FileNotFoundError(f'The resume checkpoint folder '
                                            f'{config["model"]["backup"]["resume_checkpoint_name"]} '
                                            f'does not exist')
                elif not config['model']['output_path'].joinpath(
                        config['model']['backup']['resume_checkpoint_name']).joinpath('chief').is_dir():
                    raise FileNotFoundError(f'The resume checkpoint folder "chief" does not exist')
        except SchemaError or FileNotFoundError or ValueError as e:
            logging.error(f'Error in configuration file: {e}')
            raise e
    # Logging
    setup_logger(config)
    logging.info(f'Configuration file successfully loaded: {pathlib.Path(config_path).absolute()}')

    with open(config['log_path'].parent.joinpath(f"{config['log_path'].stem}.yml"), 'w') as f:
        yaml.dump(config, f)
        logging.info(f'Configuration file to be used saved in: {config["log_path"].absolute()}')
    return config
