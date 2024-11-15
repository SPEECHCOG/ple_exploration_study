"""
@author María Andrea Cruz Blandón
@date 08.02.2022

It trains a model according to the configuration given
"""
import argparse
import logging
import sys

import tensorflow as tf

from miscellaneous import setup_logger
from models.core.apc import APCModel
from models.core.cpc import CPCModel
from read_configuration import read_configuration_yaml

__docformat__ = ['reStructuredText']
__all__ = ['train']


def train(config_path: str) -> None:
    """
    Train a neural network model using the configuration parameters provided
    :param config_path: the path to the YAML configuration file.
    :return: The trained model is saved in a h5 file
    """

    # read configuration file
    config = read_configuration_yaml(config_path)
    setup_logger(config)

    # Use correct model
    model_type = config['model']['parameters']['type']

    # If GPU set memory growth. I change this line from apc.py and cpc.py to here because it is general to both models.
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, enable=True)

    if model_type == 'apc':
        logging.info('Model set to APC')
        model = APCModel(config)
    elif model_type == 'cpc':
        model = CPCModel(config)
        logging.info('Model set to CPC')
    else:
        logging.error('The model type "%s" is not supported' % model_type)
        raise Exception('The model type "%s" is not supported' % model_type)

    logging.info('Model configuration loaded')
    logger = logging.getLogger(__name__)
    model.model.summary(print_fn=logger.info)
    logging.info('Model training to be started')
    model.train()
    logging.info('Model training finished')

    print('Training of model "%s" finished' % model_type)


if __name__ == '__main__':
    # Call from command line
    parser = argparse.ArgumentParser(description='Script for training an APC or CPC model. The configruation is load '
                                                 'from the YAML configuration file.\nUsage: python train.py '
                                                 '--config path_YAML_configuration_file')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Train model
    train(args.config)
    sys.exit(0)
