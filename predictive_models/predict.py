"""
    @author: María Andrea Cruz Blandón
    @date: 04.09.2023
    This script calculates the predictions (latent representations) of a given model for a given dataset, and
    save them in h5py files.
"""

import argparse
import sys
import pathlib
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

from miscellaneous import set_gpu_memory_growth, get_input_features
from models.core.utils import FeatureEncoder, ContrastiveLoss
from read_configuration import read_configuration_yaml
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


__docformat__ = ['reStructuredText']
__all__ = ['predict']


def predict(config: str, inputs_path: str, model_path: str, output: str, pca: bool = False) -> None:
    """
        It calculates the latent representations of a model for each of the feature files given in the input_features
        folder. The latents are stored in numpy files in the output folder. If pca is True, the latent representations
        are reduce using PCA and maintaining 95% of the variance.
    """
    # load configuration file
    config = read_configuration_yaml(config)
    model_type = config['model']['parameters']['type']

    # Set GPU memory growth
    set_gpu_memory_growth()

    # Set model
    if model_type == 'apc':
        model = tf.keras.models.load_model(model_path, compile=False)
        latent_layer = model.get_layer('latent_layer').output
    elif model_type == 'cpc':
        model = tf.keras.models.load_model(model_path, compile=False,
                                           custom_objects={'FeatureEncoder': FeatureEncoder,
                                                           'ContrastiveLoss': ContrastiveLoss,
                                                           'K': K})
        latent_layer = model.get_layer('latent_layer').output
    else:
        raise Exception('The model type "%s" is not supported' % model_type)

    # Set predictor
    input_layer = model.get_layer('input_layer').output
    predictor = Model(input_layer, latent_layer)

    all_input_files = [x for x in pathlib.Path(inputs_path).rglob('**/*.h5')]
    all_predictions = []

    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for input_file in all_input_files:
        # get input features in correct format
        features, original_frames = get_input_features(input_file, config)

        # predict
        latents = predictor.predict(features)
        latents = latents.reshape(-1, latents.shape[-1])
        latents = latents[:original_frames, :]

        if pca:
            all_predictions.append((latents, original_frames))
        else:
            # save predictions
            output_file = output.joinpath(input_file.relative_to(inputs_path))
            output_file.parent.mkdir(parents=True, exist_ok=True)

            output_file = output_file.with_suffix('.npy')
            np.save(output_file, latents)

    if pca:
        pca_calc = PCA(n_components=0.95)
        all_predictions_arrays = np.concatenate([x[0] for x in all_predictions], axis=0)
        pca_latents = pca_calc.fit_transform(all_predictions_arrays)

        # save predictions
        seen_frames = 0
        for idx, input_file in enumerate(all_input_files):
            output_file = output.joinpath(input_file.relative_to(inputs_path))
            output_file.parent.mkdir(parents=True, exist_ok=True)
            latents = pca_latents[seen_frames:seen_frames + all_predictions[idx][1], :]
            seen_frames += all_predictions[idx][1]

            output_file = output_file.with_suffix('.npy')
            np.save(output_file, latents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for calculating the latent representations for a folder of '
                                                 'given input features and a given model. \nUsage: python '
                                                 'predict.py --config model_configuration_file --input path_to_input'
                                                 '_features --model path_to_model '
                                                 '--output path_to_output_folder '
                                                 '--pca if_PCA_is_needed --numpy if_numpy_output_is_needed')
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Path to the YAML configuration file')
    parser.add_argument('--input', required=True, type=str,
                        help='Path to the directory with the input features (h5py files)')
    parser.add_argument('--model', required=True, type=str, help='Path to the model checkpoint')
    parser.add_argument('--output', required=True, type=str, help='Path to the npy output file')
    parser.add_argument('--pca', action='store_true', help='If PCA is needed')
    args = parser.parse_args()

    # Create predictions
    predict(args.config, args.input, args.model, args.output, args.pca)
    sys.exit(0)
