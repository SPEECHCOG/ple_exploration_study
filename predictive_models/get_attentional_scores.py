"""
    @author María Andrea Cruz Blandón
    @date 11.09.2023

    This script obtains the attentional scores per trial and frame, and save them in a csv files.
    The attentional score is the loss of the model.
"""

import argparse
import sys
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

__docformat__ = ['reStructuredText']
__all__ = ['get_attentional_scores']

from miscellaneous import set_gpu_memory_growth, get_input_features
from models.core.utils import ContrastiveLoss, FeatureEncoder, get_negative_samples
from read_configuration import read_configuration_yaml


def _get_overlapped_features(input_features: np.ndarray, overlap: float, sample_size) -> np.ndarray:
    """
        It obtains the overlapped features of the input features. The overlapped features are obtained by sliding a
        window of size sample_size with a stride of (1 - overlap) * sample_size.
    """
    features_dim = input_features.shape[-1]
    overlapped_frames = int(sample_size * overlap)
    stride = sample_size - overlapped_frames

    features = input_features.reshape((-1, features_dim))
    total_samples = features.shape[0] // stride

    final_features = np.zeros((total_samples, sample_size, features_dim))
    for idx in range(total_samples):
        segment = features[idx * stride:idx * stride + sample_size, :]
        idx_samples = segment.shape[0]
        final_features[idx, :idx_samples, :] = segment

    return final_features


def _remove_overlap(overlapped_features: np.ndarray, overlap: float, original_total_samples: int) -> np.ndarray:
    """
        It removes the overlap introduced by the sliding window with _get_overlapped_features.
    """
    sample_size = overlapped_features.shape[1]
    features_dim = overlapped_features.shape[-1]
    overlapped_frames = int(sample_size * overlap)
    stride = sample_size - overlapped_frames

    final_features = np.zeros((original_total_samples * sample_size, features_dim))
    total_final_frames = final_features.shape[0]

    for idx_sample in range(overlapped_features.shape[0]):
        if idx_sample == 0:
            # first sample
            final_features[0:sample_size, :] = overlapped_features[idx_sample, 0:sample_size, :]
        else:
            # everything else
            init_idx = sample_size + (stride * (idx_sample - 1))
            if init_idx + stride > total_final_frames:
                last_frames = total_final_frames - init_idx
                final_features[init_idx:, :] = \
                    overlapped_features[idx_sample, overlapped_frames: overlapped_frames + last_frames, :]
            else:
                final_features[init_idx: init_idx + stride, :] = \
                    overlapped_features[idx_sample, overlapped_frames:, :]

            if init_idx + stride >= total_final_frames:
                break
    final_features = final_features.reshape((original_total_samples, sample_size, features_dim))
    return final_features


def _calculate_mse_per_frame(predictions: np.ndarray, input_features: np.ndarray, apc_shift: float) -> np.ndarray:
    """
        It calculates the mean squared error per frame of the predictions and the real shifted input features
        (real future features).
    """
    mse_per_frame = tf.keras.metrics.mean_squared_error(input_features[apc_shift:, :], predictions[:-apc_shift, :])

    return mse_per_frame.numpy()


def _get_infonce_per_frame(true_latents: tf.Tensor, pred_latents: tf.Tensor, neg: int, steps: int) -> tf.Tensor:
    timesteps = true_latents.shape[1]
    total_samples = true_latents.shape[0]
    negative_samples = get_negative_samples(true_latents, neg)
    true_latents = K.expand_dims(true_latents, 0)
    targets = K.concatenate([true_latents, negative_samples], 0)
    copies = neg + 1
    logits = None
    for i in range(steps):
        if i == 0:
            # The time-steps are corresponding as is the first step.
            logits = tf.reshape(tf.einsum("stf,cstf->tsc", pred_latents[:, :, :, i], targets[:, :, :, :]), [-1])
        else:
            # We need to match the time-step taking into account the step for which is being predicted
            logits = tf.concat([logits, tf.reshape(tf.einsum("stf,cstf->tsc", pred_latents[:, :-i, :, i],
                                                             targets[:, :, i:, :]), [-1])], 0)

    # shape steps x timesteps x samples x copies (e.g., 12x(200,199,198, .., 189)xNx11)
    # the timesteps are decreasing because for each step calculation the predictions and target are shifted so we have
    # all the information needed (controlling for unknown future) to calculate the dot product between prediction and
    # target vectors
    logits = tf.reshape(logits, (-1, copies))  # shape (steps*decreasing timesteps*samples) x copies
    total_points = tf.shape(logits)[0]

    # Labels, this should be the true value, that is 1.0 for the first copy (positive sample) and 0.0 for the rest.
    label_idx = [True] + [False] * neg
    labels = tf.where(label_idx, tf.ones((total_points, copies)), tf.zeros((total_points, copies)))

    # Entropy per frame
    # shape: total_points
    entropy_per_frame_n_step = tf.nn.softmax_cross_entropy_with_logits(labels, logits)

    # Reduction (sum) across steps
    # InfoNCE is the sum across steps for each frame, since the timesteps are decreasing as steps go forward. First
    # we add zeros for the missing timesteps so the summation can be done at once.
    corrected_entropy_per_frame_n_step = None
    frames_idx = 0
    for step in range(steps):
        current_elements = total_samples * (timesteps - step)
        if step == 0:
            corrected_entropy_per_frame_n_step = entropy_per_frame_n_step[frames_idx:frames_idx + current_elements]
        else:
            missing_timesteps = total_samples * step
            frames_step = tf.concat([entropy_per_frame_n_step[frames_idx:frames_idx + current_elements],
                                     tf.zeros((missing_timesteps,))], 0)
            corrected_entropy_per_frame_n_step = tf.concat([corrected_entropy_per_frame_n_step, frames_step], 0)
        frames_idx += current_elements
    # shape: steps x timesteps x samples
    corrected_entropy_per_frame_n_step = tf.reshape(corrected_entropy_per_frame_n_step, (steps, timesteps, -1))
    infonce_per_frame = tf.math.reduce_sum(corrected_entropy_per_frame_n_step, axis=0)
    # shape: samples x timesteps x 1 (infoNCE value)
    infonce_per_frame = tf.reshape(tf.transpose(infonce_per_frame), (-1, timesteps, 1))

    return infonce_per_frame


def get_attentional_scores(input_dir: Union[Path, str], model_path: Union[Path, str],
                           config: Union[Path, str], overlap: float, output: Union[Path, str]) -> None:
    """
        It calculates the attentional scores per frame per trial of a given model. The scores are stored in numpy files.
    """
    # load configuration file
    config = read_configuration_yaml(config)
    model_type = config['model']['parameters']['type']

    set_gpu_memory_growth()

    # Set model
    if model_type == 'apc':
        model = tf.keras.models.load_model(model_path, compile=False)
        predictor = model
    else:  # cpc
        model = tf.keras.models.load_model(model_path, custom_objects={'ContrastiveLoss': ContrastiveLoss,
                                                                       'FeatureEncoder': FeatureEncoder,
                                                                       'K': K},
                                           compile=False)
        input_layer = model.get_layer('input_layer').output
        projection_layer = model.get_layer('project_latent').output
        steps_projection_layer = model.get_layer('project_steps').output
        predictor = Model(inputs=input_layer, outputs=[projection_layer, steps_projection_layer])

    # Get score per input file
    input_dir = Path(input_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for trial in input_dir.rglob('**/*.h5'):
        # Read file
        features, original_frames = get_input_features(trial, config)
        new_frames = features.shape[0]
        overlapped_features = _get_overlapped_features(features, overlap, config['model']['parameters']['sample_size'])

        if model_type == 'apc':
            predictions = predictor.predict(overlapped_features)
            predictions = _remove_overlap(predictions, overlap, new_frames)
            # reshape features and predictions and remove padding
            features = features.reshape(-1, features.shape[-1])
            features = features[:original_frames, :]
            predictions = predictions.reshape(-1, predictions.shape[-1])
            predictions = predictions[:original_frames, :]

            scores_per_frame = _calculate_mse_per_frame(predictions, features,
                                                        config['model']['parameters']['specific']['steps_shift'])
        else:  # cpc
            true_latents, pred_latents = predictor.predict(overlapped_features)
            overlapped_scores = _get_infonce_per_frame(true_latents, pred_latents,
                                                       config['model']['parameters']['specific']['negative_samples'],
                                                       config['model']['parameters']['specific']['steps'])
            scores_per_frame = _remove_overlap(overlapped_scores.numpy(), overlap, new_frames)
            # remove padding
            scores_per_frame = scores_per_frame.reshape(-1, scores_per_frame.shape[-1])
            scores_per_frame = scores_per_frame[:original_frames, :]

        # Save scores in numpy file
        output_file = output.joinpath(trial.relative_to(input_dir)).with_suffix('.npy')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, scores_per_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obtain attentional scores per frame and trial given a directory of '
                                                 'input features and a model path.\nUsage: python '
                                                 'get_attentional_scores.py --input path_to_input_features '
                                                 '--model path_to_model_checkpoint --config model_configuration '
                                                 '--overlap float_overlap_frames --output path_to_output_csv')
    parser.add_argument('--input', required=True, type=str,
                        help='Path to the directory with the input features (h5py files)')
    parser.add_argument('--model', required=True, type=str, help='Path to the model checkpoint')
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Path to the YAML model configuration file')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap of frames for loss calculation')
    parser.add_argument('--output', required=True, type=str, help='Path to the output npy')
    args = parser.parse_args()

    assert 0 <= args.overlap <= 1, 'Overlap must be between 0 and 1'
    get_attentional_scores(args.input, args.model, args.config, args.overlap, args.output)
    sys.exit(0)
