"""
@author María Andrea Cruz Blandón
@date 08.02.2022
Objects needed for loading a Contrastive Predictive Coding model ["Representation Learning with Contrastive
Predictive Coding", van den Oord et al., 2018]
"""
import logging
import os
import pathlib
import time

import tensorflow as tf
import tensorflow.python.keras.backend
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Dense, Conv1D, Layer, Conv2DTranspose, Lambda
from tensorflow.keras.callbacks import Callback, BackupAndRestore, TensorBoard
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Attention
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

__docformat__ = ['reStructuredText']
__all__ = ['FeatureEncoder', 'ContrastiveLoss', 'HourTrackingModelCheckpoint', 'CustomBackUpAndRestore',
           'CustomTensorBoard', 'get_negative_samples']


class Block(Layer):
    """
    Super class for all the blocks so they have get_layer method. The method is used in prediction to extract either
    features of the APC encoder or the CPC encoder
    """

    def __init__(self, name):
        super(Block, self).__init__(name=name)

    def get_layer(self, name=None, index=None):
        """
        Keras sourcecode for Model.
        :param name: String name of the layer
        :param index: int index of the layer
        :return: the layer if name or index is found, error otherwise
        """
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                                 ' but model only has ' + str(len(self.layers)) +
                                 ' layers.')
            else:
                return self.layers[index]
        else:
            if not name:
                raise ValueError('Provide either a layer name or layer index.')
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError('No such layer: ' + name)


class ContrastiveLoss(Block):
    """
    It creates the block that calculates the contrastive loss for given latent representation and context
    representations. Implementation from wav2vec
    (https://github.com/pytorch/fairseq/blob/master/fairseq/models/wav2vec.py)
    [wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862)
    """

    def __init__(self, neg, steps, name='Contrastive_Loss'):
        """
        :param neg: Number of negative samples
        :param steps: Number of steps to predict
        :param name: Name of the block, by default Contrastive_Loss
        """
        super(ContrastiveLoss, self).__init__(name=name)
        self.neg = neg
        self.steps = steps
        self.layers = []
        with K.name_scope(name):
            self.cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                         reduction=tf.keras.losses.Reduction.SUM)

    def call(self, inputs, **kwargs):
        """
        :param inputs: A list with two elements, the latent representation and the context representation
        :param kwargs:
        :return: the contrastive loss calculated
        """
        true_latent, predictions = inputs

        negative_samples = get_negative_samples(true_latent, self.neg)

        true_latent = K.expand_dims(true_latent, 0)

        targets = K.concatenate([true_latent, negative_samples], 0)
        copies = self.neg + 1  # total of samples in targets

        # samples, timesteps, features, steps = predictions.shape

        # Logits calculated from predictions and targets
        logits = None

        for i in range(self.steps):
            if i == 0:
                # The time-steps are corresponding as is the first step.
                logits = tf.reshape(tf.einsum("stf,cstf->tsc", predictions[:, :, :, i], targets[:, :, :, :]), [-1])
            else:
                # We need to match the time-step taking into account the step for which is being predicted
                logits = tf.concat([logits, tf.reshape(tf.einsum("stf,cstf->tsc", predictions[:, :-i, :, i],
                                                                 targets[:, :, i:, :]), [-1])], 0)

        logits = tf.reshape(logits, (-1, copies))
        total_points = tf.shape(logits)[0]

        # Labels, this should be the true value, that is 1.0 for the first copy (positive sample) and 0.0 for the
        # rest.
        label_idx = [True] + [False] * self.neg
        labels = tf.where(label_idx, tf.ones((total_points, copies)), tf.zeros((total_points, copies)))

        # The loss is the softmax_cross_entropy_with_logits sum over copies (classes, true and negs) and mean for all
        # steps and samples
        loss = self.cross_entropy(labels, logits)
        loss = tf.reshape(loss, (1,))
        return loss

    def get_config(self):
        return {'neg': self.neg, 'steps': self.steps}


def get_negative_samples(true_features: tf.Tensor, neg: int) -> tf.Tensor:
    """
    It calculates the negative samples re-ordering the time-steps of the true features.
    :param true_features: A tensor with the apc predictions for the input.
    :param neg: Number of negative samples to calculate.
    :return: A tensor with the negative samples.
    """
    # Shape SxTxF
    samples = K.shape(true_features)[0]
    timesteps = K.shape(true_features)[1]
    features = K.shape(true_features)[2]

    # New shape FxSxT
    true_features = K.permute_dimensions(true_features, pattern=(2, 0, 1))
    # New shape Fx (S*T)
    true_features = K.reshape(true_features, (features, -1))

    high = timesteps

    # New order for time-steps
    indices = tf.repeat(tf.expand_dims(tf.range(timesteps), axis=-1), neg)
    neg_indices = tf.random.uniform(shape=(samples, neg * timesteps), minval=0, maxval=high - 1,
                                    dtype=tf.dtypes.int32)
    neg_indices = tf.where(tf.greater_equal(neg_indices, indices), neg_indices + 1, neg_indices)

    right_indices = tf.reshape(tf.range(samples), (-1, 1)) * high
    neg_indices = neg_indices + right_indices

    # Reorder for negative samples
    # Reorder for negative samples
    negative_samples = tf.gather(true_features, tf.reshape(neg_indices, [-1]), axis=1)
    negative_samples = K.reshape(negative_samples, (features, samples, neg, timesteps))
    negative_samples = K.permute_dimensions(negative_samples, (2, 1, 3, 0))

    return negative_samples


class FeatureEncoder(Block):
    """
    It creates a keras layer for the encoder part (latent representations)
    """

    def __init__(self, n_layers, units, dropout, name='Feature_Encoder'):
        """
        :param n_layers: Number of convolutional layers
        :param units: Number of filters per convolutional layer
        :param dropout: Percentage of dropout between layers
        :param name: Name of the block, by default Feature_Encoder
        """
        super(FeatureEncoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        self.layers = []
        with K.name_scope(name):
            for i in range(n_layers):
                self.layers.append(Dense(units, activation='relu', name='dense_layer_' + str(i)))
                if dropout > 0:
                    if i == n_layers - 1:
                        self.layers.append(Dropout(dropout, name='encoder_latent_layer'))
                    else:
                        self.layers.append(Dropout(dropout, name='dense_dropout_' + str(i)))

    def call(self, inputs, **kwargs):
        """
        It is execute when an input tensor is passed
        :param inputs: A tensor with the input features
        :return: A tensor with the output of the block
        """
        features = inputs
        for layer in self.layers:
            features = layer(features)
        return features

    def get_config(self):
        return {'n_layers': self.n_layers, 'units': self.units, 'dropout': self.dropout}


class AttentionWeights(Attention):

    def _apply_scores(self, scores, value, scores_mask=None):
        """
        Overwrites the base method to output the weights as well
        :param scores: scores tensor
        :param value: value tensor
        :param scores_mask: mask to apply to scores
        :return: attention calculation and weights (two tensors)
        """
        if scores_mask is not None:
            padding_mask = math_ops.logical_not(scores_mask)
            # Bias so padding positions do not contribute to attention distribution.
            scores -= 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx())
        weights = nn.softmax(scores)
        return math_ops.matmul(weights, value), weights


class HourTrackingModelCheckpoint(Callback):
    """
        This class handle the model checkpoint saving. It will save the model according
        to the low and high frequency defined in the configuration file, any custom
        checkpoint, and at the beginning and end of the training if specified.

        The name of the files will be the name of the model, the current hours of speech seen
        and the overall hours of speech seen so far.
    """

    def __init__(self, checkpoints_dict, overall_hours, hours_per_batch, filepath, current_batch,
                 from_backup=False, batch_ckpt=0, verbose=1, **kwargs):
        super(HourTrackingModelCheckpoint, self).__init__(**kwargs)
        self.checkpoints_dict = checkpoints_dict
        self.current_batch = current_batch
        self.current_hours = 0
        self.overall_hours = overall_hours
        self.hours_per_batch = hours_per_batch
        self.filepath = str(filepath)
        self.verbose = verbose
        self.from_backup = from_backup
        self.batch_ckpt = batch_ckpt

        pathlib.Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

    def _set_logs(self, logs=None):
        logs = logs or {}
        logs['current_hours'] = self.current_hours
        logs['overall_hours'] = self.overall_hours
        logs['current_batch'] = self.current_batch
        return logs

    def _save_model(self, logs=None):
        logs = self._set_logs(logs)
        if self.verbose:
            logging.info(f'Saving checkpoint at {self.filepath.format_map(logs)}')
        self.model.save(self.filepath.format_map(logs))

    def on_train_begin(self, logs=None):
        # The model was restored from a backup we need to update the current_hours and overall_hours
        if self.from_backup:  # If BackUpAndRestore callback is used
            if self.model._training_state._ckpt_saved_batch.numpy() != -1:
                batch_num_backup = self.model._training_state._ckpt_saved_batch.numpy() + 1
                if self.batch_ckpt != 0:  # The model is restored, but it started from a checkpoint
                    batch_num_backup = batch_num_backup - self.batch_ckpt

                self.current_hours = batch_num_backup * self.hours_per_batch
                self.overall_hours = self.overall_hours + batch_num_backup * self.hours_per_batch
                logging.info(f'Model restored. Current hours: {self.current_hours}, '
                             f'Overall hours: {self.overall_hours}')
                logs = self._set_logs(logs)  # If the model is restored, and we don't save the initial checkpoint.

        # Check that custom checkpoints are still valid
        self.checkpoints_dict['custom_checkpoints'] = [custom_checkpoint for custom_checkpoint in
                                                       self.checkpoints_dict['custom_checkpoints'] if
                                                       custom_checkpoint > self.current_hours]

        if self.current_hours == 0 and self.checkpoints_dict['save_initial_checkpoint']:
            if self.verbose:
                logging.info(f'Saving initial checkpoint')
            self._save_model(logs)

    def on_train_end(self, logs=None):
        if self.checkpoints_dict['save_last_checkpoint']:
            if self.verbose:
                logging.info(f'Saving last checkpoint')
            self._save_model(logs)

    def on_train_batch_end(self, batch, logs=None):
        # update current and overall hours of speech
        self.current_batch = int(batch + 1)
        self.current_hours += self.hours_per_batch
        self.overall_hours += self.hours_per_batch

        # Check if the model should be saved according to the high and low frequency and custom checkpoints.
        # It might be cases where the saving cannot be done exactly at the desired frequency, so we check if the
        # current hours of speech are close enough to the desired frequency, given by the hours per batch.
        if self.current_hours % self.checkpoints_dict['frequency_low_resolution'] <= self.hours_per_batch \
                and self.current_hours - self.hours_per_batch <= self.checkpoints_dict['max_low_resolution']:
            logging.info('Saving low resolution checkpoint')
            self._save_model(logs)
        elif self.current_hours % self.checkpoints_dict['frequency_high_resolution'] <= self.hours_per_batch:
            logging.info('Saving high resolution checkpoint')
            self._save_model(logs)
        elif self.checkpoints_dict['custom_checkpoints'] and \
                self.current_hours % self.checkpoints_dict['custom_checkpoints'][0] <= self.hours_per_batch:
            logging.info(f'Saving custom checkpoint')
            self._save_model(logs)
            self.checkpoints_dict['custom_checkpoints'].pop(0)
        elif self.overall_hours % self.checkpoints_dict['overall_frequency'] <= self.hours_per_batch:
            logging.info('Saving overall checkpoint')
            self._save_model(logs)
        else:
            # Set logs for tensorboard logging if no checkpoint is saved
            logs = self._set_logs(logs)


class CustomBackUpAndRestore(BackupAndRestore):
    """
        This custom backup and restore allows to set the batch and epoch number on the beginning
        of the training in case there is not a backup and the model has started from a checkpoint.
    """

    def __init__(self, current_batch, starting_from_ckpt=False, **kwargs):
        super(CustomBackUpAndRestore, self).__init__(**kwargs)
        self.current_batch = current_batch
        self.starting_from_ckpt = starting_from_ckpt

    def on_train_begin(self, logs=None):
        super(CustomBackUpAndRestore, self).on_train_begin(logs)
        # If there is no backup yet
        if self.starting_from_ckpt and self._training_state._ckpt_saved_batch.numpy() == -1:
            backend.set_value(
                self._training_state._ckpt_saved_epoch, 0
            )
            backend.set_value(
                self._training_state._ckpt_saved_batch, self.current_batch
            )
            self._batches_count = self.current_batch % self.save_freq


class CustomTensorBoard(TensorBoard):
    """
        This custom tensorboard allows to plot batch train/val loss in the same plot.
        This also ensures that the train logs are written in case of an interruption.
    """

    def set_model(self, model):
        super(CustomTensorBoard, self).set_model(model)
        self._val_ori_dir = os.path.join(self._log_write_dir, "val_ori")
        self._val_ori_step = self.model._test_counter
        self._val_fil_dir = os.path.join(self._log_write_dir, "val_fil")
        self._val_fil_step = self.model._test_counter

        self._writers["val_ori"] = tf.summary.create_file_writer(
            self._val_ori_dir
        )
        self._val_ori_writer = self._writers["val_ori"]
        self._writers["val_fil"] = tf.summary.create_file_writer(
            self._val_fil_dir
        )
        self._val_fil_writer = self._writers["val_fil"]
        self.val_ori_done = False

    def on_train_batch_end(self, batch, logs=None):
        if self._should_write_train_graph:
            self._write_keras_model_train_graph()
            self._should_write_train_graph = False
        if self.write_steps_per_second:
            batch_run_time = time.time() - self._batch_start_time
            tf.summary.scalar(
                "batch_steps_per_second",
                1.0 / batch_run_time,
                step=self._train_step,
            )

        # Necessary to guarantee that the logs are written, otherwise the train log might not be written if the
        #  training is interrupted before the end of the epoch.
        if self._train_step % self.update_freq == 0:
            with tf.summary.record_if(True), self._train_writer.as_default():
                if isinstance(logs, dict):
                    for name, value in logs.items():
                        if 'val_' in name:
                            continue
                        tf.summary.scalar("batch_" + name, value, step=self._train_step)

        if not self._should_trace:
            return

        if self._is_tracing and self._global_train_batch >= self._stop_batch:
            self._stop_trace()

    def on_test_begin(self, logs=None):
        if self.val_ori_done:
            self._push_writer(self._val_fil_writer, self._val_fil_step)
        else:
            self._push_writer(self._val_ori_writer, self._val_ori_step)

    def on_test_end(self, logs=None):
        if self.model.optimizer and hasattr(self.model.optimizer, "iterations"):
            if self.val_ori_done:
                with tf.summary.record_if(True), self._val_fil_writer.as_default():
                    for name, value in logs.items():
                        tf.summary.scalar(
                            "evaluation_" + name + "_vs_iterations",
                            value,
                            step=self.model.optimizer.iterations.read_value(),
                        )
                        if self._train_step % self.update_freq == 0:
                            tf.summary.scalar("batch_" + name, value, step=self._train_step)
                self.val_ori_done = False
            else:
                with tf.summary.record_if(True), self._val_ori_writer.as_default():
                    for name, value in logs.items():
                        tf.summary.scalar(
                            "evaluation_" + name + "_vs_iterations",
                            value,
                            step=self.model.optimizer.iterations.read_value(),
                        )
                        if self._train_step % self.update_freq == 0:
                            tf.summary.scalar("batch_" + name, value, step=self._train_step)
                self.val_ori_done = True
        self._pop_writer()
