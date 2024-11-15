"""
@author María Andrea Cruz Blandón
@date 24.05.2023
New models should inherit from this class so training and prediction can be done in the same way for
all the models and independently.
"""
import logging
import pathlib

from abc import ABC, abstractmethod
from datetime import datetime
import tensorflow.compat.v2 as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.eager import context
from tensorflow.keras.callbacks import TensorBoard, BackupAndRestore, CallbackList
from keras.engine import data_adapter
from models.core.utils import HourTrackingModelCheckpoint, CustomBackUpAndRestore, CustomTensorBoard
from data.data_generators import configure_args_ple_data_generator, PLEDataGenerator

__docformat__ = ['reStructuredText']
__all__ = ['ModelBase']


def _get_batch_resume_ckpt(folder: pathlib.Path):
    counter = tf.Variable(0, dtype=tf.int64)
    ckpt = tf.train.Checkpoint(train_counter=counter)
    ckpt.restore(tf.train.latest_checkpoint(folder.joinpath('chief')))
    batch = counter.numpy()
    return batch


class ModelBase(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        """
        Loads the configuration parameters from the configuration dictionary and assign values to model's attributes
        :param config: a dictionary with the configuration for training
        :return: instance will have the configuration parameters
        """
        # Configuration
        self.configuration = config

        # properties from configuration file for training
        self.features = self.configuration['input_features']['num_features']

        model_config = config['model']
        model_params_general = model_config['parameters']

        self.output_folder = model_config['output_path']
        self.simulations_folder = self.output_folder.parent
        self.starting_ckpt = model_config['checkpoint_path']  # The training will continue from a specific checkpoint
        self.profile = model_config['profile']
        # TODO this should be a parameter in the configuration file for setting attention mechanism
        self.input_attention = False

        self.model_type = model_params_general['type']
        self.batch_size = model_params_general['batch_size']
        self.latent_dimension = model_params_general['latent_dim']
        self.sample_size = model_params_general['sample_size']  # frames per sample
        self.learning_rate = model_params_general['learning_rate']

        # model specific parameters
        self.specific_params = model_params_general['specific']
        # steps to shift the input for APC model, this is needed for the data generator.
        self.steps_shift = self.specific_params['steps_shift']

        # backup & tensorboard configuration
        # The training got interrupted abruptly
        self.resume_ckpt = model_config['backup']['resume_checkpoint_name']
        self.backup_save_freq = model_config['backup']['save_freq']  # How often to save checkpoints
        self.tensorboard_freq = model_config['tensorboard']['update_freq']

        # checkpoints
        self.checkpoints_params = model_config['checkpoints']

        # Set seen speech hours from this execution and overall training to zero
        self.current_seen_speech = 0
        self.overall_seen_speech = 0

        self.model = None

    @abstractmethod
    def train(self):
        """
        It sets model's path and callbacks according to configuration provided.
        :return: trained model
        """
        # Model file name for checkpoint and log
        # timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

        # If training was interrupted abruptly, load model from backup
        if self.resume_ckpt:
            timestamp = self.resume_ckpt

        # Check if model should be loaded from a checkpoint
        if self.starting_ckpt:
            # TODO: this should work like the backup and restore callback. that is loading a training_state that way
            #  the model knows the batch and epoch number
            # load model from checkpoint
            self.model.load_weights(self.starting_ckpt)
            logging.info(f'Model loaded from checkpoint: {self.starting_ckpt}')
            # load seen speech hours from checkpoint
            self.overall_seen_speech = float(self.starting_ckpt.stem.split('_')[-2])

        # Callbacks for training
        model_path = self.output_folder.joinpath(timestamp)
        logging.info(f'Model path: {model_path}')

        # Data generators
        val_ori_filenames, val_ori_indices = configure_args_ple_data_generator(self.configuration, validation=True,
                                                                               filtered_val=False)
        val_ori_generator = PLEDataGenerator(val_ori_filenames, val_ori_indices, self.batch_size, self.features,
                                             self.sample_size, self.model_type, self.steps_shift)
        val_fil_filenames, val_fil_indices = configure_args_ple_data_generator(self.configuration, validation=True,
                                                                               filtered_val=True)
        val_fil_generator = PLEDataGenerator(val_fil_filenames, val_fil_indices, self.batch_size, self.features,
                                             self.sample_size, self.model_type, self.steps_shift)
        train_filenames, train_indices = configure_args_ple_data_generator(self.configuration, validation=False)
        train_generator = PLEDataGenerator(train_filenames, train_indices, self.batch_size, self.features,
                                           self.sample_size, self.model_type, self.steps_shift)

        # Before starting training, the generators need to be exhausted until the number of batches seen if the
        # model was restored from a backup or a checkpoint. This is because the batches are not shuffled.
        current_batch = -1
        if self.starting_ckpt:
            current_batch = int(self.starting_ckpt.stem.split('_')[-1])
        if self.resume_ckpt:  # If starting from ckpt and backup, the backup has priority.
            current_batch = _get_batch_resume_ckpt(model_path)
        # Exhaust train generator
        if current_batch > 0:
            train_generator.set_prev_batch_number(current_batch - 1)

        # Back up and restore
        back_up_and_restore = CustomBackUpAndRestore(current_batch,  # Following tf, non execution will be -1
                                                     starting_from_ckpt=True if self.starting_ckpt else False,
                                                     backup_dir=model_path,
                                                     save_freq=self.backup_save_freq,
                                                     delete_checkpoint=True)
        # Model checkpoints
        # sample size given in number of frames of 10 ms
        hours_per_batch = round(self.batch_size * (self.sample_size / 100) / 3600, 3)
        checkpoints = HourTrackingModelCheckpoint(checkpoints_dict=self.checkpoints_params,
                                                  overall_hours=self.overall_seen_speech,
                                                  hours_per_batch=hours_per_batch,
                                                  filepath=model_path.joinpath('model_{current_hours:.1f}_'
                                                                               '{overall_hours:.1f}_'
                                                                               '{current_batch}.h5'),
                                                  current_batch=current_batch if current_batch > 0 else 0,
                                                  from_backup=True if self.resume_ckpt else False,
                                                  batch_ckpt=0 if not self.starting_ckpt else int(
                                                      self.starting_ckpt.stem.split('_')[-1]),
                                                  verbose=1)

        # Tensorboard log directory
        log_dir = self.simulations_folder.joinpath('logs').joinpath(model_path.relative_to(self.simulations_folder))
        tensorboard = CustomTensorBoard(log_dir=log_dir,
                                        write_graph=True,
                                        update_freq=self.tensorboard_freq)

        # Order is important, back_up_and_restore will recover the batch/epoch numbers, then the checkpoints will set
        # the current hours and overall hours of speech seen, and finally tensorboard will record loss and hours seen
        # in the same log file.
        callbacks = CallbackList([back_up_and_restore, checkpoints, tensorboard],
                                 add_history=True,
                                 add_progbar=True,
                                 model=self.model,
                                 verbose=1,
                                 epochs=1,
                                 steps=len(train_generator))

        # This procedure, it is as in model.fit, but since we need to validate after certain number of batches,
        # we need to do it manually.
        data_handler_tr = data_adapter.get_data_handler(
            train_generator,
            batch_size=self.batch_size,
            steps_per_epoch=len(train_generator),
            shuffle=False,
            model=self.model,
            steps_per_execution=self.model._steps_per_execution,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )

        self.model.stop_training = False
        self.model.train_function = self.model.make_train_function()
        if self.starting_ckpt and not self.resume_ckpt:
            self.model._train_counter.assign(current_batch)
        else:
            self.model._train_counter.assign(0)
        callbacks.on_train_begin()
        data_handler_tr._initial_epoch = 0
        # Initial step should be 0 if we start from scratch
        data_handler_tr._initial_step = 0 if current_batch < 0 else current_batch
        logs = None
        for epoch, iterator in data_handler_tr.enumerate_epochs():
            self.model.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            with data_handler_tr.catch_stop_iteration():
                for step in data_handler_tr.steps():
                    with tf.profiler.experimental.Trace(
                            "train",
                            epoch_num=epoch,
                            step_num=step,
                            batch_size=self.batch_size,
                            _r=1,
                    ):
                        callbacks.on_train_batch_begin(step)
                        tmp_logs = self.model.train_function(iterator)
                        if data_handler_tr.should_sync:
                            context.async_wait()
                        logs = tmp_logs
                        end_step = step + data_handler_tr.step_increment

                        if self.model.stop_training:
                            break

                        # Validate model if needed according to the tensorboard frequency, or if it is the last batch.
                        if self.model._train_counter % self.tensorboard_freq == 0 or self.model._train_counter == len(
                                train_generator):
                            if getattr(self.model, "_eval_ori_data_handler", None) is None:
                                self.model._eval_ori_data_handler = data_adapter.get_data_handler(
                                    val_ori_generator,
                                    batch_size=self.batch_size,
                                    initial_epoch=0,
                                    epochs=1,
                                    model=self.model,
                                    shuffle=False,
                                )
                            if getattr(self.model, "_eval_fil_data_handler", None) is None:
                                self.model._eval_fil_data_handler = data_adapter.get_data_handler(
                                    val_fil_generator,
                                    batch_size=self.batch_size,
                                    initial_epoch=0,
                                    epochs=1,
                                    model=self.model,
                                    shuffle=False,
                                )
                            # TODO notify tensorboard callback of which validation set is being run. Currently, it is
                            #  so it assumed that original signal is the first one.
                            val_ori_logs = self.model.evaluate(
                                x=val_ori_generator,
                                batch_size=self.batch_size,
                                callbacks=callbacks,
                                return_dict=True,
                                _use_cached_eval_dataset=True,
                            )
                            val_fil_logs = self.model.evaluate(
                                x=val_fil_generator,
                                batch_size=self.batch_size,
                                callbacks=callbacks,
                                return_dict=True,
                                _use_cached_eval_dataset=True,
                            )
                            val_ori_logs = {
                                "val_ori_" + name: val for name, val in val_ori_logs.items()
                            }
                            val_fil_logs = {
                                "val_fil_" + name: val for name, val in val_fil_logs.items()
                            }
                            logs.update(val_ori_logs)
                            logs.update(val_fil_logs)
                        callbacks.on_train_batch_end(end_step, logs)
            callbacks.on_epoch_end(epoch, logs)
        if isinstance(self.model.optimizer, Optimizer):
            self.model.optimizer.finalize_variable_values(
                self.model.trainable_variables
            )
        if getattr(self.model, "_eval_ori_data_handler", None) is not None:
            del self.model._eval_ori_data_handler
        if getattr(self.model, "_eval_fil_data_handler", None) is not None:
            del self.model._eval_fil_data_handler
        callbacks.on_train_end(logs)
