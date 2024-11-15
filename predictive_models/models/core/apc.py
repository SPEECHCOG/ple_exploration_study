""""
@author María Andrea Cruz Blandón
@date 08.02.2022
Autoregressive Predictive Coding model
[An unsupervised autoregressive model for speech representation learning]

This corresponds to a translation from the Pytorch implementation
(https://github.com/iamyuanchung/Autoregressive-Predictive-Coding) to Keras implementation
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Add, Conv1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.core.model_base import ModelBase
from models.core.utils import AttentionWeights


class APCModel(ModelBase):

    def __init__(self, config: dict):
        super(APCModel, self).__init__(config)

        # Model architecture: PreNet (stacked linear layers with ReLU activation and dropout) ->
        # APC (multi-layer LSTM network) -> Postnet(Conv1D this is only used during training)

        # Load apc configuration parameters
        self.prenet = self.specific_params['prenet']
        self.prenet_layers = self.specific_params['prenet_layers']
        self.prenet_units = self.specific_params['prenet_units']
        self.prenet_dropout = self.specific_params['prenet_dropout']

        self.rnn_layers = self.specific_params['rnn_layers']
        self.rnn_units = self.specific_params['rnn_units']
        self.rnn_dropout = self.specific_params['rnn_dropout']

        self.residual = self.specific_params['residual']
        self.steps_shift = self.specific_params['steps_shift']

        # Input tensor (samples x frames x features)
        input_feats = Input(shape=(self.sample_size, self.features) , name='input_layer')

        rnn_input = input_feats

        if self.prenet:
            for i in range(self.prenet_layers):
                rnn_input = Dense(self.prenet_units, activation='relu', name='prenet_linear_' + str(i))(rnn_input)
                rnn_input = Dropout(self.prenet_dropout, name='prenet_dropout_' + str(i))(rnn_input)

        # RNN
        for i in range(self.rnn_layers):
            # TODO Padding for sequences is not yet implemented
            if i + 1 < self.rnn_layers:
                # All LSTM layers will have rnn_units units except last one
                rnn_output = LSTM(self.rnn_units, return_sequences=True, name='rnn_layer_' + str(i))(rnn_input)
            else:
                # Last LSTM layer will have latent_dimension units
                if self.residual and self.latent_dimension == self.rnn_units:
                    # The latent representation will be then the output of the residual connection.
                    rnn_output = LSTM(self.latent_dimension, return_sequences=True, name='rnn_layer_' + str(i))(
                        rnn_input)
                else:
                    rnn_output = LSTM(self.latent_dimension, return_sequences=True, name='latent_layer')(rnn_input)

            if i + 1 < self.rnn_layers and self.rnn_dropout:
                # Dropout to all layers except last layer
                rnn_output = Dropout(self.rnn_dropout, name='rnn_dropout_' + str(i))(rnn_output)

            if self.residual:
                # residual connection is applied to last layer if the latent dimension and rnn_units are the same,
                # otherwise is omitted. And to the first layer if the PreNet units and RNN units are the same,
                # otherwise is omitted also for first layer.
                residual_last = (i + 1 == self.rnn_layers and self.latent_dimension == self.rnn_units)
                residual_first = (i == 0 and self.prenet_units == self.rnn_units)

                if (i + 1 < self.rnn_layers and i != 0) or residual_first:
                    rnn_input = Add(name='rnn_residual_' + str(i))([rnn_input, rnn_output])

                # Update output for next layer (PostNet) if this is the last layer. This will also be the latent
                # representation.
                if residual_last:
                    rnn_output = Add(name='latent_layer')([rnn_input, rnn_output])

                # Update input for next layer
                if not residual_first and i == 0:
                    # Residual connection won't be applied but we need to update input value for next RNN layer
                    # to the output of RNN + dropout
                    rnn_input = rnn_output
            else:
                # Output of the dropout or RNN layer in the case of the last layer
                rnn_input = rnn_output

        if self.input_attention:
            # add self-attention layer after rnn output
            attention_output, _ = AttentionWeights(name='attention_layer', causal=True)([rnn_output, rnn_output])
            input_feats_att = Concatenate()([rnn_output, attention_output])

            # After PreNet
            rnn_output = input_feats_att

        # PostNet
        postnet_layer = Conv1D(self.features, kernel_size=1, padding='same', name='postnet_conv1d')(rnn_output)

        # APC Model
        self.model = Model(input_feats, postnet_layer)

    def train(self):
        """
        Train APC model, optimiser adam and loss L1 (mean absolute error)
        :return: an APC trained model. The model is also saved in the specified folder (output_path param in the
                 training configuration)
        """
        # Configuration of learning process
        adam = Adam(learning_rate=self.learning_rate, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss='mean_squared_error')

        # training
        super(APCModel, self).train()

        return self.model
