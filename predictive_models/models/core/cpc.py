"""
@author María Andrea Cruz Blandón
@date 08.02.2022
Contrastive Predictive Coding model
[Representation Learning with Contrastive Predictive Coding]
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Input, LSTM, Conv2DTranspose, Lambda, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.core.apc import AttentionWeights
from models.core.model_base import ModelBase
from models.core.utils import FeatureEncoder, ContrastiveLoss


class CPCModel(ModelBase):

    def __init__(self, config: dict):
        super(CPCModel, self).__init__(config)

        # Model architecture: Feature_Encoder -> Dropout -> LSTM -> Dropout

        # feature encoder params
        self.encoder_layers = self.specific_params['encoder_layers']
        self.encoder_units = self.specific_params['encoder_units']
        self.encoder_dropout = self.specific_params['encoder_dropout']

        # autoregressive model params
        self.gru_units = self.specific_params['gru_units']

        # contrastive loss params
        self.neg = self.specific_params['negative_samples']
        self.steps = self.specific_params['steps']

        # dropout and learning rate params
        self.dropout = self.specific_params['dropout']

        # Input tensor
        input_feats = Input(shape=(self.sample_size, self.features), name='input_layer')
        # Feature Encoder
        feature_encoder = FeatureEncoder(self.encoder_layers, self.encoder_units, self.encoder_dropout)
        encoder_features = feature_encoder(input_feats)

        # Dropout layer
        dropout_layer = Dropout(self.dropout, name='dropout_block')

        if self.dropout > 0:
            encoder_output = dropout_layer(encoder_features)

        # Autoregressive model
        autoregressive_model = LSTM(self.gru_units, return_sequences=True, name='autoregressive_layer')
        if self.dropout > 0:
            autoregressive_output = autoregressive_model(encoder_output)
        else:
            autoregressive_output = autoregressive_model(encoder_features)
        autoregressive_model2 = LSTM(self.gru_units, return_sequences=True, name='latent_layer')
        autoregressive_output = autoregressive_model2(autoregressive_output)

        if self.input_attention:
            # add self-attention layer after input features
            attention_output, _ = AttentionWeights(name='attention_layer', causal=True)([autoregressive_output,
                                                                                         autoregressive_output])
            autoregressive_output = attention_output

        if self.dropout > 0:
            autoregressive_output = dropout_layer(autoregressive_output)

        # Preparation for Contrastive loss
        # Linear transformation of latent representation into the vector space of context representations
        project_latents_layer = Conv1D(self.gru_units, kernel_size=1, strides=1, name='project_latent')
        true_latents = project_latents_layer(encoder_features)

        # Calculate the following steps using context_latent
        expand_context_layer = Lambda(lambda x: K.expand_dims(x, -1), name='expand_context')
        expanded_context_latents = expand_context_layer(autoregressive_output)
        project_steps_layer = Conv2DTranspose(self.steps, kernel_size=1, strides=1, name='project_steps')
        predictions = project_steps_layer(expanded_context_latents)

        contrastive_loss = ContrastiveLoss(self.neg, self.steps)
        contrastive_loss_output = contrastive_loss([true_latents, predictions])

        # Model
        self.model = Model(input_feats, contrastive_loss_output)

    def train(self):
        """
        Train a CPC model
        :return: a trained model saved on disk
        """
        adam = Adam(learning_rate=self.learning_rate, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss={'Contrastive_Loss': lambda y_true, y_pred: y_pred})

        # training
        super(CPCModel, self).train()

        return self.model
