"""Description
   -----------
This script includes all functions that are related to model
architecture generation, compilation and optimization
"""
import tensorflow as tf
from tensorflow.keras import models, layers, Model
from tensorflow.keras.optimizers import Adam
import importlib
import numpy as np
import os
from utils.loss import get_metrics
from classification_models.tfkeras import Classifiers
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Conv1D, Conv1DTranspose, ConvLSTM1D, Convolution1D, Convolution1DTranspose, \
#     DepthwiseConv1D, SeparableConv1D, SeparableConvolution1D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ConvLSTM2D, Convolution2D, Convolution2DTranspose, \
    DepthwiseConv2D, SeparableConv2D, SeparableConvolution2D
# from tensorflow.keras.layers import Conv3D, Conv3DTranspose, ConvLSTM3D, Convolution3D, Convolution3DTranspose


def bayesify(model, apply_to: list or None = None, dropout_prob=0.5):
    # Dense + all 2D-Conv-Layers
    APPLY_TO = [Dense, Conv2D, Conv2DTranspose, ConvLSTM2D, Convolution2D, Convolution2DTranspose,
                DepthwiseConv2D, SeparableConv2D, SeparableConvolution2D] if apply_to is None else apply_to
    # Dense + all Conv-Layers, including 1D+3D
    # APPLY_TO = [Dense, Conv1D, Conv1DTranspose, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, ConvLSTM1D,
    # ConvLSTM2D, ConvLSTM3D, Convolution1D, Convolution1DTranspose, Convolution2D, Convolution2DTranspose,
    # Convolution3D, Convolution3DTranspose, DepthwiseConv1D, DepthwiseConv2D,
    # SeparableConv1D, SeparableConv2D, SeparableConvolution1D, SeparableConvolution2D]

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            next_layer = node.outbound_layer.name
            if next_layer not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update({next_layer: [layer.name]})
            else:
                network_dict['input_layers_of'][next_layer].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if it matches the layerlist
        if type(layer) in APPLY_TO:
            x = Dropout(dropout_prob)(layer_input, training=True)
            x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    new_model = Model(inputs=model.inputs, outputs=model_outputs)
    new_model.save('temp.h5')
    new_model = load_model('temp.h5')
    os.remove('temp.h5')

    return new_model


class Classification(object):
    def __init__(self, model_name, dims, lr, loss, num_classes, train_backbone, num_labels, dropout=0,
                 bayesian=None):
        self.dims = dims
        self.num_classes = num_classes
        self.lr = lr
        self.train_backbone = train_backbone
        self.num_labels = num_labels
        self.bayesian = bayesian
        self.metric = get_metrics(self.num_classes, self.num_labels)
        if self.num_labels > 0:
            self.loss = 'binary_crossentropy'
        else:
            self.loss = loss
        # Converting the model name from string to a function
        if model_name in ['ResNet18', 'ResNet34', 'SEResNet18', 'SEResNet34', 'SEResNet50', 'SEResNeXt50', 'ResNeXt50']:
            _model_attr, _ = Classifiers.get(model_name.lower())
            _model = _model_attr(input_shape=(self.dims[0], self.dims[1], 3), weights='imagenet', include_top=False)
            pool = layers.GlobalAveragePooling2D()(_model.output)
            _model = models.Model(inputs=[_model.input], outputs=[pool])
        else:
            _model_attr = getattr(importlib.import_module("tensorflow.keras.applications"), model_name)
            _model = _model_attr(
                include_top=False,
                weights='imagenet',
                input_shape=(self.dims[0], self.dims[1], 3),
                pooling='avg',
                classes=self.num_classes
            )
        if 'EfficientNet' in model_name:
            x = layers.BatchNormalization()(_model.output)
            x = layers.Dropout(rate=0.2, name="top_dropout")(x)
            _model = models.Model(_model.inputs, x)

        _model.trainable = self.train_backbone
        last_layer = _model.layers[-1].output

        if dropout > 0:
            last_layer = layers.Dropout(rate=dropout, name="top_dropout")(last_layer)

        if self.num_labels > 0:
            output = layers.Dense(self.num_labels, activation='sigmoid')(last_layer)
        elif self.num_classes == 2:
            output = layers.Dense(1, activation='sigmoid')(last_layer)
        elif self.num_classes > 2:
            output = layers.Dense(self.num_classes, activation='softmax')(last_layer)
        else:
            raise Exception("Expected two classes or more")
        self.model = Model(inputs=_model.input, outputs=output)

        if self.bayesian:
            self.model = bayesify(self.model, dropout_prob=self.bayesian)
        self.optimizer = Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metric)

    def to_functional(self):
        if len(self.model.layers) == 2:
            backbone = self.model.layers[0]
            head = self.model.layers[1]
            x = backbone.output
            x = head(x)
            self.model = Model(inputs=backbone.input, outputs=x)
