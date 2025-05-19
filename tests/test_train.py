import numpy as np
import tensorflow as tf
import pytest

from tensorflow.keras import Input, Model
from train import run_experiment

def build_dummy_model(input_shape, num_classes):
    inp = Input(shape=input_shape)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(tf.keras.layers.Flatten()(inp))
    return Model(inp, out)

def test_run_experiment_minimal():
    # create tiny dataset
    x = np.random.rand(20, 8, 8, 1).astype(np.float32)
    # one-hot 2 classes
    y = tf.keras.utils.to_categorical(np.random.randint(0,2,20), num_classes=2)
    model = build_dummy_model((8,8,1), 2)
    history = run_experiment(model, x, y, num_epochs=1, batch_size=4, validation_split=0.2)
    # history.history should have keys
    assert "accuracy" in history.history
    assert "val_accuracy" in history.history
