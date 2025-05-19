import os
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import config


def load_data():
    data_path = config.data_bin_path
    label_path = config.labels_bin_path
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    return data, labels


def encode_labels(labels, save_encoder_path=None):
    labels = labels.reshape(-1, 1)
    ohe = OneHotEncoder(sparse=False)
    label_transformed = ohe.fit_transform(labels)
    if save_encoder_path is not None:
        with open(save_encoder_path, 'wb') as f:
            pickle.dump(ohe, f)
    return label_transformed, ohe


def get_class_counts(labels):
    classes, counts = np.unique(labels, return_counts=True)
    return classes, counts
