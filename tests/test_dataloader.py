import os
import pickle
import numpy as np
import tempfile
import pytest

import dataloader
from sklearn.preprocessing import OneHotEncoder

@pytest.fixture
def synthetic_data(tmp_path):
    # Create fake binary files
    data = np.random.rand(10, 128, 128, 3).astype(np.float32)
    labels = np.random.randint(0, 4, size=(10,))
    data_file = tmp_path / "img_data.bin"
    label_file = tmp_path / "labels.bin"
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)
    # Monkeypatch config paths
    from config import config
    config.data_bin_path = str(data_file)
    config.labels_bin_path = str(label_file)
    return data, labels

def test_load_data(synthetic_data):
    data, labels = synthetic_data
    loaded_data, loaded_labels = dataloader.load_data()
    assert isinstance(loaded_data, np.ndarray)
    assert isinstance(loaded_labels, np.ndarray)
    np.testing.assert_array_equal(loaded_data, data)
    np.testing.assert_array_equal(loaded_labels, labels)

def test_encode_labels_no_save(tmp_path):
    _, labels = synthetic_data(tmp_path)
    transformed, encoder = dataloader.encode_labels(labels)
    assert transformed.shape == (labels.size, 4)
    assert isinstance(encoder, OneHotEncoder)

def test_encode_labels_with_save(tmp_path):
    _, labels = synthetic_data(tmp_path)
    save_file = tmp_path / "ohe.pkl"
    transformed, encoder = dataloader.encode_labels(labels, str(save_file))
    assert save_file.exists()
    with open(save_file, 'rb') as f:
        obj = pickle.load(f)
    assert isinstance(obj, OneHotEncoder)

def test_get_class_counts():
    labels = np.array([0,1,1,2,2,2])
    classes, counts = dataloader.get_class_counts(labels)
    assert list(classes) == [0,1,2]
    assert list(counts)    == [1,2,3]
