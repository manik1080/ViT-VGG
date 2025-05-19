import os
import pytest
from config import config

def test_paths_and_shapes_exist():
    # Attributes from config.py should exist
    attrs = ["parent_dir", "img_bin_path", "labels_bin_path",
             "image_width", "image_height", "image_channels",
             "num_classes", "input_shape",
             "vit_config", "vgg_config", "checkpoint_dir", "checkpoint_pattern"]
    for a in attrs:
        assert hasattr(config, a), f"config is missing '{a}'"

def test_paths_are_valid_strings():
    assert isinstance(config.parent_dir, str)
    assert os.path.basename(config.img_data_file).endswith('.bin')
    assert os.path.basename(config.labels_file).endswith('.bin')

def test_input_shape_consistency():
    w, h, c = config.image_width, config.image_height, config.image_channels
    assert config.input_shape == (w, h, c)
