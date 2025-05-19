import numpy as np
import tensorflow as tf
import pytest

from models import (
    create_mlp, Patches, PatchEncoder,
    VisualTransformer, vgg_16_pretrained, ViT_VGG
)
from config import config

def test_create_mlp_returns_tensor():
    inp = tf.keras.Input(shape=(10,))
    out = create_mlp(inp, hidden_units=[5, 3], dropout_rate=0.0)
    model = tf.keras.Model(inp, out)
    x = np.random.rand(2,10).astype(np.float32)
    y = model(x)
    assert y.shape == (2,3)

def test_patches_and_config():
    layer = Patches(patch_size=16)
    dummy = tf.zeros((1, 32, 32, 3))
    patches = layer(dummy)
    # Expect (1, num_patches, patch_dim)
    assert patches.ndim == 3
    cfg = layer.get_config()
    assert cfg["patch_size"] == 16

def test_patch_encoder_and_config():
    num_patches = 4
    layer = PatchEncoder(num_patches, projection_dim=8)
    dummy = tf.random.uniform((1, num_patches, 32))
    encoded = layer(dummy)
    assert encoded.shape == (1, num_patches, 8)
    cfg = layer.get_config()
    assert cfg["num_patches"] == num_patches

def test_visual_transformer_forward():
    inp_shape = config.input_shape
    vt = VisualTransformer(inp_shape,
                           num_heads=2, key_dim=8,
                           patch_size=16, projection_dim=8,
                           num_layers=1, dropout_rate=0.0)
    x = tf.zeros((2, *inp_shape))
    out = vt(x)
    # mlp_head_units default [1024,512] → final shape (batch,512)
    assert out.shape == (2, 512)

def test_vgg16_pretrained_forward():
    inp_shape = config.input_shape
    num_classes = config.num_classes
    vgg = vgg_16_pretrained(inp_shape, num_classes)
    x = tf.zeros((2, *inp_shape))
    out = vgg(x)
    assert out.shape == (2, num_classes)

def test_vit_vgg_fusion_forward():
    inp_shape = config.input_shape
    vit_cfg = config.vit_config
    vgg_cfg = config.vgg_config
    fusion = ViT_VGG(inp_shape, vit_cfg, vgg_cfg)
    x = tf.zeros((3, *inp_shape))
    out = fusion(x)
    assert out.shape == (3, vgg_cfg["num_classes"])
