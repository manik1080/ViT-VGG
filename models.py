import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from config import config


def create_mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, (batch_size, -1, patch_dims))
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        embedded = self.projection(patch) + self.position_embedding(positions)
        return embedded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


class VisualTransformer(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        num_heads=8,
        key_dim=256,
        patch_size=32,
        projection_dim=256,
        transformer_units=None,
        mlp_head_units=None,
        num_layers=6,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # compute number of patches
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.patches = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim)
        self.transformer_layers = []
        self.transformer_units = (
            [projection_dim * 2, projection_dim] if transformer_units is None else transformer_units
        )
        self.mlp_head_units = (
            [1024, 512] if mlp_head_units is None else mlp_head_units
        )
        for _ in range(num_layers):
            self.transformer_layers.append({
                'norm1': tf.keras.layers.LayerNormalization(epsilon=1e-6),
                'mha': tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim),
                'add1': tf.keras.layers.Add(),
                'norm2': tf.keras.layers.LayerNormalization(epsilon=1e-6),
                'mlp': lambda x: create_mlp(x, self.transformer_units, dropout_rate),
                'add2': tf.keras.layers.Add(),
            })
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.mlp_head = lambda x: create_mlp(x, self.mlp_head_units, dropout_rate)

    def call(self, inputs):
        x = self.patches(inputs)
        x = self.encoder(x)
        # transformer blocks
        for layer in self.transformer_layers:
            y = layer['norm1'](x)
            y = layer['mha'](y, y)
            x = layer['add1']([y, x])
            y = layer['norm2'](x)
            y = layer['mlp'](y)
            x = layer['add2']([y, x])
        x = self.final_norm(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.mlp_head(x)


class vgg_16_pretrained(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(input_shape[0], input_shape[1], input_shape[2])
        )
        self.pool = tf.keras.layers.GlobalMaxPooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.base(inputs)
        x = self.pool(x)
        return self.classifier(x)


class ViT_VGG(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        vit_config,
        vgg_config,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vit = VisualTransformer(input_shape, **vit_config)
        self.vgg = vgg_16_pretrained(input_shape, **vgg_config)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.fusion_dense = tf.keras.layers.Dense(1024, activation='relu')
        self.fusion_dropout = tf.keras.layers.Dropout(0.2)
        self.final_dense = tf.keras.layers.Dense(512, activation='relu')
        self.output_layer = tf.keras.layers.Dense(vgg_config['num_classes'], activation='softmax')

    def call(self, inputs):
        vgg_out = self.vgg(inputs)
        vit_out = self.vit(inputs)
        x = self.concat([vgg_out, vit_out])
        x = self.fusion_dense(x)
        # x = self.fusion_dropout(x)
        # x = self.final_dense(x)
        # x = self.fusion_dropout(x)
        return self.output_layer(x)

if __name__ == '__main__':
    vit_cfg = {
        "num_heads": 8,
        "key_dim": 256,
        "patch_size": 32,
        "projection_dim": 256,
        "transformer_units": None,
        "mlp_head_units": None
    }
    
    vgg_cfg = {
        "num_classes": config.num_classes
    }
    
    vit = VisualTransformer(config.input_shape, **vit_cfg)
    vgg = vgg_16_pretrained(config.input_shape, **vgg_cfg)
    patches = Patches(32)
    patch_encoder = PatchEncoder(16, 256)
    model = ViT_VGG(
        input_shape=config.input_shape,
        vit_config=vit_cfg,
        vgg_config=vgg_cfg
    )
    
    dummy = tf.zeros((1, *config.input_shape))
    _ = patches(dummy)
    _ = patch_encoder(patches(dummy))
    _ = vit(dummy)
    _ = vgg(dummy)
    _ = model(dummy)
    
    vit.summary(expand_nested=True)
    vgg.summary(expand_nested=True)
    model.summary(expand_nested=True)
