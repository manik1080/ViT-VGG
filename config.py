import os

class Config:
    # paths
    parent_dir = '/content/drive/MyDrive/Apple Disease'

    # Data
    img_data_file = os.path.join(parent_dir, 'img_data.bin')
    labels_file = os.path.join(parent_dir, 'labels.bin')

    image_width = 128
    image_height = 128
    image_channels = 3
    num_classes = 4
    input_shape = (image_width, image_height, image_channels)

    # ViT config
    vit_config = {
        'num_heads': 8,
        'key_dim': 256,
        'patch_size': 32,
        'projection_dim': 256,
        'transformer_units': None,
        'mlp_head_units': None,
        'num_layers': 6,
        'dropout_rate': 0.1
    }

    # VGG config
    vgg_config = {
        'num_classes': num_classes
    }

    # Checkpoints
    checkpoint_dir = os.path.join(parent_dir, 'Assets', 'checkpoints')
    checkpoint_pattern = 'cp-{epoch:04d}.ckpt'
