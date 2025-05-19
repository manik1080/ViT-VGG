import os
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataloader import load_data, encode_labels, get_class_counts
from models import FusionModel
from config import config


def run_experiment(model, x_train, y_train, num_epochs, batch_size, validation_split):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=config.optimizer,
        metrics=['accuracy']
    )
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    cp_path = os.path.join(config.checkpoint_dir, config.checkpoint_pattern)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        cp_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True
    )
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True
    )
    history = model.fit(
        x_train, y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[checkpoint_cb, early_cb]
    )
    return history


def main():
    data, labels = load_data(config.parent_dir)
    y_encoded, _ = encode_labels(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        data, y_encoded, test_size=0.2, random_state=42
    )

    inp = Input(shape=config.input_shape)
    out = FusionModel(
        input_shape=config.input_shape,
        vit_config=config.vit_config,
        vgg_config=config.vgg_config
    )(inp)
    model = Model(inp, out)
    model.summary()

    history = run_experiment(
        model,
        X_train, y_train,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split
    )

    model.save(os.path.join(config.drive_path, 'final_model.h5'))


if __name__ == '__main__':
    main()
