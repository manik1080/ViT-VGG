import os
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataloader import load_data, encode_labels, get_class_counts
from models import FusionModel
from config import config
from utils import (
    plot_class_distribution,
    visualize_samples,
    plot_classification_report,
    plot_multiclass_roc,
    plot_training_history
)

def run_experiment(model, x_train, y_train, num_epochs, batch_size, validation_split):
    optimizer = 'adam'
    validation_split = 0.1
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
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


def main(save_data_plots=False):
    batch_size = 16
    num_epochs = 10
    validation_split = 0.1
    
    data, labels = load_data(config.parent_dir)
    classes, counts = get_class_counts(labels)

    plot_class_distribution(classes, counts)
    plt.savefig('class_distribution.png', bbox_inches='tight')
    plt.close()
    
    visualize_samples(data, labels, samples_per_class=1)
    plt.savefig('samples.png', bbox_inches='tight')
    plt.close()

    y_encoded, _ = encode_labels(labels)

    global X_test, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        data, y_encoded, test_size=0.2
    )

    d = {'x_train': X_train, 'x_test': X_test, 'y_train': y_train, 'y_test': y_test}
    for i in d:
        print(i, ' : ', d[i].shape, ' --> ', d[i].dtype)

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
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )

    model.save(config.model_save_dir)
    return model, history


if __name__ == '__main__':
    X_test = y_test = None
    model, history = main(True)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
    
    plot_training_history(history)
    plt.savefig('training_hist.png', bbox_inches='tight')
    plt.close()
    
    y_pred_prob = model.predict(X_test)
    y_pred_classes = y_pred_prob.argmax(axis=1)
    y_true_classes = y_test.argmax(axis=1)
    
    from sklearn.metrics import classification_report
    cr = classification_report(y_true_classes, y_pred_classes, target_names=[str(c) for c in classes])
    plot_classification_report(cr, title='Test Set Classification Report')
    plt.savefig('classification_report.png', bbox_inches='tight')
    plt.close()
    
    plot_multiclass_roc(y_test, y_pred_prob, class_names=[str(c) for c in classes])
    plt.savefig('multiclass_roc.png', bbox_inches='tight')
    plt.close()

    
    
