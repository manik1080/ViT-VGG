import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display
import pytest

import utils

def test_plot_class_distribution(tmp_path):
    utils.plot_class_distribution(['a','b','c'], [5,2,3])

def test_visualize_samples():
    data = np.random.rand(6, 8, 8, 3)
    labels = np.array([0,1,2,0,1,2])
    utils.visualize_samples(data, labels, samples_per_class=1)

def test_heatmap_and_helpers(tmp_path):
    matrix = np.array([[1,2],[3,4]])
    utils.heatmap(matrix, "t", "x", "y", ["c1","c2"], ["r1","r2"])

def test_classification_report_plot():
    # basic report for 2 classes
    from sklearn.metrics import classification_report
    y_true = np.array([0,1,1,0])
    y_pred = np.array([0,1,0,0])
    cr = classification_report(y_true, y_pred, target_names=['A','B'])
    utils.plot_classification_report(cr)

def test_multiclass_roc():
    y_true = np.eye(3)[[0,1,2,1]]
    y_score = np.random.rand(4,3)
    utils.plot_multiclass_roc(y_true, y_score, class_names=['X','Y','Z'])

def test_plot_training_history():
    class DummyHist:
        history = {"accuracy":[0.1,0.2], "val_accuracy":[0.1,0.15]}
    utils.plot_training_history(DummyHist())
