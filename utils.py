import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report


def plot_class_distribution(class_names, counts):
    """
    Plot a bar graph of class distribution.

    Args:
        class_names (list of str): Names of the classes.
        counts (list or np.ndarray): Frequency of each class.
    """
    positions = np.arange(len(class_names))
    fig, ax = plt.subplots()
    ax.bar(positions, counts)
    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Frequency')
    ax.set_title('Class Distribution')
    plt.tight_layout()
    plt.show()


def visualize_samples(data, labels, samples_per_class=1):
    """
    Display a grid of sample images—one per class by default.

    Args:
        data (np.ndarray): image array of shape (N, H, W, C).
        labels (np.ndarray): 1D array of integer labels.
        samples_per_class (int): Number of samples to show for each class.
    """
    unique_labels = np.unique(labels)
    n = len(unique_labels)
    cols = samples_per_class
n_rows = n
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3, n_rows * 3))
    for i, cls in enumerate(unique_labels):
        idxs = np.where(labels == cls)[0][:samples_per_class]
        for j, idx in enumerate(idxs):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            ax.imshow(data[idx])
            ax.set_title(f"Class {cls}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        text_color = (0, 0, 0) if np.all(color[:3] > 0.5) else (1, 1, 1)
        ax.text(x, y, fmt % value, ha="center", va="center", color=text_color, **kw)


def cm2inch(*tupl):
    """
    Convert centimeters to inches for figure sizing.
    """
    inch = 2.54
    vals = tupl[0] if isinstance(tupl[0], tuple) else tupl
    return tuple(v / inch for v in vals)


def heatmap(
    matrix,
    title,
    xlabel,
    ylabel,
    xticklabels,
    yticklabels,
    figure_size=(8, 6),
    cmap='RdBu'
):
    """
    Plot a heatmap with annotated cell values.

    Args:
        matrix (2D array): Data to visualize.
        title (str): Plot title.
        xlabel, ylabel (str): Axis labels.
        xticklabels, yticklabels (list): Labels for ticks.
        figure_size (tuple): Inches (width, height).
        cmap (str): Matplotlib colormap.
    """
    fig, ax = plt.subplots(figsize=figure_size)
    pc = ax.pcolor(matrix, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(pc)
    show_values(pc)
    plt.tight_layout()
    plt.show()


def plot_classification_report(cr_text, title='Classification report', cmap='RdBu'):
    """
    Plot a classification report (precision, recall, f1) as a heatmap.

    Args:
        cr_text (str): Output from sklearn.metrics.classification_report.
        title (str): Plot title.
    """
    lines = cr_text.strip().split('\n')
    lines = [l for l in lines if l]
    header = lines[0].split()
    classes = []
    values = []
    supports = []
    for line in lines[1:-3]:
        parts = line.split()
        cls = parts[0]
        nums = list(map(float, parts[1:-1]))
        sup = int(parts[-1])
        classes.append(cls)
        values.append(nums)
        supports.append(sup)
    xticks = header[:-1]
    yticks = [f"{cls} ({sup})" for cls, sup in zip(classes, supports)]
    heatmap(
        np.array(values),
        title,
        'Metrics',
        'Classes',
        xticks,
        yticks,
        figure_size=(8, len(classes)*0.5+2),
        cmap=cmap
    )


def plot_multiclass_roc(y_true, y_score, class_names=None):
    """
    Plot ROC curves and compute AUC.

    Args:
        y_true (array): True one-hot labels.
        y_score (array): Predicted probabilities.
        class_names (list of str, optional): Names for each class.
    """
    n_classes = y_true.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i in range(n_classes):
        label = f"Class {i}" if class_names is None else class_names[i]
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f"ROC {label} (AUC = {roc_auc[i]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
