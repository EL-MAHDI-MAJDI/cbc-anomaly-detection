import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)


def evaluate_model(model, X_train, y_train, X_test, y_test, label=""):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    metrics = {
        'Modèle': label,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'Temps (s)': round(train_time, 2)
    }
    return model, y_pred, metrics


def plot_confusion_matrices(configs, save_path=None, figsize=(11, 10)):
    n = len(configs)
    n_cols = 2
    n_rows = (n + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i, (y_true, y_pred, title) in enumerate(configs):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Normal', 'Anomalie'],
                    yticklabels=['Normal', 'Anomalie'])
        axes[i].set_title(title)
        axes[i].set_xlabel('Prédit')
        axes[i].set_ylabel('Réel')

    # cacher les axes vides s'il y en a
    for j in range(len(configs), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def results_table(results_list):
    return pd.DataFrame(results_list)