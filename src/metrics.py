"""One fixed-label metric implementation for all models and notebooks."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .common import labels


def score(y_true, y_pred):
    return {"n": len(y_true), "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels(), average="macro", zero_division=0))}


def detailed_outputs(y_true, y_pred, model_id, directory, prefix):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    order = labels()
    report = classification_report(y_true, y_pred, labels=order, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(directory / (prefix + "_per_class.csv"))
    matrix = confusion_matrix(y_true, y_pred, labels=order)
    pd.DataFrame(matrix, index=order, columns=order).to_csv(directory / (prefix + "_confusion.csv"))
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    shown = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(shown, ax=ax, fraction=0.04)
    ax.set(xticks=range(len(order)), yticks=range(len(order)), xticklabels=order,
           yticklabels=order, xlabel="Predicted intent", ylabel="True intent", title=model_id + " · " + prefix)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.savefig(directory / (prefix + "_confusion.png"), dpi=160)
    plt.close(fig)
