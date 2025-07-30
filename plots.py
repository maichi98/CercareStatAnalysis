import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import constants


# Colormap reference from matplotlib: https://matplotlib.org/stable/gallery/color/colormap_reference.html
CONFUSION_MATRIX_CMAP = "bwr"

PALETTE_ROC = {
    "Train": "blue",
    "Test": "green",
    "random": "gray",
    "youden": "red"
}


def plot_confusion_matrix(ax,
                          cm,
                          title,
                          labels=(0, 1)):

    df_cm = pd.DataFrame(cm,
                         index=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
                         columns=[f"Pred {labels[0]}", f"Pred {labels[1]}"])

    sns.heatmap(df_cm,
                annot=True,
                fmt="d",
                cmap=CONFUSION_MATRIX_CMAP,
                cbar=False,
                ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def plot_roc(ax,
             phase,
             fpr,
             tpr,
             auc,
             ci,
             title,
             cutoff=None,
             add_youden=False,
             best_idx=None,
             roc_label=None,
             youden_label=None):

    roc_label = roc_label or f"{phase} AUC = {auc:.3f} (95% CI: {ci[0]:.2f}–{ci[1]:.2f})"

    ax.plot(fpr, tpr, label=roc_label, color=PALETTE_ROC[phase])
    ax.plot([0, 1], [0, 1], linestyle='--', color=PALETTE_ROC["random"])

    if add_youden:
        youden_label = youden_label or f"Youden Cutoff (Train): {cutoff:.2f}"
        ax.axvline(fpr[best_idx], color='red', linestyle='--', label=youden_label)

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)


def plot_distribution_with_cutoff(ax,
                                  data,
                                  feature,
                                  target,
                                  cutoff,
                                  title,
                                  bins):
    data = data.copy()
    data[target] = data[target]

    palette = {k: v['histogram_color'] for k, v in constants.DIAGNOSIS_INFO.items()}

    sns.histplot(data=data,
                 x=feature,
                 hue=target,
                 element="step",
                 stat="density",
                 common_norm=False,
                 palette=palette,
                 ax=ax,
                 bins=bins)

    ax.axvline(cutoff, color='red', linestyle='--', label=f"Cutoff = {cutoff:.2f}")
    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")

    # Custom legend handles to clarify which class has which color
    handles = [
        plt.Line2D([0], [0], color=palette[0], label=f"{target} = 0"),
        plt.Line2D([0], [0], color=palette[1], label=f"{target} = 1"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"Cutoff = {cutoff:.2f}")
    ]
    ax.legend(handles=handles, title="Legend")
