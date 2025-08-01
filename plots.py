import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import constants
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import calibration_curve
from matplotlib.colors import ListedColormap


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


def plot_sigmoid_with_ci(ax, x, y_proba, y_true, x_range, y_pred, y_lower, y_upper, threshold, title, feature):

    palette = {0: "#1f77b4", 1: "#d62728"}  # blue = 0, red = 1
    sns.scatterplot(x=x[feature], y=y_proba, hue=y_true, ax=ax, palette=palette, edgecolor='black', s=70)
    ax.plot(x_range, y_pred, color='black', label='Fitted Sigmoid')
    ax.fill_between(x_range, y_lower, y_upper, color='gray', alpha=0.3, label='95% CI')
    ax.axhline(threshold, color='red', linestyle='--', label=f"Youden Thresh = {threshold:.2f}")
    ax.set_title(title)
    ax.set_xlabel(feature)
    ax.set_ylabel("Predicted Probability")
    ax.legend()


def plot_calibration_curve(ax, prob_pred, prob_true, title, brier_score=None):

    ax.plot(prob_pred, prob_true, marker='o', label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    ax.set_title(title)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Proportion")
    ax.legend()

    if brier_score is not None:
        ax.text(0.05, 0.05, f"Brier = {brier_score:.3f}", transform=ax.transAxes, fontsize=10, color="black")


def plot_proba_distribution(ax, y_proba, y_true, threshold, title, target_bin_count=100):
    df = pd.DataFrame({
        "proba": y_proba,
        "true_label": y_true.astype(str)
    })

    palette = {"0": "#66c2a5", "1": "#fc8d62"}  # same consistent palette
    bins = np.linspace(0, 1, target_bin_count + 1)

    sns.histplot(data=df, x="proba", hue="true_label", bins=bins,
                 stat="density", element="step", common_norm=False, palette=palette, ax=ax)

    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")

    handles = [
        plt.Line2D([0], [0], color=palette["0"], label="Class 0"),
        plt.Line2D([0], [0], color=palette["1"], label="Class 1"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"Threshold = {threshold:.2f}")
    ]
    ax.legend(handles=handles, title="Legend")


def plot_logistic_decision_surface(ax, model, X_scaled, y, feature_names, threshold=0.5, cutoff=None, title=""):

    xx, yy = np.meshgrid(
        np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 200),
        np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    background_cmap = ListedColormap(["#deebf7", "#fee0d2"])   # soft blue and soft red
    point_colors = np.array(["#3182bd", "#de2d26"])  # strong blue, strong red

    # Probability background
    ax.contourf(xx, yy, probs, levels=25, cmap=background_cmap, alpha=0.8)

    # Decision boundaries
    ax.contour(xx, yy, probs, levels=[threshold], colors='black', linestyles='--', linewidths=2)
    if cutoff is not None:
        ax.contour(xx, yy, probs, levels=[cutoff], colors='green', linestyles='-', linewidths=2)

    # Data points
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], color=point_colors[y], edgecolor="k", s=30)

    # Labels and legend
    ax.set_title(title)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(handles=[
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=point_colors[0], label='Class 0', markersize=10),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=point_colors[1], label='Class 1', markersize=10),
        plt.Line2D([], [], color='black', linestyle='--', label='Threshold 0.5'),
        plt.Line2D([], [], color='green', linestyle='-', label="Youden's J")
    ])


def plot_svm_calibration_curve(ax, y_true, proba, margin, title):

    # Scale margin scores to [0, 1] for comparison
    margin_scaled = MinMaxScaler().fit_transform(margin.reshape(-1, 1)).flatten()

    # Compute calibration curves
    prob_true_margin, prob_pred_margin = calibration_curve(y_true, margin_scaled, n_bins=10)
    prob_true_proba, prob_pred_proba = calibration_curve(y_true, proba, n_bins=10)

    # Plot
    ax.plot(prob_pred_margin, prob_true_margin, marker='o', linestyle='-', label="(Margin)")
    ax.plot(prob_pred_proba, prob_true_proba, marker='s', linestyle='-', label="Platt-Calibrated")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_title(title)
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Empirical Frequency")
    ax.legend()


def plot_svm_decision_surfaces(ax_geom, ax_proba, model, X_scaled, y, feature_names, youden_f, youden_p, title_prefix):

    # Generate mesh grid
    xx, yy = np.meshgrid(
        np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 200),
        np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)
    P = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    bg_geom_cmap = ListedColormap(["#f0f9e8", "#bae4bc", "#7bccc4", "#2b8cbe"])  # blue-green gradient
    bg_proba_cmap = ListedColormap(["#deebf7", "#fee0d2"])  # soft blue/red for proba
    point_colors = np.array(["#3182bd", "#de2d26"])  # bold blue/red

    # === Geometric Plot (f(x)) ===
    ax_geom.contourf(xx, yy, Z, levels=25, cmap=bg_geom_cmap, alpha=0.8)
    ax_geom.contour(xx, yy, Z, levels=[0], colors='darkorange', linewidths=2)
    ax_geom.contour(xx, yy, Z, levels=[-1, 1], colors='darkorange', linestyles=':', linewidths=1)
    ax_geom.contour(xx, yy, Z, levels=[youden_f], colors='limegreen', linewidths=2)
    ax_geom.scatter(X_scaled[:, 0], X_scaled[:, 1], color=point_colors[y], edgecolor="k", s=30)
    ax_geom.set_title(f"{title_prefix} – Geometric (f(x))")
    ax_geom.set_xlabel(feature_names[0])
    ax_geom.set_ylabel(feature_names[1])
    ax_geom.legend(handles=[
        plt.Line2D([], [], color='darkorange', linestyle='-', label='f(x) = 0'),
        plt.Line2D([], [], color='darkorange', linestyle=':', label='f(x) = ±1'),
        plt.Line2D([], [], color='limegreen', linestyle='-', label=f'Youden f(x) = {youden_f:.2f}')
    ])

    # === Probabilistic Plot (P(x)) ===
    ax_proba.contourf(xx, yy, P, levels=25, cmap=bg_proba_cmap, alpha=0.8)
    ax_proba.contour(xx, yy, P, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    ax_proba.contour(xx, yy, P, levels=[youden_p], colors='limegreen', linewidths=2)
    ax_proba.scatter(X_scaled[:, 0], X_scaled[:, 1], color=point_colors[y], edgecolor="k", s=30)
    ax_proba.set_title(f"{title_prefix} – Probabilistic (P(x))")
    ax_proba.set_xlabel(feature_names[0])
    ax_proba.set_ylabel(feature_names[1])
    ax_proba.legend(handles=[
        plt.Line2D([], [], color='black', linestyle='--', label='P(x) = 0.5'),
        plt.Line2D([], [], color='limegreen', linestyle='-', label=f'Youden P(x) = {youden_p:.2f}')
    ])


def plot_embedding_scatter(x_embedded, y_labels, title, ax=None, style=None):

    label_map = {k: v["label"] for k, v in constants.DIAGNOSIS_INFO.items()}
    palette = {v["label"]: v["color"] for v in constants.DIAGNOSIS_INFO.values()}

    y_mapped = y_labels.map(label_map)

    scatter_args = {
        "x": x_embedded[:, 0],
        "y": x_embedded[:, 1],
        "hue": y_mapped,
        "palette": palette,
        "edgecolor": "black",
        "s": 80,
        "alpha": 0.8
    }

    if style is not None:
        scatter_args["style"] = style

    sns.scatterplot(**scatter_args, ax=ax)
    ax.set_title(title)
    ax.legend(title="Diagnosis" if style is None else "Diagnosis / Set")