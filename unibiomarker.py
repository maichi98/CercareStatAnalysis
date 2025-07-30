from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.stats import pearsonr, spearmanr
from sklearn.calibration import calibration_curve

from biomarker import Biomarker
import constants
from delong import delong_roc_variance
from plots import plot_roc, plot_confusion_matrix, plot_sigmoid_with_ci, plot_calibration_curve, plot_proba_distribution
from metrics import compute_youden_cutoff, compute_metrics, get_auc_ci, print_metrics, compute_sigmoid_ci


class Unibiomarker(Biomarker):

    MISSINGNESS_CMAP = {
        "present": "#4A90E2",  # Blue for Present Data
        "missing": "#B22222"  # Crimson for Missing Data
    }

    LESION_VS_CONTROL_CMAP = {
    }

    def __init__(self, name, data, test_data=None):

        self.name = name
        if name not in constants.DICT_MARKERS:
            raise ValueError(f"Biomarker '{name}' not found in DICT_MARKERS ! ")

        params = list(constants.DICT_MARKERS[name].values())
        self.path = constants.DICT_MARKERS[name]["path"]
        self.control = constants.DICT_MARKERS[name]["control"]
        self.ratio = constants.DICT_MARKERS[name]["ratio"]

        super().__init__(params=params, data=data, test_data=test_data)

    def check_missing_data(self):

        features = self.params
        # Compute the missing percentages :
        train_missing = self.data[features].isnull().mean().mul(100).rename("Train Missing %")
        test_missing = self.test_data[features].isnull().mean().mul(100).rename("Test Missing %")
        missing_df = pd.concat([train_missing, test_missing], axis=1)

        # Count complete cases
        n_train, n_test = len(self.data), len(self.test_data)
        n_train_complete = self.data[features].dropna().shape[0]
        n_test_complete = self.test_data[features].dropna().shape[0]

        print(f"Fully usable rows in train: {n_train_complete}/{n_train} ({n_train_complete / n_train * 100:.1f}%)")
        print(f"Fully usable rows in test:  {n_test_complete}/{n_test} ({n_test_complete / n_test * 100:.1f}%)")

        # Display missing value comparison table
        display(missing_df.style.format("{:.1f}").set_caption("Missing Value Comparison (Train vs Test)"))

        # Prepare binary masks for heatmap: 0 = present, 1 = missing
        train_mask = self.data[features].isnull().astype(int)
        test_mask = self.test_data[features].isnull().astype(int)

        # Create colormap :
        cmap = ListedColormap([self.MISSINGNESS_CMAP["present"], self.MISSINGNESS_CMAP["missing"]])
        norm = BoundaryNorm([0, 0.5, 1], ncolors=2)

        def draw_missing_heatmap(ax, data_mask, title):

            sns.heatmap(data_mask, ax=ax, cmap=cmap, norm=norm, cbar=False)
            ax.set_title(title)
            ax.set_xlabel("Features")
            ax.set_ylabel("Samples")

        # Plot side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        draw_missing_heatmap(axs[0], train_mask, "Train Missing Values")
        draw_missing_heatmap(axs[1], test_mask, "Test Missing Values")

        # Add legend
        present_patch = mpatches.Patch(color=self.MISSINGNESS_CMAP["present"], label="Present")
        missing_patch = mpatches.Patch(color=self.MISSINGNESS_CMAP["missing"], label="Missing")
        fig.legend(handles=[present_patch, missing_patch], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

        plt.tight_layout()
        plt.show()

    def plot_diagnostic_scatter_and_ratio(self, target="Diagnosis"):

        df = self.data[[self.path, self.control, self.ratio, target]].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        palette = {k: v['color'] for k, v in constants.DIAGNOSIS_INFO.items()}
        colors = df[target].map(palette)
        axes[0].scatter(df[self.control], df[self.path], c=colors, alpha=0.6, edgecolor='k', linewidth=0.5)

        for label in sorted(df[target].unique()):
            axes[0].scatter([], [], c=palette[label], label=f"{target} = {label}")
        axes[0].set_title("Scatter: Path vs Control")
        axes[0].set_xlabel("Control")
        axes[0].set_ylabel("Path")
        axes[0].legend()

        axes[1].hist(df[self.ratio], bins=30, alpha=0.7, color='steelblue')
        axes[1].axvline(1.0, color='red', linestyle='--', label='Ratio = 1.0')
        axes[1].set_title("Histogram: Ratio Distribution")
        axes[1].set_xlabel("Ratio")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def evaluate_path_control_correlation(self, target="Diagnosis"):

        df = self.data[[self.path, self.control, self.ratio, target]].dropna()
        print("=" * 150)
        print(f"🔍 Analyzing Biomarker: {self.name}")
        print("=" * 150)

        # --- Correlation Analysis ---
        print("-" * 150)
        print("--- Step 1: Correlation Analysis between Path and Control ")
        print("-" * 150)
        pearson_corr, pearson_p = pearsonr(df[self.path], df[self.control])
        spearman_corr, spearman_p = spearmanr(df[self.path], df[self.control])
        print(f"- Pearson  correlation: r = {pearson_corr:.3f}, p = {pearson_p:.4f}")
        print(f"- Spearman correlation: r = {spearman_corr:.3f}, p = {spearman_p:.4f}\n")

        # Interpretation :
        print("-" * 150)
        print("--- Step 2: Interpretation Guidance ")
        print("-" * 150)

        if pearson_corr > 0.95 and pearson_p < 0.05:
            print("Path and Control are highly linearly correlated (Pearson > 0.95, p < 0.05).")
            print("The ratio is likely not informative ! ")

        elif pearson_p < 0.05 < spearman_p:
            print("Red Flag ! Pearson correlation is statistically significant (p < 0.05), but Spearman is not !")

        elif pearson_p > 0.05 > spearman_p:
            print("Spearman correlation is significant (p < 0.05), but Pearson is not ! ")
            print("This suggests a monotonic but non-linear relationship ! ")
            print("The ratio may still be useful in this case")

        elif pearson_p < 0.05 and spearman_p < 0.05:
            if pearson_corr > 0.6:
                print("Both Pearson and Spearman correlations are significant.")
                print("Moderate to strong linear and monotonic association detected.")
                print("There is linear correlation (>0.6), but not enough to discard the ratio as a feature")
            else:
                print("Both correlations are significant, but linear correlation is weak")
                print("The ratio is probably useful as a feature")

        elif pearson_p > 0.05 and spearman_p > 0.05:
            print("Neither Pearson nor Spearman correlation is statistically significant ! ")
            print("No clear association between Path and Control ! ")
            print("The ratio is a worthwhile feature to explore ! ")

        print("=" * 150)

    def evaluate_logistic_univariate_model(self, feature, target="Diagnosis", brier_bins_train=10, brier_bins_test=5, target_bin_count=100):

        # Prepare data
        df_train = self.data[[feature, target]].dropna()
        df_test = self.test_data[[feature, target]].dropna()

        x_train, y_train = df_train[feature], df_train[target]
        x_test, y_test = df_test[feature], df_test[target]

        model = LogisticRegression(solver="liblinear")
        model.fit(x_train, y_train)
        y_train_proba, y_test_proba = model.predict_proba(x_train)[:, 1], model.predict_proba(x_test)[:, 1]

        # ROC + optimal cutoff :
        fpr_train, tpr_train, best_idx, cutoff = compute_youden_cutoff(y_true=y_train, y_pred=y_train_proba)
        fpr_test, tpr_test, _, _ = compute_youden_cutoff(y_true=y_test, y_pred=y_test_proba)
        # predictions :
        y_train_pred_05, y_test_pred_05 = (y_train_proba >= 0.5).astype(int), (y_test_proba >= 0.5).astype(int)
        y_train_pred_youden, y_test_pred_youden = (y_train_proba >= cutoff).astype(int), (y_test_proba >= cutoff).astype(int)

        # AUCs and 95% CIs :
        auc_train, auc_cov_train = delong_roc_variance(y_train.to_numpy(), y_train_proba)
        ci_train = get_auc_ci(auc_train, auc_cov_train)
        auc_test, auc_cov_test = delong_roc_variance(y_test.to_numpy(), y_test_proba)
        ci_test = get_auc_ci(auc_test, auc_cov_test)

        # Metrics :
        train_metrics_05 = compute_metrics(y_true=y_train, y_pred=y_train_pred_05)
        train_metrics_youden = compute_metrics(y_true=y_train, y_pred=y_train_pred_youden)
        test_metrics_05 = compute_metrics(y_true=y_test, y_pred=y_test_pred_05)
        test_metrics_youden = compute_metrics(y_true=y_test, y_pred=y_test_pred_youden)

        # Log-odds and odds ratio:
        log_odds = model.coef_[0][0]
        odds_ratio = np.exp(log_odds)

        # Print Summary :
        print("=" * 100)
        print(f"\nLogistic Regression Summary for Feature: {feature}")
        print("-" * 100)
        print(f"Log-Odds Coefficient : {log_odds:.4f}")
        print(f"Odds Ratio           : {odds_ratio:.4f}")
        print(f"Youden's J Threshold : {cutoff:.4f}")
        print("-" * 100)

        print_metrics(f"[{feature}] Train @ 0.5", train_metrics_05)
        print_metrics(f"[{feature}] Train @ Youden", train_metrics_youden)
        print_metrics(f"[{feature}] Test @ 0.5", test_metrics_05)
        print_metrics(f"[{feature}] Test @ Youden", test_metrics_youden)

        # Plot the confusion matrices :
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        plot_confusion_matrix(ax=axes[0, 0], cm=train_metrics_05["cm"], title=f"[{feature}] Train @ 0.5")
        plot_confusion_matrix(ax=axes[0, 1], cm=train_metrics_youden["cm"], title=f"[{feature}] Train @ Youden")
        plot_confusion_matrix(ax=axes[1, 0], cm=test_metrics_05["cm"], title=f"[{feature}] Test @ 0.5")
        plot_confusion_matrix(ax=axes[1, 1], cm=test_metrics_youden["cm"], title=f"[{feature}] Test @ Youden")

        plt.tight_layout()
        plt.show()

        # Calibration + brier :
        brier_train = brier_score_loss(y_true=y_train, y_prob=y_train_proba)
        brier_test = brier_score_loss(y_true=y_test, y_prob=y_test_proba)
        prob_true_train, prob_pred_train = calibration_curve(y_true=y_train, y_prob=y_train_proba, n_bins=brier_bins_train)
        prob_true_test, prob_pred_test = calibration_curve(y_true=y_test, y_prob=y_test_proba, n_bins=brier_bins_test)

        # Sigmoid CI computation :
        x_range = np.linspace(x_train[feature].min(), x_train[feature].max(), 300)
        y_pred, y_lower, y_upper = compute_sigmoid_ci(x_train, y_train_proba, feature, model, x_range)

        # ROC + Sigmoid + Calibration plots :
        fig, axes = plt.subplots(4, 2, figsize=(16, 18))

        # === Row 0: ROC Curves ===
        plot_roc(ax=axes[0, 0], fpr=fpr_train, tpr=tpr_train, phase="Train", title=f"ROC Curve (Train) [{feature}]",
                 auc=auc_train, ci=ci_train, cutoff=cutoff, add_youden=True, best_idx=best_idx)

        plot_roc(ax=axes[0, 1], fpr=fpr_test, tpr=tpr_test, phase="Test", title=f"ROC Curve (Test) [{feature}]",
                 auc=auc_test, ci=ci_test, cutoff=cutoff)

        # === Row 1: Sigmoid Curves ===
        plot_sigmoid_with_ci(ax=axes[1, 0], x=x_train, y_proba=y_train_proba, y_true=y_train, x_range=x_range,
                             y_pred=y_pred, y_lower=y_lower, y_upper=y_upper, threshold=cutoff,
                             title=f"Fitted Sigmoid (Train) [{feature}]", feature=feature)

        plot_sigmoid_with_ci(ax=axes[1, 1], x=x_test, y_proba=y_test_proba, y_true=y_test, x_range=x_range,
                             y_pred=y_pred, y_lower=y_lower, y_upper=y_upper, threshold=cutoff,
                             title=f"Fitted Sigmoid (Test) [{feature}]", feature=feature)

        # === Row 2: Predicted Probability Distributions ===
        plot_proba_distribution(ax=axes[2, 0], y_proba=y_train_proba, y_true=y_train,
                                threshold=cutoff, title="Predicted Probability Distribution (Train)",
                                target_bin_count=target_bin_count)

        plot_proba_distribution(ax=axes[2, 1], y_proba=y_test_proba, y_true=y_test,
                                threshold=cutoff, title="Predicted Probability Distribution (Test)",
                                target_bin_count=target_bin_count)

        # === Row 3: Calibration Curves ===
        plot_calibration_curve(ax=axes[3, 0], prob_pred=prob_pred_train, prob_true=prob_true_train,
                               title=f"Calibration Curve (Train) [{feature}]", brier_score=brier_train)

        plot_calibration_curve(ax=axes[3, 1], prob_pred=prob_pred_test, prob_true=prob_true_test,
                               title=f"Calibration Curve (Test) [{feature}]", brier_score=brier_test)

        plt.tight_layout()
        plt.show()
