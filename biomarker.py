from IPython.display import display
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score

import constants
from delong import delong_roc_variance
from plots import plot_roc, plot_confusion_matrix, plot_distribution_with_cutoff
from metrics import compute_youden_cutoff, compute_metrics, get_auc_ci, print_metrics
from utils import get_adaptive_distribution_bins


class Biomarker:

    # Plotting palettes: Selected from seaborn color palettes,
    # See: https://seaborn.pydata.org/tutorial/color_palettes.html
    PALETTE_FEATURE_DISTRIBUTION = {
        "boxplot": "Set3",
        "kde": "Set1",
    }

    def __init__(self,
                 params,
                 data=None,
                 test_data=None):

        self.params = params if isinstance(params, list) else [params]
        self.data = data
        self.test_data = test_data

    def __repr__(self):

        lines = [
            f"Biomarker object",
            f"- Parameters: {', '.join(self.params)}",
            f"- Training samples: {len(self.data) if self.data is not None else 'None'}",
            f"- Test samples: {len(self.test_data) if self.test_data is not None else 'None'}\n"
        ]
        return "\n".join(lines)

    def describe_features(self, features, target="Diagnosis"):

        if target not in self.data.columns:
            raise ValueError(f"Target column '{target}' not found in data ! ")

        # --- Overall statistics ---
        print("Overall Descriptive Statistics : ")
        overall_stats = self.data[features].describe().T
        display(overall_stats)

        print(f"\nDescriptive Statistics by '{target}' Class:")

        # --- Compute class-wise stats ---
        grouped = self.data.groupby(target)
        summary_rows = []

        for class_label, group_df in grouped:
            stats = group_df[features].describe().T
            stats.insert(0, "Class", class_label)
            summary_rows.append(stats)

        df_result = pd.concat(summary_rows).reset_index().rename(columns={"index": "Feature"})

        # Custom sort order for features (you can adjust as needed)
        lesion_features = [f for f in self.params if "lésion" in f or "path" in f]
        control_features = [f for f in self.params if "control" in f]
        ratio_features = [f for f in self.params if "ratio" in f]

        feature_order = []
        for list_features in [lesion_features, control_features, ratio_features]:
            if len(list_features) > 0:
                feature_order.extend(list_features)

        df_result["Feature"] = pd.Categorical(df_result["Feature"], categories=feature_order, ordered=True)
        df_result = df_result.sort_values(by=["Feature", "Class"])

        # Final column order
        cols = ["Feature", "Class", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        df_result = df_result[cols]

        display(df_result)

    def plot_feature_distributions(self, features, plots=None):

        if plots is None:
            plots = ["boxplot", "kde"]

        n_features, n_plots = len(features), len(plots)
        fig = plt.figure(figsize=(6 * n_plots, 4 * n_features))

        for i, feature in enumerate(features):
            for j, plot_type in enumerate(plots):
                idx = i * n_plots + j + 1
                ax = fig.add_subplot(n_features, n_plots, idx)

                ax.set_title(f"{feature} by Diagnosis ({plot_type.capitalize()})")

                if plot_type == "boxplot":
                    sns.boxplot(
                        x="Diagnosis",
                        y=feature,
                        hue="Diagnosis",
                        data=self.data,
                        ax=ax,
                        palette=self.PALETTE_FEATURE_DISTRIBUTION["boxplot"],
                        dodge=False
                    )
                    ax.set_xlabel("Diagnosis")
                    ax.set_ylabel(feature)

                elif plot_type == "kde":
                    sns.kdeplot(
                        data=self.data,
                        x=feature,
                        hue="Diagnosis",
                        ax=ax,
                        fill=True,
                        common_norm=False,
                        palette=self.PALETTE_FEATURE_DISTRIBUTION["kde"],
                    )
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Density")

        plt.tight_layout()
        plt.show()

    def plot_qq_by_group(self, feature, target="Diagnosis"):

        df = self.data[[feature, target]].dropna()
        groups = df[target].unique()

        if len(groups) != 2:
            raise ValueError(f"Expected exactly 2 groups in '{target}', but found {len(groups)}.")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for i, group in enumerate(sorted(groups)):
            group_data = df[df[target] == group][feature]
            stats.probplot(group_data, dist="norm", plot=axes[i])
            axes[i].set_title(f"Q–Q Plot: {feature} for {constants.DIAGNOSIS_INFO[group]['label']}")

        plt.suptitle(f"Q–Q Plots for '{feature}' by {target}")
        plt.tight_layout()
        plt.show()

    def test_group_difference(self, feature, target='Diagnosis', alpha=0.05):

        df = self.data[[feature, target]].dropna()
        groups = sorted(df[target].unique())

        if len(groups) != 2:
            raise ValueError("Only two groups are supported.")

        values = {
            group: df.loc[df[target] == group, feature].values
            for group in groups
        }

        # Welch's T-test :
        t_stat, t_p = stats.ttest_ind(values[groups[0]], values[groups[1]], equal_var=False)

        # Mann-Whitney U test :
        mw_stat, mw_p = stats.mannwhitneyu(values[groups[0]], values[groups[1]], alternative='two-sided')

        # Conclusions :
        t_conclusion = "✅ Significant difference" if t_p < alpha else "❌ No significant difference"
        mw_conclusion = "✅ Significant difference" if mw_p < alpha else "❌ No significant difference"

        print("=" * 100)
        print("Group Comparison Results:")
        print("-" * 100)
        print(f"Welch’s t-test       → statistic = {t_stat:.4f}, p = {t_p:.4f}   → {t_conclusion}")
        print(f"Mann–Whitney U test  → statistic = {mw_stat:.4f}, p = {mw_p:.4f}   → {mw_conclusion}")
        print("=" * 100)

    def evaluate_feature_predictive_power(self, feature, target="Diagnosis", target_bin_count=100):

        # Prepare data
        df_train = self.data[[feature, target]].dropna()
        df_test = self.test_data[[feature, target]].dropna()

        x_train, y_train = df_train[feature], df_train[target]
        x_test, y_test = df_test[feature], df_test[target]

        # Flip feature if inversely predictive
        flip_score = False
        auc_check = roc_auc_score(y_train, x_train)
        if auc_check < 0.5:
            flip_score = True
            x_train, x_test = -x_train, -x_test

        # Compute ROC and optimal threshold
        fpr, tpr, best_idx, cutoff = compute_youden_cutoff(y_train, x_train)
        if flip_score:
            cutoff = -cutoff

        # Classifier rule logic
        positive_if_higher = tpr[best_idx] > fpr[best_idx]
        if positive_if_higher:
            y_train_pred = (df_train[feature] >= cutoff).astype(int)
            y_pred = (x_test >= cutoff).astype(int)
            rule = f"{feature} {'≥' if not flip_score else '≤'} {cutoff:.3f}"
        else:
            y_train_pred = (df_train[feature] <= cutoff).astype(int)
            y_pred = (x_test <= cutoff).astype(int)
            rule = f"{feature} {'≤' if not flip_score else '≥'} {cutoff:.3f}"

        # Train metrics and AUC
        train_metrics = compute_metrics(y_true=y_train, y_pred=y_train_pred)
        auc_train, auc_train_cov = delong_roc_variance(y_train.to_numpy(), x_train.to_numpy())
        ci_train = get_auc_ci(auc_train, auc_train_cov)

        # Test metrics and AUC
        fpr_test, tpr_test, _, _ = compute_youden_cutoff(y_test, x_test)
        test_metrics = compute_metrics(y_true=y_test, y_pred=y_pred)
        auc_test, auc_test_cov = delong_roc_variance(y_test.to_numpy(), x_test.to_numpy())
        ci_test = get_auc_ci(auc_test, auc_test_cov)

        # Print results :
        print("\n" + "=" * 80)
        print(f"🔎 [ROC-Based Classification] Feature: '{feature}'")
        print("-" * 80)
        print(f"AUC (Train) : {auc_train:.3f}  (95% CI: {ci_train[0]:.3f} – {ci_train[1]:.3f})")
        print(f"AUC (Test)  : {auc_test:.3f}  (95% CI: {ci_test[0]:.3f} – {ci_test[1]:.3f})\n")
        print(f" Optimal threshold (Youden’s J): {cutoff:.3f}")
        print(f" Classification Rule          : Class = 1 if {rule}")

        if flip_score:
            print("\n⚠️  Feature is inversely associated with the positive class (AUC < 0.5)")
            print("   → ROC and threshold computed using the negated feature.\n")

        print_metrics(f"TRAIN SET METRICS at threshold: {cutoff:.3f}", train_metrics)
        print_metrics(f"TEST SET METRICS at threshold: {cutoff:.3f}", test_metrics)
        print("=" * 80)

        # Add ROC curve,

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        plot_roc(ax=axes[0, 0], fpr=fpr, tpr=tpr,
                 phase="Train", title=f"ROC Curve (Train): {feature}",
                 auc=auc_train, ci=ci_train, cutoff=cutoff,
                 add_youden=True, best_idx=best_idx)

        bins = get_adaptive_distribution_bins(df_train, feature, target_bin_count=target_bin_count)

        plot_distribution_with_cutoff(ax=axes[0, 1], data=df_train,
                                      feature=feature, target=target,
                                      cutoff=cutoff, bins=bins,
                                      title=f"Train Distribution: {feature}")
        #
        plot_confusion_matrix(ax=axes[0, 2], cm=train_metrics["cm"],
                              title="Confusion Matrix (Train)")

        plot_roc(ax=axes[1, 0], fpr=fpr_test, tpr=tpr_test,
                 phase="Test", title=f"ROC Curve (Test): {feature}",
                 auc=auc_test, ci=ci_test, cutoff=cutoff)

        plot_distribution_with_cutoff(ax=axes[1, 1], data=df_test,
                                      feature=feature, target=target,
                                      cutoff=cutoff, bins=bins,
                                      title=f"Test Distribution: {feature}")

        plot_confusion_matrix(ax=axes[1, 2],
                              cm=test_metrics["cm"],
                              title="Confusion Matrix (Test)")

        plt.tight_layout()
        plt.show()
