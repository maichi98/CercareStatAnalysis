from IPython.display import display
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd

import constants


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
