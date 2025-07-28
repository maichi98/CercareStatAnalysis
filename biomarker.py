from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
