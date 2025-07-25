from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from biomarker import Biomarker
import constants


class Unibiomarker(Biomarker):

    MISSINGNESS_CMAP = {
        "present": "#4A90E2",  # Blue for Present Data
        "missing": "#B22222"  # Crimson for Missing Data
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

    def analyze_path_control_ratio(self, target="Diagnosis"):

        df = self.data[[self.path, self.control, self.ratio, target]].dropna()
        print("=" * 80)
        print(f"🔍 Analyzing Biomarker: {self.name}")
        print("=" * 80)

        # --- Correlation Analysis ---
        print("-- Step 1: Correlation Analysis between Path and Control " + "-" * 22)
        pearson_corr, pearson_p = pearsonr(df[self.path], df[self.control])
        spearman_corr, spearman_p = spearmanr(df[self.path], df[self.control])
        print(f"- Pearson  correlation: r = {pearson_corr:.3f}, p = {pearson_p:.4f}")
        print(f"- Spearman correlation: r = {spearman_corr:.3f}, p = {spearman_p:.4f}\n")

        # --- Ratio distribution summary ---
        print("-- Step 2: Summary Statistics for Ratio " + "-" * 41)
        self.describe_features(features=[self.ratio])

        # --- Scatter plot and histogram ---
        print("-- Step 3: Visualize Relationships " + "-" * 45)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        palette = {0: 'steelblue', 1: 'indianred'}
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

        # --- Boxplot + KDE using improved method ---
        print("-- Step 4: Ratio Distributions by Diagnosis " + "-" * 40)
        self.plot_feature_distributions(features=[self.ratio], plots=["boxplot", "kde"])

        # Interpretation :
        print("-- Step 5: Interpretation Guidance " + "-" * 47)

        if pearson_p > 0.05:
            print("Pearson correlation is not statistically significant (p > 0.05) ! ")

        elif pearson_corr > 0.95:
            print("Path and control are highly linearly correlated (Pearson > 0.95, p < 0.05) ! ")
            print("The Ratio is likely redundant and not informative")

        elif spearman_p > 0.05:
            print("Spearman correlation is not statistically significant (p > 0.05) !")

        elif spearman_corr > 0.9:
            print("Path and control are strongly monotonically related (Spearman > 0.9, p < 0.05) ! ")
            print("The Ratio may still offer useful non-linear contrast")
        else:
            print("Path and control are weakly correlated, with statistically significant results ! ")
            print("The Ratio may introduce meaningful variation and is worth exploring")

        print("=" * 80)
