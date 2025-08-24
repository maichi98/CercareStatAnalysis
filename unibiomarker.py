from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from IPython.display import display
from itertools import combinations
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import sklearn

from metrics import compute_youden_cutoff, compute_metrics, get_auc_ci, print_metrics, compute_sigmoid_ci
from delong import delong_roc_variance
from biomarker import Biomarker
from plots import *


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

        x_train, y_train = df_train[[feature]], df_train[target]
        x_test, y_test = df_test[[feature]], df_test[target]

        model = LogisticRegression(penalty=None,)

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

        # Additional stat inference using statmodels :
        x_train_sm = sm.add_constant(x_train)
        logit_model = sm.Logit(y_train, x_train_sm).fit(disp=0)

        # coef = logit_model.params[1]
        # se = logit_model.bse[1]
        # pval = logit_model.pvalues[1]
        # ci_low, ci_high = logit_model.conf_int().iloc[1]

        coef = logit_model.params.iloc[1]
        se = logit_model.bse.iloc[1]
        pval = logit_model.pvalues.iloc[1]
        ci_low, ci_high = logit_model.conf_int().iloc[1]

        # Convert to odds ratio scale
        odds_ratio_sm = np.exp(coef)
        ci_low_exp, ci_high_exp = np.exp(ci_low), np.exp(ci_high)

        print("\n--- Inference Using statsmodels ---")
        print(f"Odds Ratio (statsmodels)  : {odds_ratio_sm:.4f}")
        print(f"95% CI for OR             : ({ci_low_exp:.4f}, {ci_high_exp:.4f})")
        print(f"p-value                   : {pval:.4e}")
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
        #
        # === Row 3: Calibration Curves ===
        plot_calibration_curve(ax=axes[3, 0], prob_pred=prob_pred_train, prob_true=prob_true_train,
                               title=f"Calibration Curve (Train) [{feature}]", brier_score=brier_train)

        plot_calibration_curve(ax=axes[3, 1], prob_pred=prob_pred_test, prob_true=prob_true_test,
                               title=f"Calibration Curve (Test) [{feature}]", brier_score=brier_test)

        plt.tight_layout()
        plt.show()

    def evaluate_logistic_bivariate_model(self, target="Diagnosis", brier_bins_train=10, brier_bins_test=5, target_bin_count=100):

        features = [self.path, self.control]

        df_train, df_test = self.data[[*features, target]].dropna(), self.test_data[[*features, target]].dropna()
        x_train, y_train, x_test, y_test = df_train[features], df_train[target], df_test[features], df_test[target]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = LogisticRegression(penalty=None,)
        model.fit(x_train_scaled, y_train)
        y_train_proba, y_test_proba = model.predict_proba(x_train_scaled)[:, 1], model.predict_proba(x_test_scaled)[:, 1]

        # ROC + Youden
        fpr_train, tpr_train, best_idx, cutoff = compute_youden_cutoff(y_train, y_train_proba)
        fpr_test, tpr_test, _, _ = compute_youden_cutoff(y_test, y_test_proba)

        # Predictions
        y_train_pred_05, y_test_pred_05 = (y_train_proba >= 0.5).astype(int), (y_test_proba >= 0.5).astype(int)
        y_train_pred_youden, y_test_pred_youden = (y_train_proba >= cutoff).astype(int), (y_test_proba >= cutoff).astype(int)

        # Metrics :
        train_metrics_05 = compute_metrics(y_train, y_train_pred_05)
        train_metrics_youden = compute_metrics(y_train, y_train_pred_youden)
        test_metrics_05 = compute_metrics(y_test, y_test_pred_05)
        test_metrics_youden = compute_metrics(y_test, y_test_pred_youden)

        # AUC and CIs :
        auc_train, auc_cov_train = delong_roc_variance(y_train.to_numpy(), y_train_proba)
        auc_test, auc_cov_test = delong_roc_variance(y_test.to_numpy(), y_test_proba)
        ci_train = get_auc_ci(auc_train, auc_cov_train)
        ci_test = get_auc_ci(auc_test, auc_cov_test)

        # Brier
        brier_train = brier_score_loss(y_train, y_prob=y_train_proba)
        brier_test = brier_score_loss(y_test, y_prob=y_test_proba)

        # Calibration
        prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_proba, n_bins=brier_bins_train)
        prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_proba, n_bins=brier_bins_test)

        # Print summary
        print("=" * 100)
        print(f"BIVARIATE LOGISTIC REGRESSION: {self.path} + {self.control}")
        print("-" * 100)
        print(f"AUC (Train): {auc_train:.3f} (95% CI: {ci_train[0]:.3f} – {ci_train[1]:.3f})")
        print(f"AUC (Test) : {auc_test:.3f} (95% CI: {ci_test[0]:.3f} – {ci_test[1]:.3f})")
        print(f"Optimal Threshold (Youden’s J): {cutoff:.3f}")
        print("=" * 100)

        # Adding ORs + CI :
        x_train_sm = sm.add_constant(x_train_scaled)
        try:
            # Attempt MLE inference on the STANDARDIZED design
            logit_model = sm.Logit(y_train, x_train_sm).fit(disp=0)
        
            conf  = logit_model.conf_int()
            pvals = logit_model.pvalues
        
            print("\n--- Inference Using statsmodels ---")
            print("Adjusted ORs per +1 SD (Training scale)")
        
            for j, feat in enumerate(features):  # features = [self.path, self.control]
                idx = j + 1  # intercept is at index 0
                coef_sd = float(logit_model.params[idx])
                ci_low_sd, ci_high_sd = conf.iloc[idx]
        
                OR_sd    = np.exp(coef_sd)
                OR_sd_lo = np.exp(ci_low_sd)
                OR_sd_hi = np.exp(ci_high_sd)
                pval     = float(pvals[idx])
        
                print(f"Odds Ratio ({feat})     : {OR_sd:.4f}")
                print(f"95% CI for OR ({feat})  : ({OR_sd_lo:.4f}, {OR_sd_hi:.4f})")
                print(f"p-value ({feat})        : {pval:.4e}")
        
                sigma = float(scaler.scale_[j])
                print(f"Standard Deviation ({feat}) used for scaling: {sigma:.6g}")
        
        except (PerfectSeparationError, np.linalg.LinAlgError, FloatingPointError) as e:
            # Fallback: bootstrap the STANDARDIZED coefficients using sklearn, then report ORs per +1 SD
            print("\n--- Inference (bootstrap fallback due to separation) ---")
            print("Adjusted ORs per +1 SD (Training scale) estimated via bootstrap.")
            print(f"Reason: {type(e).__name__}: {e}")
        
            B = 1000  # you can lower to 200 if speed is a concern
            rng = np.random.default_rng(42)
            n  = len(y_train)
            coefs = np.empty((B, len(features)))
        
            for b in range(B):
                idxs = rng.integers(0, n, n)
                # Fit *on the standardized scale* (reuse x_train_scaled rows)
                clf_b = LogisticRegression(penalty=None).fit(x_train_scaled[idxs], y_train.iloc[idxs])
                coefs[b, :] = clf_b.coef_[0]
        
            # Point estimate from your main sklearn model (standardized scale)
            beta_hat = model.coef_[0]
        
            for j, feat in enumerate(features):
                beta_dist = coefs[:, j]
                # 95% percentile CI on the log-odds scale, then exponentiate
                lo, hi = np.percentile(beta_dist, [2.5, 97.5])
                OR_sd    = np.exp(beta_hat[j])
                OR_sd_lo = np.exp(lo)
                OR_sd_hi = np.exp(hi)
        
                # Two-sided bootstrap p: fraction of bootstrap coefs on the opposite side of zero
                p_boot = 2 * min((beta_dist >= 0).mean(), (beta_dist <= 0).mean())
        
                print(f"Odds Ratio ({feat})     : {OR_sd:.4f}")
                print(f"95% CI for OR ({feat})  : ({OR_sd_lo:.4f}, {OR_sd_hi:.4f})")
                print(f"p-value ({feat})        : {p_boot:.4e}")
        
                sigma = float(scaler.scale_[j])
                print(f"Standard Deviation ({feat}) used for scaling: {sigma:.6g}")


        print_metrics(f"[{self.path} + {self.control}] Train @ 0.5", train_metrics_05)
        print_metrics(f"[{self.path} + {self.control}] Train @ Youden", train_metrics_youden)
        print_metrics(f"[{self.path} + {self.control}] Test @ 0.5", test_metrics_05)
        print_metrics(f"[{self.path} + {self.control}] Test @ Youden", test_metrics_youden)

        # Confusion matrices :
        fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 8))
        plot_confusion_matrix(axes_cm[0, 0], train_metrics_05["cm"], f"[{self.path} + {self.control}] Train @ 0.5")
        plot_confusion_matrix(axes_cm[0, 1], train_metrics_youden["cm"], f"[{self.path} + {self.control}] Train @ Youden")
        plot_confusion_matrix(axes_cm[1, 0], test_metrics_05["cm"], f"[{self.path} + {self.control}] Test @ 0.5")
        plot_confusion_matrix(axes_cm[1, 1], test_metrics_youden["cm"], f"[{self.path} + {self.control}] Test @ Youden")
        plt.tight_layout()
        plt.show()

        # ROC curves
        fig_roc, axes_roc = plt.subplots(1, 2, figsize=(14, 6))
        plot_roc(ax=axes_roc[0], fpr=fpr_train, tpr=tpr_train, phase="Train",
                 title=f"ROC Curve (Train) [{self.path} + {self.control}]",
                 auc=auc_train, ci=ci_train, cutoff=cutoff, add_youden=True, best_idx=best_idx)

        plot_roc(ax=axes_roc[1], fpr=fpr_test, tpr=tpr_test, phase="Test",
                 title=f"ROC Curve (Test) [{self.path} + {self.control}]",
                 auc=auc_test, ci=ci_test, cutoff=cutoff)
        plt.tight_layout()
        plt.show()

        # Calibration curves :
        fig_cal, axes_cal = plt.subplots(1, 2, figsize=(14, 6))
        plot_calibration_curve(ax=axes_cal[0], prob_pred=prob_pred_train, prob_true=prob_true_train,
                               title=f"Calibration Curve (Train) [{self.path} + {self.control}]",
                               brier_score=brier_train)

        plot_calibration_curve(ax=axes_cal[1], prob_pred=prob_pred_test, prob_true=prob_true_test,
                               title=f"Calibration Curve (Test) [{self.path} + {self.control}]",
                               brier_score=brier_test)
        plt.tight_layout()
        plt.show()

        # === Decision Surfaces with Threshold Boundaries ===
        fig_dec, axes_dec = plt.subplots(1, 2, figsize=(14, 6))

        plot_logistic_decision_surface(ax=axes_dec[0], model=model, X_scaled=x_train_scaled, y=y_train,
                                       feature_names=[self.path, self.control], threshold=0.5, cutoff=cutoff,
                                       title=f"Decision Surface (Train) [{self.path} + {self.control}]")

        plot_logistic_decision_surface(ax=axes_dec[1], model=model, X_scaled=x_test_scaled, y=y_test,
                                       feature_names=[self.path, self.control], threshold=0.5, cutoff=cutoff,
                                       title=f"Decision Surface (Test) [{self.path} + {self.control}]")

        plt.tight_layout()
        plt.show()

        # === Predicted Probability Distributions (Train/Test) ===
        fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 6))

        plot_proba_distribution(ax=axes_dist[0], y_proba=y_train_proba, y_true=y_train, threshold=cutoff,
                                title=f"Predicted Probability Distribution (Train) [{self.path} + {self.control}]",
                                target_bin_count=target_bin_count)

        plot_proba_distribution(ax=axes_dist[1], y_proba=y_test_proba, y_true=y_test, threshold=cutoff,
                                title=f"Predicted Probability Distribution (Test) [{self.path} + {self.control}]",
                                target_bin_count=target_bin_count)

        plt.tight_layout()
        plt.show()

    def evaluate_svm_bivariate_model(self, target="Diagnosis", kernel="linear"):

        features = [self.path, self.control]

        df_train, df_test = self.data[[*features, target]].dropna(), self.test_data[[*features, target]].dropna()
        x_train, y_train, x_test, y_test = df_train[features], df_train[target], df_test[features], df_test[target]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = SVC(kernel=kernel, probability=True)
        model.fit(x_train_scaled, y_train)
        f_train, f_test = model.decision_function(x_train_scaled), model.decision_function(x_test_scaled)
        p_train, p_test = model.predict_proba(x_train_scaled)[:, 1], model.predict_proba(x_test_scaled)[:, 1]

        # ROC + Youden :
        fpr_f_train, tpr_f_train, best_idx_f_train, cutoff_f = compute_youden_cutoff(y_train, f_train)
        fpr_f_test, tpr_f_test, _, _ = compute_youden_cutoff(y_test, f_test)
        fpr_p_train, tpr_p_train, best_idx_p_train, cutoff_p = compute_youden_cutoff(y_train, p_train)
        fpr_p_test, tpr_p_test, _, _ = compute_youden_cutoff(y_test, p_test)

        # Predictions :
        y_train_pred_f_05, y_test_pred_f_05 = (f_train >= 0).astype(int), (f_test >= 0).astype(int)
        y_train_pred_f_youden, y_test_pred_f_youden = (f_train >= cutoff_f).astype(int), (f_test >= cutoff_f).astype(int)
        y_train_pred_p_05, y_test_pred_p_05 = (p_train >= 0.5).astype(int), (p_test >= 0.5).astype(int)
        y_train_pred_p_youden, y_test_pred_p_youden = (p_train >= cutoff_p).astype(int), (p_test >= cutoff_p).astype(int)

        # Metrics :
        train_metrics_f_05 = compute_metrics(y_train, y_train_pred_f_05)
        train_metrics_f_youden = compute_metrics(y_train, y_train_pred_f_youden)
        test_metrics_f_05 = compute_metrics(y_test, y_test_pred_f_05)
        test_metrics_f_youden = compute_metrics(y_test, y_test_pred_f_youden)
        train_metrics_p_05 = compute_metrics(y_train, y_train_pred_p_05)
        train_metrics_p_youden = compute_metrics(y_train, y_train_pred_p_youden)
        test_metrics_p_05 = compute_metrics(y_test, y_test_pred_p_05)
        test_metrics_p_youden = compute_metrics(y_test, y_test_pred_p_youden)

        # AUC and CIs :
        auc_f_train, auc_cov_f_train = delong_roc_variance(y_train.to_numpy(), f_train)
        auc_f_test, auc_cov_f_test = delong_roc_variance(y_test.to_numpy(), f_test)
        auc_p_train, auc_cov_p_train = delong_roc_variance(y_train.to_numpy(), p_train)
        auc_p_test, auc_cov_p_test = delong_roc_variance(y_test.to_numpy(), p_test)
        ci_f_train = get_auc_ci(auc_f_train, auc_cov_f_train)
        ci_f_test = get_auc_ci(auc_f_test, auc_cov_f_test)
        ci_p_train = get_auc_ci(auc_p_train, auc_cov_p_train)
        ci_p_test = get_auc_ci(auc_p_test, auc_cov_p_test)

        print("=" * 100)
        print(f"SVM ({kernel.upper()}) BIVARIATE: {self.path} + {self.control}")
        print("-" * 100)
        print(f"[Margin] Train AUC : {auc_f_train:.3f} (95% CI: {ci_f_train[0]:.3f} – {ci_f_train[1]:.3f})")
        print(f"[Margin] Test  AUC : {auc_f_test:.3f} (95% CI: {ci_f_test[0]:.3f} – {ci_f_test[1]:.3f})")
        print(f"Optimal Margin Cutoff (Youden): {cutoff_f:.3f}")
        print("-" * 100)
        print(f"[Proba]  Train AUC : {auc_p_train:.3f} (95% CI: {ci_p_train[0]:.3f} – {ci_p_train[1]:.3f})")
        print(f"[Proba]  Test  AUC : {auc_p_test:.3f} (95% CI: {ci_p_test[0]:.3f} – {ci_p_test[1]:.3f})")
        print(f"Optimal Proba  Cutoff (Youden): {cutoff_p:.3f}")
        print("=" * 100)

        print_metrics(f"[{self.path} + {self.control}] Train @ f(x) ≥ 0", train_metrics_f_05)
        print_metrics(f"[{self.path} + {self.control}] Train @ f(x) ≥ Youden", train_metrics_f_youden)
        print_metrics(f"[{self.path} + {self.control}] Train @ P(x) ≥ 0.5", train_metrics_p_05)
        print_metrics(f"[{self.path} + {self.control}] Train @ P(x) ≥ Youden", train_metrics_p_youden)
        print_metrics(f"[{self.path} + {self.control}] Test  @ f(x) ≥ 0", test_metrics_f_05)
        print_metrics(f"[{self.path} + {self.control}] Test  @ f(x) ≥ Youden", test_metrics_f_youden)
        print_metrics(f"[{self.path} + {self.control}] Test  @ P(x) ≥ 0.5", test_metrics_p_05)
        print_metrics(f"[{self.path} + {self.control}] Test  @ P(x) ≥ Youden", test_metrics_p_youden)

        fig_cm, axes_cm = plt.subplots(4, 2, figsize=(14, 16))

        plot_confusion_matrix(ax=axes_cm[0, 0], cm=train_metrics_f_05["cm"], title=f"[{self.path} + {self.control}] Train @ f(x) ≥ 0")
        plot_confusion_matrix(ax=axes_cm[0, 1], cm=test_metrics_f_05["cm"], title=f"[{self.path} + {self.control}] Test  @ f(x) ≥ 0")
        plot_confusion_matrix(ax=axes_cm[1, 0], cm=train_metrics_f_youden["cm"], title=f"[{self.path} + {self.control}] Train @ f(x) ≥ Youden")
        plot_confusion_matrix(ax=axes_cm[1, 1], cm=test_metrics_f_youden["cm"], title=f"[{self.path} + {self.control}] Test  @ f(x) ≥ Youden")
        plot_confusion_matrix(ax=axes_cm[2, 0], cm=train_metrics_p_05["cm"], title=f"[{self.path} + {self.control}] Train @ P(x) ≥ 0.5")
        plot_confusion_matrix(ax=axes_cm[2, 1], cm=test_metrics_p_05["cm"], title=f"[{self.path} + {self.control}] Test  @ P(x) ≥ 0.5")
        plot_confusion_matrix(ax=axes_cm[3, 0], cm=train_metrics_p_youden["cm"], title=f"[{self.path} + {self.control}] Train @ P(x) ≥ Youden")
        plot_confusion_matrix(ax=axes_cm[3, 1], cm=test_metrics_p_youden["cm"], title=f"[{self.path} + {self.control}] Test  @ P(x) ≥ Youden")

        plt.tight_layout()
        plt.show()

        # === ROC Curves ===
        fig_roc, axes_roc = plt.subplots(2, 2, figsize=(14, 12))

        # Margin-based ROC curves
        plot_roc(ax=axes_roc[0, 0], fpr=fpr_f_train, tpr=tpr_f_train, phase="Train",
                 title=f"ROC Curve (Train, Margin) [{self.path} + {self.control}]",
                 auc=auc_f_train, ci=ci_f_train, cutoff=cutoff_f,
                 add_youden=True, best_idx=best_idx_f_train)
        plot_roc(ax=axes_roc[0, 1], fpr=fpr_f_test, tpr=tpr_f_test, phase="Test",
                 title=f"ROC Curve (Test, Margin) [{self.path} + {self.control}]",
                 auc=auc_f_test, ci=ci_f_test, cutoff=cutoff_f)

        # Proba-based ROC curves
        plot_roc(ax=axes_roc[1, 0], fpr=fpr_p_train, tpr=tpr_p_train, phase="Train",
                 title=f"ROC Curve (Train, Proba) [{self.path} + {self.control}]",
                 auc=auc_p_train, ci=ci_p_train, cutoff=cutoff_p,
                 add_youden=True, best_idx=best_idx_p_train)

        plot_roc(ax=axes_roc[1, 1], fpr=fpr_p_test, tpr=tpr_p_test, phase="Test",
                 title=f"ROC Curve (Test, Proba) [{self.path} + {self.control}]",
                 auc=auc_p_test, ci=ci_p_test, cutoff=cutoff_p)

        plt.tight_layout()
        plt.show()

        # Plot calibration curves
        fig_cal, axes_cal = plt.subplots(1, 2, figsize=(14, 6))
        plot_svm_calibration_curve(ax=axes_cal[0], y_true=y_train, proba=p_train, margin=f_train,
                                   title=f"Calibration Curve (Train) [{self.path} + {self.control}]")
        plot_svm_calibration_curve(ax=axes_cal[1], y_true=y_test, proba=p_test, margin=f_test,
                                   title=f"Calibration Curve (Test) [{self.path} + {self.control}]")
        plt.tight_layout()
        plt.show()

        # === Decision Surfaces (Separated by Geometric / Probabilistic) ===
        fig_dec, axes_dec = plt.subplots(2, 2, figsize=(14, 12))
        plot_svm_decision_surfaces(ax_geom=axes_dec[0, 0], ax_proba=axes_dec[0, 1],
                                   model=model, X_scaled=x_train_scaled, y=y_train,
                                   feature_names=[self.path, self.control],
                                   youden_f=cutoff_f, youden_p=cutoff_p,
                                   title_prefix="Train")

        plot_svm_decision_surfaces(ax_geom=axes_dec[1, 0], ax_proba=axes_dec[1, 1],
                                   model=model, X_scaled=x_test_scaled, y=y_test,
                                   feature_names=[self.path, self.control],
                                   youden_f=cutoff_f, youden_p=cutoff_p,
                                   title_prefix="Test")

        plt.tight_layout()
        plt.show()

    def evaluate_tree_trivariate_model(self, target="Diagnosis", max_depth=3, criterion="gini"):

        features = [self.path, self.control, self.ratio]

        df_train, df_test = self.data[[*features, target]].dropna(), self.test_data[[*features, target]].dropna()
        x_train, y_train, x_test, y_test = df_train[features], df_train[target], df_test[features], df_test[target]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
        model.fit(x_train_scaled, y_train)
        y_train_proba, y_test_proba = model.predict_proba(x_train_scaled)[:, 1], model.predict_proba(x_test_scaled)[:, 1]

        # ROC + Youden
        fpr_train, tpr_train, _, _ = compute_youden_cutoff(y_train, y_train_proba)
        fpr_test, tpr_test, _, _ = compute_youden_cutoff(y_test, y_test_proba)

        # Predictions
        y_pred_train, y_pred_test = model.predict(x_train_scaled), model.predict(x_test_scaled)

        # Metrics
        train_metrics = compute_metrics(y_train, y_pred_train)
        test_metrics = compute_metrics(y_test, y_pred_test)

        # AUC and CIs
        auc_train, auc_cov_train = delong_roc_variance(y_train.to_numpy(), y_train_proba)
        auc_test, auc_cov_test = delong_roc_variance(y_test.to_numpy(), y_test_proba)
        ci_train = get_auc_ci(auc_train, auc_cov_train)
        ci_test = get_auc_ci(auc_test, auc_cov_test)

        # Brier
        brier_train = brier_score_loss(y_train, y_prob=y_train_proba)
        brier_test = brier_score_loss(y_test, y_prob=y_test_proba)

        # Calibration
        prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_proba, n_bins=10)
        prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_proba, n_bins=10)

        # Print summary
        print("=" * 100)
        print(f"DECISION TREE (max_depth={max_depth}, criterion={criterion.upper()}): {self.path} + {self.control} + {self.ratio}")
        print("-" * 100)
        print(f"AUC (Train): {auc_train:.3f} (95% CI: {ci_train[0]:.3f} – {ci_train[1]:.3f})")
        print(f"AUC (Test) : {auc_test:.3f} (95% CI: {ci_test[0]:.3f} – {ci_test[1]:.3f})")
        print("-" * 100)

        print_metrics(f"[{self.path} + {self.control} + {self.ratio}] Train", train_metrics)
        print_metrics(f"[{self.path} + {self.control} + {self.ratio}] Test", test_metrics)

        # Confusions matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_confusion_matrix(ax=axes[0], cm=train_metrics["cm"], title=f"[{self.path} + {self.control} + {self.ratio}] Train")
        plot_confusion_matrix(ax=axes[1], cm=test_metrics["cm"], title=f"[{self.path} + {self.control} + {self.ratio}] Test")
        plt.tight_layout()
        plt.show()

        # ROC curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_roc(ax=axes[0], fpr=fpr_train, tpr=tpr_train, phase="Train",
                 title=f"ROC Curve (Train) [{self.path} + {self.control} + {self.ratio}]",
                 auc=auc_train, ci=ci_train)

        plot_roc(ax=axes[1], fpr=fpr_test, tpr=tpr_test, phase="Test",
                 title=f"ROC Curve (Test) [{self.path} + {self.control} + {self.ratio}]",
                 auc=auc_test, ci=ci_test)

        plt.tight_layout()
        plt.show()

        # Calibration curves
        fig_cal, axes_cal = plt.subplots(1, 2, figsize=(14, 6))
        plot_calibration_curve(ax=axes_cal[0], prob_pred=prob_pred_train, prob_true=prob_true_train,
                               title=f"Calibration Curve (Train) [{self.path} + {self.control} + {self.ratio}]",
                               brier_score=brier_train)
        plot_calibration_curve(ax=axes_cal[1], prob_pred=prob_pred_test, prob_true=prob_true_test,
                               title=f"Calibration Curve (Test) [{self.path} + {self.control} + {self.ratio}]",
                               brier_score=brier_test)

        plt.tight_layout()
        plt.show()

        # Decision surfaces for each 2D projection
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pairs = list(combinations(range(3), 2))

        for ax, (i, j) in zip(axes, pairs):
            xi = x_train_scaled[:, i]
            xj = x_train_scaled[:, j]

            # Create a meshgrid
            xi_range = np.linspace(xi.min() - 0.5, xi.max() + 0.5, 200)
            xj_range = np.linspace(xj.min() - 0.5, xj.max() + 0.5, 200)
            xx, yy = np.meshgrid(xi_range, xj_range)

            # Fill in all 3 dimensions
            grid = np.zeros((xx.size, 3))
            grid[:, i] = xx.ravel()
            grid[:, j] = yy.ravel()
            preds = model.predict(grid).reshape(xx.shape)

            ax.contourf(xx, yy, preds, cmap="RdBu", alpha=0.6)
            ax.scatter(xi, xj, c=y_train, cmap="RdBu_r", edgecolor="k", s=30)
            ax.set_xlabel(features[i])
            ax.set_ylabel(features[j])
            ax.set_title(f"Decision Surface: {features[i]} vs {features[j]}")

        plt.tight_layout()
        plt.show()

        # Plot the decision tree
        plt.figure(figsize=(14, 8))
        plot_tree(model, feature_names=features, class_names=["0", "1"],
                  filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree Structure")
        plt.tight_layout()
        plt.show()

        # Print textual tree rules
        print("\n" + "=" * 80)
        print("Tree Rules :")
        print("-" * 80)
        print(export_text(model, feature_names=features))
        print("=" * 80)
