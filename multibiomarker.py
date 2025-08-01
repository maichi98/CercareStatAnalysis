from sklearn.metrics import brier_score_loss
import umap
import shap
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from delong import *
from plots import *
from metrics import *
from utils import *
from biomarker import Biomarker
import constants


class MultiBiomarker(Biomarker):

    def __init__(self, biomarkers, data, test_data=None):

        params = [constants.DICT_MARKERS[biomarker]["ratio"] for biomarker in biomarkers]
        super().__init__(params, data, test_data)

    def plot_tsne(self, target="Diagnosis", perplexity=30):

        #  prepare data
        df_train = self.data[self.params + [target]].dropna()
        df_test = self.test_data[self.params + [target]].dropna()

        configs = {
            "Train Only": {
                "x_data": df_train[self.params].values,
                "y_labels": df_train[target],
                "style": None
            },
            "Test Only": {
                "x_data": df_test[self.params].values,
                "y_labels": df_test[target],
                "style": None
            },
            "Train + Test": {
                "x_data": pd.concat([df_train, df_test])[self.params].values,
                "y_labels": pd.concat([df_train[target], df_test[target]]),
                "style": ["Train"] * len(df_train) + ["Test"] * len(df_test)
            }
        }

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for ax, (title, cfg) in zip(axes, configs.items()):
            x_embedded = fit_tsne(x_data=cfg["x_data"], perplexity=perplexity)
            plot_embedding_scatter(
                x_embedded=x_embedded,
                y_labels=cfg["y_labels"],
                title=f"t-SNE — {title}",
                ax=ax,
                style=cfg["style"]
            )

        plt.tight_layout()
        plt.show()

    def plot_umap(self, target="Diagnosis", n_neighbors=15, min_dist=0.1, random_state=None):

        df_train = self.data[self.params + [target]].dropna()
        df_test = self.test_data[self.params + [target]].dropna()

        x_train, y_train = df_train[self.params].values, df_train[target]
        x_test, y_test = df_test[self.params].values, df_test[target]

        # Fit umap on training data, then transform both train and test data
        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state)

        x_train_embedded = umap_model.fit_transform(x_train)
        x_test_embedded = umap_model.transform(x_test)

        configs = {
            "Train Only": {
                "x_data": x_train_embedded,
                "y_labels": y_train,
                "style": None
            },
            "Test Only": {
                "x_data": x_test_embedded,
                "y_labels": y_test,
                "style": None
            },
            "Train + Test": {
                "x_data": np.vstack([x_train_embedded, x_test_embedded]),
                "y_labels": pd.concat([y_train, y_test]),
                "style": ["Train"] * len(df_train) + ["Test"] * len(df_test)
            }
        }

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for ax, (title, cfg) in zip(axes, configs.items()):
            plot_embedding_scatter(
                x_embedded=cfg["x_data"],
                y_labels=cfg["y_labels"],
                title=f"UMAP — {title}",
                ax=ax,
                style=cfg["style"]
            )

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, method="pearson"):

        df_train = self.data[self.params].dropna()
        df_test = self.test_data[self.params].dropna()
        df_combined = pd.concat([df_train, df_test], ignore_index=True)

        datasets = {
            "Train": df_train,
            "Test": df_test,
            "Train + Test": df_combined
        }

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        for ax, (title, df) in zip(axes, datasets.items()):

            corr = df.corr(method=method)
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                square=True,
                cbar=False,
                ax=ax
            )
            ax.set_title(f"Correlation Heatmap — {title}")

        plt.suptitle(f"Feature {method} Correlation Matrices", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()

    def plot_feature_pairplot(self,
                              target="Diagnosis",
                              s=40,
                              alpha=0.7,
                              lower=0.00,
                              upper=1.00):

        # Label map
        label_map = {k: v["label"] for k, v in constants.DIAGNOSIS_INFO.items()}

        # Apply winsorization with provided quantile bounds
        df_train = winsorize_df_to_nan(self.data[self.params + [target]], lower=lower, upper=upper).dropna()
        df_test = winsorize_df_to_nan(self.test_data[self.params + [target]], lower=lower, upper=upper).dropna()

        # Annotate sets and map diagnosis labels
        df_train["Set"], df_test["Set"] = "Train", "Test"
        df_train[target] = df_train[target].map(label_map)
        df_test[target] = df_test[target].map(label_map)

        # Combine datasets
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        datasets = {
            "Train": {"data": df_train, "style": False},
            "Test": {"data": df_test, "style": False},
            "Train + Test": {"data": df_combined, "style": True}
        }

        for name, cfg in datasets.items():
            df = cfg["data"].dropna(subset=self.params + [target])
            use_style = cfg["style"]

            g = sns.PairGrid(df, vars=self.params, hue=target, corner=True)

            if use_style:
                g.map_lower(sns.scatterplot, style=df["Set"], edgecolor="k", s=s, alpha=alpha)
            else:
                g.map_lower(sns.scatterplot, edgecolor="k", s=s, alpha=alpha)

            g.map_diag(sns.kdeplot, fill=True)
            g.add_legend(title="Diagnosis")

            plt.suptitle(f"Pairwise Feature Relationships — {name}", y=1.02, fontsize=16)
            plt.tight_layout()
            plt.show()

    def evaluate_logistic_regression_model(self, target="Diagnosis"):
        features = self.params

        df_train = self.data[features + [target]].dropna()
        df_test = self.test_data[features + [target]].dropna()

        x_train, y_train = df_train[features], df_train[target]
        x_test, y_test = df_test[features], df_test[target]

        # === Scale
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # === Train logistic regression
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train_scaled, y_train)

        # === Predict probabilities and labels
        y_train_proba = model.predict_proba(x_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(x_test_scaled)[:, 1]
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        y_test_pred = (y_test_proba >= 0.5).astype(int)

        # === Metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        auc_train = roc_auc_score(y_train, y_train_proba)
        auc_test = roc_auc_score(y_test, y_test_proba)

        print("=" * 100)
        print("MULTIVARIATE LOGISTIC REGRESSION")
        print("-" * 100)
        print(f"AUC (Train): {auc_train:.3f}")
        print(f"AUC (Test) : {auc_test:.3f}")
        print("-" * 100)

        print_metrics("Logistic Regression Train", train_metrics)
        print_metrics("Logistic Regression Test", test_metrics)

        # === Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_confusion_matrix(axes[0], train_metrics["cm"], title="Confusion Matrix — Train")
        plot_confusion_matrix(axes[1], test_metrics["cm"], title="Confusion Matrix — Test")
        plt.tight_layout()
        plt.show()

        # === ROC Curves
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_roc(axes[0], "Train", fpr_train, tpr_train, auc_train, (0, 0), title="ROC Curve — Train")
        plot_roc(axes[1], "Test", fpr_test, tpr_test, auc_test, (0, 0), title="ROC Curve — Test")
        plt.tight_layout()
        plt.show()

        # === Coefficients
        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_[0]
        }).sort_values(by="Coefficient", key=abs, ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=coef_df, x="Coefficient", y="Feature")
        plt.title("Logistic Regression Coefficients")
        plt.tight_layout()
        plt.show()

    def evaluate_xgboost_model(self,
                               target="Diagnosis",
                               max_depth=3,
                               n_estimators=100,
                               learning_rate=0.1,
                               eval_metric="logloss",
                               use_label_encoder=True):

        features = self.params

        df_train = self.data[features + [target]].dropna()
        df_test = self.test_data[features + [target]].dropna()
        x_train, y_train = df_train[features], df_train[target]
        x_test, y_test = df_test[features], df_test[target]

        # === Scale
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # === Train model
        model = XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric=eval_metric,
            use_label_encoder=use_label_encoder,
            random_state=42
        )
        model.fit(x_train_scaled, y_train)

        y_train_proba = model.predict_proba(x_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(x_test_scaled)[:, 1]

        # === ROC + Youden
        fpr_train, tpr_train, best_idx_train, cutoff = compute_youden_cutoff(y_train, y_train_proba)
        fpr_test, tpr_test, best_idx_test, _ = compute_youden_cutoff(y_test, y_test_proba)

        # === Predict labels at optimal cutoff
        y_train_pred = (y_train_proba >= cutoff).astype(int)
        y_test_pred = (y_test_proba >= cutoff).astype(int)

        # === Metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        auc_train, auc_cov_train = delong_roc_variance(y_train.to_numpy(), y_train_proba)
        auc_test, auc_cov_test = delong_roc_variance(y_test.to_numpy(), y_test_proba)
        ci_train = get_auc_ci(auc_train, auc_cov_train)
        ci_test = get_auc_ci(auc_test, auc_cov_test)

        brier_train = brier_score_loss(y_train, y_train_proba)
        brier_test = brier_score_loss(y_test, y_test_proba)

        prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_proba, n_bins=10)
        prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_proba, n_bins=10)

        # === Summary
        print("=" * 100)
        print(f"XGBOOST (depth={max_depth}, lr={learning_rate}, n_estimators={n_estimators})")
        print("-" * 100)
        print(f"AUC (Train): {auc_train:.3f} (95% CI: {ci_train[0]:.3f} – {ci_train[1]:.3f})")
        print(f"AUC (Test) : {auc_test:.3f} (95% CI: {ci_test[0]:.3f} – {ci_test[1]:.3f})")
        print("-" * 100)

        print_metrics("XGBoost Train", train_metrics)
        print_metrics("XGBoost Test", test_metrics)

        # === Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_confusion_matrix(axes[0], train_metrics["cm"], title="Confusion Matrix — Train")
        plot_confusion_matrix(axes[1], test_metrics["cm"], title="Confusion Matrix — Test")
        plt.tight_layout()
        plt.show()

        # === ROC Curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_roc(axes[0], "Train", fpr_train, tpr_train, auc_train, ci_train, title="ROC Curve — Train",
                 best_idx=best_idx_train, cutoff=cutoff, add_youden=True)
        plot_roc(axes[1], "Test", fpr_test, tpr_test, auc_test, ci_test, title="ROC Curve — Test",
                 best_idx=best_idx_test, cutoff=cutoff, add_youden=True)
        plt.tight_layout()
        plt.show()

        # === Calibration Curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_calibration_curve(axes[0], prob_pred_train, prob_true_train,
                               title="Calibration Curve — Train", brier_score=brier_train)
        plot_calibration_curve(axes[1], prob_pred_test, prob_true_test,
                               title="Calibration Curve — Test", brier_score=brier_test)
        plt.tight_layout()
        plt.show()

        # === Probability Distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_proba_distribution(axes[0], y_train_proba, y_train, threshold=cutoff,
                                title="Probability Distribution — Train")
        plot_proba_distribution(axes[1], y_test_proba, y_test, threshold=cutoff,
                                title="Probability Distribution — Test")
        plt.tight_layout()
        plt.show()

        # === SHAP Summary Plot
        explainer = shap.Explainer(model, x_train_scaled, feature_names=features)
        shap_values = explainer(x_train_scaled)

        shap.summary_plot(shap_values, features=x_train_scaled, feature_names=features)

    def evaluate_random_forest_model(self,
                                     target="Diagnosis",
                                     n_estimators=100,
                                     max_depth=3,
                                     max_features="sqrt",
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     bootstrap=True,
                                     random_state=42):

        features = self.params

        df_train = self.data[features + [target]].dropna()
        df_test = self.test_data[features + [target]].dropna()
        x_train, y_train = df_train[features], df_train[target]
        x_test, y_test = df_test[features], df_test[target]

        # === Scale
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # === Train Random Forest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=random_state
        )
        model.fit(x_train_scaled, y_train)

        y_train_proba = model.predict_proba(x_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(x_test_scaled)[:, 1]

        # === Predict labels at fixed threshold = 0.5
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        y_test_pred = (y_test_proba >= 0.5).astype(int)
        threshold = 0.5

        # === Metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        auc_train, auc_cov_train = delong_roc_variance(y_train.to_numpy(), y_train_proba)
        auc_test, auc_cov_test = delong_roc_variance(y_test.to_numpy(), y_test_proba)
        ci_train = get_auc_ci(auc_train, auc_cov_train)
        ci_test = get_auc_ci(auc_test, auc_cov_test)

        # === Summary
        print("=" * 100)
        print(f"RANDOM FOREST (n={n_estimators}, max_depth={max_depth}, max_features={max_features})")
        print("-" * 100)
        print(f"AUC (Train): {auc_train:.3f} (95% CI: {ci_train[0]:.3f} – {ci_train[1]:.3f})")
        print(f"AUC (Test) : {auc_test:.3f} (95% CI: {ci_test[0]:.3f} – {ci_test[1]:.3f})")
        print("-" * 100)

        print_metrics("Random Forest Train", train_metrics)
        print_metrics("Random Forest Test", test_metrics)

        # === Plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_confusion_matrix(axes[0], train_metrics["cm"], title="Confusion Matrix — Train")
        plot_confusion_matrix(axes[1], test_metrics["cm"], title="Confusion Matrix — Test")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_roc(axes[0], "Train", *roc_curve(y_train, y_train_proba)[:2], auc_train, ci_train,
                 title="ROC Curve — Train")
        plot_roc(axes[1], "Test", *roc_curve(y_test, y_test_proba)[:2], auc_test, ci_test, title="ROC Curve — Test")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_proba_distribution(axes[0], y_train_proba, y_train, threshold=threshold,
                                title="Probability Distribution — Train")
        plot_proba_distribution(axes[1], y_test_proba, y_test, threshold=threshold,
                                title="Probability Distribution — Test")
        plt.tight_layout()
        plt.show()

        # === Feature Importances
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[sorted_idx], y=np.array(features)[sorted_idx])
        plt.title("Random Forest Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def grid_search_random_forest(self, param_grid, target="Diagnosis"):

        features = self.params

        df_train = self.data[features + [target]].dropna()
        df_test = self.test_data[features + [target]].dropna()
        x_train, y_train = df_train[features], df_train[target]
        x_test, y_test = df_test[features], df_test[target]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        keys, values = zip(*param_grid.items())
        best_auc = -1
        best_config = None
        results = []

        combos = list(product(*values))

        for v in combos:
            params = dict(zip(keys, v))
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(x_train_scaled, y_train)

            try:
                y_test_proba = model.predict_proba(x_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_test_proba)
            except Exception as e:
                print(f"Error calculating AUC for params {params}: {e}")
                auc = 0.0

            results.append((params, auc))

            if auc > best_auc:
                best_auc = auc
                best_config = params

        return best_config, best_auc, results


if __name__ == "__main__":
    import json
    import pandas as pd
    from itertools import combinations
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    from constants import DICT_RENAMING_MAPPING
    from utils import format_dataframe, add_ratio_columns

    def evaluate_subset(args):
        subset, df_data, df_test, param_grid = args
        try:
            mb = MultiBiomarker(biomarkers=subset, data=df_data, test_data=df_test)
            best_config, best_auc, _ = mb.grid_search_random_forest(param_grid)
            return {
                "subset": subset,
                "auc": best_auc,
                "config": best_config
            }
        except Exception:
            return {
                "subset": subset,
                "auc": 0.0,
                "config": None
            }

    # === Load and preprocess data
    df_data = pd.read_excel("data/cercare_training_data.xlsx").rename(columns=DICT_RENAMING_MAPPING)
    df_test = pd.read_excel("data/cercare_test_data.xlsx").rename(columns=DICT_RENAMING_MAPPING)

    df_data = format_dataframe(df=df_data)
    df_test = format_dataframe(df=df_test)

    add_ratio_columns(df_data)
    add_ratio_columns(df_test)

    print("Data loaded and formatted successfully.")

    # === Create biomarker subsets
    BIOMARKERS = ["CBV_corr", "CBV_noncorr", "DELAY", "CTH", "CTH MAX", "OEF", "rLEAKAGE", "rCMRO2", "COV"]
    subsets = [combo for r in range(2, len(BIOMARKERS) + 1) for combo in combinations(BIOMARKERS, r)]

    param_grid = {
        "n_estimators": [1, 2, 3, 4, 5, 10, 20, 50, 100],
        "max_depth": [1, 2, 3, 4, 5, None],
        "bootstrap": [True, False],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 6]
    }

    args_list = [
        (subset, df_data, df_test, {
            **param_grid,
            "max_features": list(range(1, len(subset) + 1)) + [None]
        })
        for subset in subsets
    ]

    print(f"Running multiprocessing grid search on {len(args_list)} biomarker subsets using {cpu_count()} cores...")

    results = []
    with Pool(processes=cpu_count()) as pool:
        progress = tqdm(total=len(args_list), desc="Evaluating subsets", dynamic_ncols=True)
        for result in pool.imap(evaluate_subset, args_list):
            if result["subset"]:
                subset_label = ", ".join(result["subset"])
                progress.set_postfix_str(f"{subset_label}")
            results.append(result)
            progress.update(1)
        progress.close()

    # === Filter and save results
    valid_results = [r for r in results if r["config"] is not None]
    best_result = max(valid_results, key=lambda x: x["auc"]) if valid_results else None

    df_results = pd.DataFrame([
        {
            "subset": ", ".join(r["subset"]),
            "auc": r["auc"],
            **r["config"]
        }
        for r in valid_results
    ])
    df_results.to_csv("biomarker_subset_results.csv", index=False)

    if best_result:
        with open("best_biomarker_config.json", "w") as f:
            json.dump(best_result, f, indent=2)

        print("\nBest subset:")
        print(", ".join(best_result["subset"]))
        print(f"Best AUC: {best_result['auc']:.3f}")
        print("Best configuration:")
        print(json.dumps(best_result["config"], indent=2))
    else:
        print("No valid model configurations found.")
