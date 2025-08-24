from sklearn.metrics import roc_curve, confusion_matrix
from scipy import stats
import numpy as np
import warnings


def compute_youden_cutoff(y_true, y_pred):

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    best_idx = np.argmax(tpr - fpr)
    cutoff = thresholds[best_idx]

    return fpr, tpr, best_idx, cutoff


def compute_metrics(y_true, y_pred):
    """
    Implementation without using sklearn's functions aside from confusion_matrix,
    the metrics are computed manually to raise exceptions in the case of zero division ...

    """

    #     acc = accuracy_score(y_true, y_pred) * 100
    #     recall = recall_score(y_true, y_pred, zero_division=0) * 100
    #     precision = precision_score(y_true, y_pred, zero_division=0) * 100
    #     f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    #     specificity = (
    #         confusion_matrix(y_true, y_pred)[0, 0] /
    #         max(1, confusion_matrix(y_true, y_pred)[0].sum())
    #     ) * 100
    #
    #     cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2 !")

    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    if total == 0:
        raise ValueError("Confusion matrix is empty !")

    # Accuracy : (TP + TN) / Total
    acc = (tp + tn) * 100 / total

    # Recall (Sensitivity) : TP / (TP + FN)
    if (tp + fn) == 0:
        raise ValueError("No Positive samples in confusion matrix !")
    recall = tp * 100 / (tp + fn)

    # Precision : TP / (TP + FP)  (no positive predictions -> set to 0.0)
    if (tp + fp) == 0:
        warnings.warn("No positive predictions in y_pred; precision is ill-defined. Setting precision=0.0.", RuntimeWarning)
        precision = 0.0
    else:
        precision = tp * 100.0 / (tp + fp)

    # F1 Score : 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) == 0.0:
        # Both are zero → set F1 to 0.0 (consistent with sklearn's zero_division=0 behavior)
        f1_score = 0.0
    else:
        f1_score = (2.0 * precision * recall) / (precision + recall)

    # Specificity : TN / (TN + FP)
    # If there are no negative samples in y_true (tn+fp==0), set specificity=0.0 and warn.
    if (tn + fp) == 0:
        warnings.warn("No negative samples in y_true; specificity is ill-defined. Setting specificity=0.0.", RuntimeWarning)
        specificity = 0.0
    else:
        specificity = tn * 100.0 / (tn + fp)

    return {
        "acc": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1_score,
        "specificity": specificity,
        "cm": cm,
    }


def get_auc_ci(auc, auc_cov):

    if auc_cov == 0 or np.isnan(auc_cov):
        print(f"Covariance of AUC is {auc_cov}; defaulting to a small delta for CI computation.")
        delta = 1e-5
        return max(0, auc - delta), min(1, auc + delta)
        # raise ValueError(f"AUC Covariance is {auc_cov}, cannot compute CI !")

    # The AUC covariance is computed using DeLong's method,
    # To get the confidence interval we use the normal approximation:
    # # CI = AUC ± Z * sqrt(AUC_cov)

    ci_lower, ci_upper = stats.norm.interval(0.95, loc=auc, scale=np.sqrt(auc_cov))

    return max(0, ci_lower), min(1, ci_upper)


def print_metrics(start_txt, dict_metrics):
    """
    Prints performance metrics in a clean, styled format.
    """
    print("\n" + "=" * 50)
    print(f"▶ {start_txt.upper()}")
    print("-" * 50)

    print(f"{'Metric':<20} | {'Value (%)':>10}")
    print("-" * 50)
    print(f"{'Accuracy':<20} | {dict_metrics['acc']:>9.1f}")
    print(f"{'Recall (Sensitivity)':<20} | {dict_metrics['recall']:>9.1f}")
    print(f"{'Specificity':<20} | {dict_metrics['specificity']:>9.1f}")
    print(f"{'Precision':<20} | {dict_metrics['precision']:>9.1f}")
    print(f"{'F1 Score':<20} | {dict_metrics['f1']:>9.1f}")
    print("=" * 50)


def compute_sigmoid_ci(x_train, y_train_proba, feature, model, x_range, z=1.96):

    # Build design matrix for new points (x_range)
    X_design = np.hstack([
        np.ones_like(x_range).reshape(-1, 1),
        x_range.reshape(-1, 1)
    ])

    # Extract model coefficients
    coef = np.concatenate([model.intercept_, model.coef_.flatten()])

    # Design matrix for the training set :
    X_fit = np.hstack([
        np.ones_like(x_train[feature]).reshape(-1, 1),
        x_train[[feature]].values
    ])

    # Compute V matrix = diag(p * (1 - p))
    p = y_train_proba
    V = np.diag(p * (1 - p))

    # Covariance of beta (asymptotic)
    cov_matrix = np.linalg.inv(X_fit.T @ V @ X_fit)

    # Logit predictions on the grid
    logit_preds = X_design @ coef

    # Standard errors: SE[η(x)] = sqrt(xᵗ Σ x)
    std_errors = np.sqrt(np.sum((X_design @ cov_matrix) * X_design, axis=1))

    # Confidence bounds on logit
    lower_logit = logit_preds - z * std_errors
    upper_logit = logit_preds + z * std_errors

    # Apply sigmoid
    y_pred = 1 / (1 + np.exp(-logit_preds))
    y_lower = 1 / (1 + np.exp(-lower_logit))
    y_upper = 1 / (1 + np.exp(-upper_logit))

    return y_pred, y_lower, y_upper