from sklearn.manifold import TSNE
import numpy as np


def format_dataframe(df):
    
    _df = df.copy()

    # Fill missing num_patient values :
    _df["num_patient"] = _df["num_patient"].ffill()
    # Correct Primitifs column type :
    _df["Primitifs"] = _df["Primitifs"].astype("str")

    float_cols = [
        "DELAY_control", "CTH_control", "CTH MAX_lésion", "OEF_control",
        "rLEAKAGE_lésion", "rLEAKAGE_control", "COV_control"
    ]

    for col in float_cols:
        _df[col] = _df[col].astype(float)

    return _df


def add_ratio_columns(df):

    # Biomarkers with "_path" suffix :
    for marker in ["ADC", "CBV_corr", "CBV_noncorr"]:
        df[f"{marker}_ratio"] = df[f"{marker}_path"] / df[f"{marker}_control"]

    # Biomarkers with "_lésion" suffix :
    for marker in ["DELAY", "CTH", "CTH MAX", "OEF", "rLEAKAGE", "rCMRO2", "COV"]:
        df[f"{marker}_ratio"] = df[f"{marker}_lésion"] / df[f"{marker}_control"]


def get_adaptive_distribution_bins(data, feature, target_bin_count=100):

    min_val = data[feature].min()
    max_val = data[feature].max()
    data_range = max_val - min_val

    if data_range == 0:
        return np.linspace(min_val - 0.5, max_val + 0.5, 2)  # fallback: 1 bin
    bin_width = data_range / target_bin_count

    return np.arange(min_val, max_val + bin_width, bin_width)


def fit_tsne(x_data, perplexity):
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42
    ).fit_transform(x_data)


def winsorize_df_to_nan(df, lower=0.0, upper=1.0):

    df = df.copy()

    for col in df.select_dtypes(include=[int, float, np.number]).columns:
        q_low = df[col].quantile(lower)
        q_high = df[col].quantile(upper)

        mask = (df[col] >= q_low) & (df[col] <= q_high)
        df.loc[~mask, col] = np.nan  # Set outliers to NaN

    return df
