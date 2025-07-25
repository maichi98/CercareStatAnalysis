from pathlib import Path
import pandas as pd


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
