# Label Map :
LABEL_MAP = {
    0: "Radiation Necrosis",
    1: "Tumor Recurrence"
}

# Rename some columns :
DICT_RENAMING_MAPPING = {
    "N°patient ": "num_patient",
    "Hypothèse radio avant Tram": "pred_wo_tram",
    "Hypothèse radio avec tram": "pred_w_tram"
}

# Biomarker definitions:
DICT_MARKERS = {}

# Markers with "path" :
for marker in ["ADC", "CBV_corr", "CBV_noncorr"]:

    DICT_MARKERS[marker] = {
        "path": f"{marker}_path",
        "control": f"{marker}_control",
        "ratio": f"{marker}_ratio"
    }

# Markers with "lésion" :
for marker in ["DELAY", "CTH", "CTH MAX", "OEF", "rLEAKAGE", "rCMRO2", "COV"]:

    DICT_MARKERS[marker] = {
        "path": f"{marker}_lésion",
        "control": f"{marker}_control",
        "ratio": f"{marker}_ratio"
    }

# Special-case markers (no ratio/control)
DICT_MARKERS["rRHP"] = {"path": "rRHP", "control": None, "ratio": None}
DICT_MARKERS["tram"] = {"path": "Tram", "control": None, "ratio": None}


# ML models to use :
L_MODELS = [
    "logistic_regression",
    "SVM",
    "decision_tree",
]

# Plotting palettes :


PALETTE_BINARY = {"0": "#66c2a5", "1": "#fc8d62"}   # Class 0 = green, Class 1 = orange
PALETTE_TARGET = {0: 'steelblue', 1: 'indianred'}   # Diagnosis 0 = blue, 1 = red

PALETTE_DECISION = {
    "threshold": "black",
    "youden": "limegreen",
    "default": "red"
}
