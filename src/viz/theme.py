"""Color palettes and style constants for visualization.

Colourblind-friendly (Wong 2011), distinguishable under deuteranopia,
protanopia, and tritanopia.
"""

FORMAT_COLORS = {
    "MXINT-8":  "#0072B2",
    "MXFP-8":   "#D55E00",
    "INT8-PC":  "#009E73",
    "MXINT-4":  "#56B4E9",
    "MXFP-4":   "#E69F00",
    "INT4-PC":  "#F0E442",
    "NF4-PC":   "#CC79A7",
}

TRANSFORM_COLORS = {
    "None":        "#0072B2",
    "SmoothQuant": "#D55E00",
    "Hadamard":    "#009E73",
}

HIST_COLORS = {
    "fp32_hist":  "#0072B2",
    "quant_hist": "#D55E00",
    "err_hist":   "#999999",
}

FALLBACK_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7",
                  "#56B4E9", "#E69F00", "#999999", "#000000", "#E5C494"]
