import os
import matplotlib.pyplot as plt


def save_figure(fig, output_dir: str, name: str) -> str:
    """Save matplotlib Figure as PNG and PDF.

    Args:
        fig: matplotlib Figure.
        output_dir: Output root directory. Figures saved to ``<output_dir>/figures/``.
        name: Base filename without extension.

    Returns:
        Path to the saved PNG file. The caller is responsible for closing
        the figure via ``plt.close(fig)`` when done.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    return os.path.join(fig_dir, f"{name}.png")
