import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.viz.save import save_figure


class TestSaveFigure:
    def test_save_figure_creates_png_and_pdf(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, tmpdir, "test_chart")
            assert os.path.exists(os.path.join(tmpdir, "figures", "test_chart.png"))
            assert os.path.exists(os.path.join(tmpdir, "figures", "test_chart.pdf"))

    def test_save_figure_returns_path(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, tmpdir, "my_plot")
            assert path.endswith("my_plot.png")
