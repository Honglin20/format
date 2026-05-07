"""Tests for src.report._registry -- table/figure registry error paths."""

import pytest


class TestGetTableFn:
    """Test get_table_fn error paths."""

    def test_unknown_key(self):
        from src.report._registry import get_table_fn

        with pytest.raises(KeyError, match="nonexistent_key"):
            get_table_fn("nonexistent_key")


class TestGetFigureFn:
    """Test get_figure_fn error paths."""

    def test_unknown_key(self):
        from src.report._registry import get_figure_fn

        with pytest.raises(KeyError, match="nonexistent_key"):
            get_figure_fn("nonexistent_key")
