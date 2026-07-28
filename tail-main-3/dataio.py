"""Backwards-compatibility shim for the baselines-only layout.

In the baselines-only package the CSV/npy loaders lived in `dataio.py` and the
evaluation stack in `evaluation/`.  In this merged package the canonical
locations are:

    dataio.load_returns            -> fit_returns.load_returns
    dataio.feature_names_from_csv  -> fit_returns.feature_names_from_csv
    evaluation.<anything>          -> tailfm.<anything>   (identical code)

This module keeps `from dataio import load_returns` working in scripts written
against the old layout; prefer the canonical imports in new code.
"""

from fit_returns import load_returns, feature_names_from_csv

__all__ = ["load_returns", "feature_names_from_csv"]
