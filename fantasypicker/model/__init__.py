"""Projection model: panel construction, features, quantile training, prediction.

Submodules are imported explicitly (``from fantasypicker.model import train``)
rather than re-exported here, so that importing the dataset builder does not
drag LightGBM into memory.
"""
