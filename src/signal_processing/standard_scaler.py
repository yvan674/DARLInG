"""Standard Scaler.

Standardizes the signal from a given mean and stddev value from the mean_std.csv
file.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from csv import DictReader
from pathlib import Path

import numpy as np

from signal_processing.base import SignalProcessor


class StandardScaler(SignalProcessor):
    def __init__(self, data_dir: Path, signal_type: str):
        """Standardize by removing the mean and scaling to unit variance.

        Args:
            data_dir: Path to the root data directory containing the mean and
                std CSV file.
            signal_type: The type of signal this specific object will
                standardize. Options are [`amp`, `phase`].
        """
        super().__init__()
        if signal_type not in ("amp", "phase"):
            raise ValueError(f"Signal type {signal_type} must be one of "
                             f"[`amp`, `phase`]")

        fp = data_dir / "mean_std.csv"
        with open(fp, "r") as f:
            reader: DictReader[any] = DictReader(f)
            for row in reader:
                self.mean = float(row[f"{signal_type}_mean"])
                self.std = float(row[f"{signal_type}_std"])

    def process(self, x: np.ndarray, **kwargs):
        return (x - self.mean) / self.std
