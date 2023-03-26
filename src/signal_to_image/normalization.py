"""Normalization.

Performs normalization on the data.

Author:
    Yvan Satyawan <y.p.satyawan@student.tue.nl>
"""
from typing import Tuple

import numpy as np


def normalize_data(train_arr: np.ndarray,
                   test_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalizes data using a sigmoid function to be between [0, 1].

    Normalization steps:
        2. Use standard scaling on seen RSSI values
        3. Apply sigmoid function to normalize to [0, 1]
        4. Replace unseen RSSI values with 0.

    Args:
        train_arr: The training array to be normalized.
        test_arr: The test array to be normalized.
    """
    extracted = train_arr[train_arr < 0]
    mean = extracted.mean()
    std = extracted.std()

    train_normalized = np.where(train_arr < 0,
                                1 / (1 + np.exp(-((train_arr - mean) / std))),
                                0.)
    test_normalized = np.where(test_arr < 0,
                               1 / (1 + np.exp(-((test_arr - mean) / std))),
                               0.)

    return train_normalized, test_normalized
