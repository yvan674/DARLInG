"""Base UI.

Base class for all UIs so implementation is consistent across CLI and GUI.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from abc import ABC, abstractmethod

from PIL.Image import Image


class BaseUI(ABC):
    def __init__(self, train_steps: int, valid_steps: int, epochs: int,
                 initial_data: dict[str, any]):
        """Base UI class for training.

        Args:
            train_steps: Number of steps used in the training. Can be
                calculated using len(train_loader).
            valid_steps: Number of steps used for validation. Can be calculated
                using len(valid_loader).
            epochs: Total number of epochs to train for.
        """
        self.train_steps = train_steps
        self.valid_steps = valid_steps
        self.total_steps = (valid_steps + train_steps) * epochs
        self.epochs = epochs
        self.data_dict = initial_data
        self.current_epoch = 1

    @abstractmethod
    def update_status(self, status: str):
        """Updates the status presented in the UI."""
        raise NotImplementedError

    @abstractmethod
    def update_data(self, data: dict[str, any]):
        """Updates the data presented in the UI."""
        raise NotImplementedError

    @abstractmethod
    def update_image(self, ori_img: Image, null_img: Image,
                     embed_image: Image | None):
        """Updates the image presented in the UI."""
        raise NotImplementedError

    def step(self, n: int):
        """Takes a step; both for validation and training steps."""
        raise NotImplementedError
