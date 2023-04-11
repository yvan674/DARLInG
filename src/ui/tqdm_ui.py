"""TQDM UI.

Basic CLI interface which uses tqdm to display basic information.
"""
from PIL.Image import Image
from tqdm import tqdm

from ui.base_ui import BaseUI


class TqdmUI(BaseUI):
    """Basic CLI interface which only updates a tqdm progress bar."""
    def __init__(self, train_steps: int, valid_steps: int, epochs: int,
                 initial_data: dict[str, any]):
        super().__init__(train_steps, valid_steps, epochs, initial_data)
        total_steps = (valid_steps + 2 * train_steps) * epochs
        self.prog_bar = tqdm(desc="Training", total=total_steps)

    def update_data(self, data: dict[str, any]):
        if "train_loss" in data:
            self.prog_bar.set_description(f"Train loss: "
                                          f"{data['train_loss']:.2f}")
        elif "valid_loss" in data:
            self.prog_bar.set_description(f"Validation loss: "
                                          f"{data['valid_loss']:.2f}")
        self.prog_bar.write(f"{data}")

    def update_image(self, img: Image):
        pass

    def update_status(self, status: str):
        self.prog_bar.write(status)

    def step(self, n: int = 1):
        self.prog_bar.update(n)
