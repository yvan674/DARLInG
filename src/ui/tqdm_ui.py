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
        total_steps = (valid_steps + train_steps) * epochs
        self.current_epoch = 1
        self.total_epochs = epochs
        self.prog_bar = tqdm(desc=f"Epoch: 1 of {epochs} | "
                                  f"Training loss: -.--", total=total_steps)

    def update_data(self, data: dict[str, any]):
        if "epoch" in data:
            self.current_epoch = data["epoch"] + 1
        if "train_loss" in data:
            self.prog_bar.set_description(f"Epoch: {self.current_epoch} of "
                                          f"{self.total_epochs} | "
                                          f"Training loss: "
                                          f"{data['train_loss']:.2f}")
        elif "valid_loss" in data:
            self.prog_bar.set_description(f"Epoch: {self.current_epoch} of "
                                          f"{self.total_epochs} | "
                                          f"Validation loss: "
                                          f"{data['valid_loss']:.2f}")
        for key in data.keys():
            if isinstance(data[key], float):
                data[key] = round(data[key], 3)
        self.prog_bar.write(f"{data}")

    def update_image(self, ori_img: Image, null_img: Image, embed_image: Image):
        pass

    def update_status(self, status: str):
        self.prog_bar.write(status)

    def step(self, n: int = 1):
        self.prog_bar.update(n)
