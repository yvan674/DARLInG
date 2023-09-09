"""Cluster Logger UI.

Basic CLI interface WITHOUT tqdm, for use on the server for output to a log
file.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from PIL.Image import Image
from time import time
from datetime import datetime, timezone

from ui.base_ui import BaseUI


class ClusterLoggerUI(BaseUI):
    def __init__(self, train_steps: int, valid_steps: int, epochs: int,
                 initial_data: dict[str, any]):
        super().__init__(train_steps, valid_steps, epochs, initial_data)
        self.curr_step = 0

    def print_formatter(self, text):
        """Prints the text with the timestamp in front of it."""
        local_time = datetime.fromtimestamp(time(), timezone.utc).astimezone()
        local_time = local_time.strftime("%Y-%m-%d %H:%M:%S")

        timestamp = (f"[{local_time} E{self.current_epoch:03d} "
                     f"S{self.curr_step:07d}] ")

        spaces = " " * len(timestamp)

        if "\n" in text:
            text.replace("\n", spaces + "\n")

        print(f"{timestamp} {text}")

    def update_status(self, status: str):
        """Updates the status by printing it to the console with a timestamp."""
        self.print_formatter(status)

    def update_data(self, data: dict[str, any]):
        text = []
        if "epoch" in data:
            self.current_epoch = data["epoch"] + 1
            text.append(f"Started epoch {self.current_epoch}")
        for key in data.keys():
            if isinstance(data[key], float):
                data[key] = round(data[key], 2)
            text.append(f"{key}: {data[key]}")
        text = ", ".join(text)
        self.print_formatter(text)

    def update_image(self, ori_img: Image, null_img: Image,
                     embed_image: Image | None):
        pass

    def step(self, n: int = 1):
        self.curr_step += n
