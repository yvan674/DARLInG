"""Training GUI.

This makes it possible to view the training progress through a GUI.
"""
import tkinter as tk
from datetime import timedelta

import numpy as np
from PIL import Image, ImageTk
import traceback
import sys


class Status(tk.Frame):
    def __init__(self, valid_steps: int, master=None):
        """Creates a frame that contains all the string-based statuses."""
        super().__init__(master=master, bg="#282c34", width=400, height=300)
        self.columnconfigure(0, minsize=150)
        self.columnconfigure(1, minsize=150)

        # Store max values for epoch and step
        self.max_step = 0
        self.max_epoch = 0
        self.valid_steps = valid_steps

        # Prepare tk variables with default values
        self.step_var = tk.StringVar(master, value="Step: 0/0")
        self.epoch_var = tk.StringVar(master, value="Epoch: 0/0")

        self.rate_var = tk.StringVar(master, value="Rate: 0 steps/s")
        self.time_var = tk.StringVar(master, value="Time left: 0 seconds")

        self.elbo_var = tk.StringVar(master, value="ELBO: 0.000")
        self.ce_var = tk.StringVar(master, value="CE: 0.000")

        self.status = tk.StringVar(master, value="")

        self.labels = [
            # Row 0 Labels
            tk.Label(self, textvariable=self.step_var),
            tk.Label(self, textvariable=self.epoch_var),
            # Row 1 Labels
            tk.Label(self, textvariable=self.elbo_var),
            tk.Label(self, textvariable=self.ce_var),
            # Row 2 Labels
            tk.Label(self, textvariable=self.rate_var),
            tk.Label(self, textvariable=self.time_var),
            # Row 3 Labels
            tk.Label(self, textvariable=self.status)
        ]

        # Configure all the labels and put them on the grid
        counter = 0
        for label in self.labels:
            label["bg"] = "#282c34"
            label["fg"] = "#a8afb8"
            if counter > 7:
                label.grid(row=int(counter / 2), column=counter % 2, sticky="W",
                           padx=5, pady=5)
            else:
                label.grid(row=int(counter / 2), column=counter % 2, sticky="W",
                           columnspan=2, padx=5, pady=5)
            counter += 1

    def update_data(self, step, epoch, elbo, ce, rate, validation=False):
        """Updates the string-based information within the GUI.
        The information displayed by the GUI should be updated after every step
        done by the trainer.
        Args:
            step (int): The current step of the training process.
            epoch (int): The current epoch of the training process.
            elbo (float): The ELBO loss of the network at the current step.
            ce (float): The Cross Entropy of the network at the current step.
            rate (float): The rate the network is running at in steps per
                          second.
            validation (bool): The state of the training, if it is in validation
                or in training where False means training. Defaults to False.
        """

        max_step = self.valid_steps if validation else self.max_step
        # Row 0 labels
        self.step_var.set("Step: {}/{}".format(step, max_step))
        self.epoch_var.set("Epoch: {}/{}".format(epoch, self.max_epoch))

        # Row 1 labels
        if validation:
            self.elbo_var.set("OOD: {:.3f}".format(elbo))
            self.ce_var.set("CE: {:.3f}".format(ce))
        else:
            self.elbo_var.set("ELBO: {:.3f}".format(elbo))
            self.ce_var.set("CE: {:.3f}".format(ce))

        # Row 2 labels
        if rate < 1:
            self.rate_var.set("Rate: {:.3f} secs/step".format(1 / rate))
        else:
            self.rate_var.set("Rate: {:.3f} steps/sec".format(rate))

        total_steps = (self.max_step + self.valid_steps) * self.max_epoch
        steps_taken = (self.max_step + self.valid_steps) * (epoch - 1)
        if validation:
            steps_taken += self.max_step

        steps_taken += step
        steps_left = total_steps - steps_taken
        seconds_left = int(steps_left * rate)
        self.time_var.set("Time left: {}".format(
            timedelta(seconds=seconds_left)
        ))

    def update_status(self, message):
        """Updates the status message within the GUI.
        Args:
            message (str): The new message that should be displayed.
        """
        self.status.set(message)
        print(message)

    def set_max(self, max_step, max_epoch):
        """Sets the maximum values for step and epoch.
        Args:
            max_step (int): The maximum number of steps.
            max_epoch (int): The maximum number of epochs.
        """
        self.max_step = max_step
        self.max_epoch = max_epoch

        self.step_var.set("Step: 0/{}".format(max_step))
        self.epoch_var.set("Epoch: 0/{}".format(max_epoch))


class ImageFrame(tk.Frame):
    def __init__(self, master=None):
        """Super class for frames that can contain images."""
        super().__init__(master=master, bg="#282c34",
                         width=320, height=320,
                         borderwidth=0)
        super().configure(background="#282c34")

        # Create a black image to initialize the canvas with
        black_image = np.zeros((300, 400))
        black_image = ImageTk.PhotoImage(image=Image.fromarray(black_image))

        # Set up the canvas
        self.canvas = tk.Canvas(self, bg="#282c34",
                                width=320, height=320)
        self.img = black_image
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw",
                                                   image=self.img)
        self.canvas.pack()

    def update_image(self, input_image):
        """Updates the image that is to be displayed.
        Args:
            input_image (Image): The image to be placed in this frame
        """
        input_image = input_image.resize((320, 320), Image.NEAREST)
        self.img = ImageTk.PhotoImage(image=input_image)
        self.canvas.itemconfig(self.canvas_img, image=self.img)


class TrainingGUI:
    def __init__(self, valid_steps: int):
        """Creates a GUI for training using tkinter."""
        self.root = tk.Tk()
        self.root.title("Training")
        self.root.configure(background="#282c34")
        self.root.geometry("640x640")
        self.root.resizable(False, False)

        # Configure the grid and geometry
        self.root.columnconfigure(0, minsize=320)
        self.root.columnconfigure(1, minsize=200)
        self.root.rowconfigure(0, minsize=320)
        self.root.rowconfigure(1, minsize=320)

        # Setup the widgets
        self.widgets = [
            ImageFrame(self.root),
            Status(valid_steps, self.root),
            ImageFrame(self.root)
        ]

        # Place the widgets
        self.widgets[0].grid(row=0, column=0)
        self.widgets[1].grid(row=0, column=1)
        self.widgets[2].grid(row=1, column=0)

        # Finally, lift the window to the top
        self._lift()

    def update_data(self, step, epoch, elbo, ce, rate, validation=False):
        """Updates the string-based data in the GUI."""
        self.widgets[1].update_data(step, epoch, elbo, ce, rate, validation)
        self._update()

    def update_status(self, message):
        """Updates the status message in the GUI."""
        self.widgets[1].update_status(message)
        self._update()

    def set_max_values(self, total_steps, total_epochs):
        """Sets the max value for steps and epochs."""
        self.widgets[1].set_max(total_steps, total_epochs)

    def update_image(self, original: Image,
                     reconstructed: Image):
        """Updates the images in the GUI."""
        self.widgets[0].update_image(original)
        self.widgets[2].update_image(reconstructed)

        self._update()

    def _lift(self):
        """Brings the tkinter window to the front.

        Note:
            This is required, and not simply calling root.lift(), because of
            quirks of macOS.
        """
        self.root.lift()
        self.root.call('wm', 'attributes', '.', '-topmost', '1')
        self.root.call('wm', 'attributes', '.', '-topmost', '0')

    def _update(self):
        """Internal update call."""
        self.root.update()

        # Since we can only detect if the root has been destroyed after
        # update(), then we have to use a try except block to make sure that we
        # don't try to update idle tasks after root has been destroyed and to
        # instead just kill the process.
        try:
            self.root.update_idletasks()
        except tk.TclError:
            # Enable safe exit
            traceback.clear_frames(sys.exc_info()[2])
            sys.exit()

    def mainloop(self):
        """Called at the end of training to keep the window active."""
        self.root.mainloop()
