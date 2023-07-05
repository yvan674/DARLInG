"""Signal to Image Exploration.

To explore what the images look like.

Research Question:
    - What does each signal to image technique actually produce?

Answers:
    - A nice 2d square image. MTF actually seems the most human interpretable.
      All 3 types are distinct though.

References:
    J.-P Eckmann, S. Oliffson Kamphorst and D Ruelle, “Recurrence Plots of
        Dynamical Systems”. Europhysics Letters (1987).

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from pathlib import Path

import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, MarkovTransitionField, \
    RecurrencePlot

from data_utils.widar_dataset import WidarDataset
from signal_processing.pipeline import Pipeline
from signal_processing.phase_unwrap import PhaseUnwrap
from signal_processing.phase_filter import PhaseFilter
from signal_processing.lowpass_filter import LowPassFilter
from signal_processing.standard_scaler import StandardScaler


def process_and_show(dataset: WidarDataset, amp_pipe: Pipeline,
                     phase_pipe: Pipeline, transform: any,
                     title: str):
    amp_imgs = []
    phase_imgs = []

    for i in range(12):
        x_amp, x_phase, _, _ = dataset[i]
        amp_processed = amp_pipe(x_amp)
        phase_processed = phase_pipe(x_phase)

        amp_imgs.append(transform.transform(amp_processed[:, :, 0]))
        phase_imgs.append(transform.transform(phase_processed[:, :, 0]))

    # Create a grid of the images as side by side amp and phase images for each
    # sample
    fig, axs = plt.subplots(12, 2, figsize=(4, 12))
    # Hide ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(12):
        axs[i, 0].imshow(amp_imgs[i][0])
        axs[i, 1].imshow(phase_imgs[i][0])

    # Set title for each column
    axs[0, 0].set_title("Amplitude")
    axs[0, 1].set_title("Phase")

    # Set figure title
    fig.suptitle(title)

    plt.show()


def main():
    data_path = Path("/Users/Yvan/Git/DARLInG/data/")
    std_path = Path("/Users/Yvan/Git/DARLInG/data/widar_small")

    phase_scaler = StandardScaler(std_path, "phase")
    amp_scaler = StandardScaler(std_path, "amp")

    phase_pipe = Pipeline([PhaseUnwrap(),
                           PhaseFilter([3, 3, 1], [3, 3, 1]),
                           LowPassFilter(250, 1000),
                           phase_scaler])
    amp_pipe = Pipeline([LowPassFilter(250, 1000),
                         amp_scaler])

    dataset = WidarDataset(data_path, "train", "small", return_bvp=False)

    process_and_show(dataset, amp_pipe, phase_pipe, GramianAngularField(),
                     "GAF")

    process_and_show(dataset, amp_pipe, phase_pipe, MarkovTransitionField(),
                     "MTF")
    process_and_show(dataset, amp_pipe, phase_pipe, RecurrencePlot(),
                     "Recurrence Plot")


if __name__ == '__main__':
    main()
