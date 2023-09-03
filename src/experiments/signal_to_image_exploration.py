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
from tqdm import tqdm

from data_utils.widar_dataset import WidarDataset
from signal_processing.pipeline import Pipeline
from signal_processing.standard_scaler import StandardScaler


def process_and_show(samples: dict[int, tuple],
                     save_fp: Path):
    """Processes and shows the images.

    Shows a 6x6 grid of images where each row represents a gesture and each
    pair of columns shows the amplitude and phase images for that gesture.
    The first 2 columns show the GAF transform, the second two the MTF
    transform and the last two the RP transform.

    Args:
        samples: Samples to transform.
        save_fp: Save file path.
    """
    # Make a grid of subplots with 6 rows and 6 columns
    fig, axs = plt.subplots(6, 6, figsize=(8, 8))
    # Hide ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    # Set column titles
    axs[0, 0].set_title("GAF (Amplitude)")
    axs[0, 1].set_title("GAF (Phase)")
    axs[0, 2].set_title("MTF (Amplitude)")
    axs[0, 3].set_title("MTF (Phase)")
    axs[0, 4].set_title("RP (Amplitude)")
    axs[0, 5].set_title("RP (Phase)")

    # Set row titles
    axs[0, 0].set_ylabel("Gesture 0")
    axs[1, 0].set_ylabel("Gesture 1")
    axs[2, 0].set_ylabel("Gesture 2")
    axs[3, 0].set_ylabel("Gesture 3")
    axs[4, 0].set_ylabel("Gesture 4")
    axs[5, 0].set_ylabel("Gesture 5")

    # Transforms
    transforms = (GramianAngularField(), MarkovTransitionField(),
                  RecurrencePlot())

    for i in tqdm(range(6)):
        x_amp, x_phase = samples[i]

        for j, transform in enumerate(transforms):
            amp_img = transform.transform(x_amp[:, :, 0])
            phase_img = transform.transform(x_phase[:, :, 0])

            axs[i, (j * 2) + 0].imshow(amp_img[0])
            axs[i, (j * 2) + 1].imshow(phase_img[0])

        # Tight layout
    fig.tight_layout()

    # Save figure
    fig.savefig(save_fp / f"transforms.png")
    plt.close(fig)


def main():
    data_path = Path("/Users/Yvan/Git/DARLInG/data/")

    save_fp = Path("/Users/Yvan/Git/DARLInG/figures")

    # Make sure save_fp exists
    save_fp.mkdir(exist_ok=True)

    phase_scaler = StandardScaler(data_path, "phase")
    amp_scalar = StandardScaler(data_path, "amp")

    amp_pipe = Pipeline.from_str_list(["lowpass_filter",
                                       "standard_scaler"],
                                      None,
                                      amp_scalar, 0)
    phase_pipe = Pipeline.from_str_list(["phase_unwrap", "phase_filter",
                                         "lowpass_filter", "standard_scaler"],
                                        None, phase_scaler,
                                        0)

    samples = {}

    gestures = {0, 1, 2, 3, 4, 5}

    dataset = WidarDataset(data_path, "train",
                           "single_user_small",
                           False, None,
                           True,
                           amp_pipe, phase_pipe, False)

    for i in range(len(dataset) // 3, 0, -1):
        amp, phase, bvp, info = dataset[i]
        if info["gesture"] in gestures:
            samples[info["gesture"]] = (amp, phase)
            print(f"Found gesture {info['gesture']}")
            gestures.remove(info["gesture"])
            if len(gestures) == 0:
                break

    process_and_show(samples, save_fp)


if __name__ == '__main__':
    main()
