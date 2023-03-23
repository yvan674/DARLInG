"""Low-pass Filter.

Removes high-frequency noise from a signal.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    ChatGPT <chat.openai.com>
"""
import numpy as np
from scipy import signal

from src.signal_processing.base import SignalProcessor


class LowPassFilter(SignalProcessor):
    def __init__(self, cutoff: float, fs: float, order: int = 4):
        """Low-pass filter.

        Removes high-frequency noise from a signal.

        Args:
            cutoff: Cutoff frequency of the filter.
            fs: Sampling frequency of the signal.
            order: Order of the filter. Higher order means more attenuation of
                high-frequency noise, but also means more delay in the signal.
        """
        super().__init__()
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        nyquist_freq = fs / 2
        cutoff_norm = cutoff / nyquist_freq
        self.b, self.a = signal.butter(self.order, cutoff_norm, btype='low',
                                       analog=False, output="ba")

    def process(self, x: np.ndarray) -> np.ndarray:
        """Processes the signal.

        Args:
            x: Signal to process.

        Returns:
            Processed signal.
        """
        return signal.filtfilt(self.b, self.a, x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fs = 1000

    def plot_signals(time_steps, original, filtered):
        plt.plot(time_steps, original, label='Original')
        plt.plot(time_steps, filtered, label='Filtered')
        plt.legend()
        plt.show()

    # Try this out with the Widar dataset data
    from data_utils import WidarDataset
    from pathlib import Path
    data = WidarDataset(Path("../../data/"), "train", True)
    x = data[0][0][:, 0, 0]
    signal_time_cutoff = int(np.argwhere(x == 0)[0])
    x = x[:signal_time_cutoff]
    t = np.arange(len(x))

    for cutoff in (10, 50, 200, 400):
        lpf = LowPassFilter(cutoff, fs)
        filtered_signal = lpf(x)
        plot_signals(t, x, filtered_signal)
