from signal_processing.lowpass_filter import LowPassFilter
from signal_processing.phase_unwrap import PhaseUnwrap

import matplotlib.pyplot as plt

def plot_signals(time_steps, original, processed, title, original_label,
                 processed_label):
    plt.title(title)
    plt.plot(time_steps, original, label=original_label)
    plt.plot(time_steps, processed, label=processed_label)
    plt.legend()
    plt.show()


__all__ = ["LowPassFilter", "PhaseUnwrap", "plot_signals"]
