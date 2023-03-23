from signal_processing.lowpass_filter import LowPassFilter
from signal_processing.phase_unwrap import PhaseUnwrap

import matplotlib.pyplot as plt

def plot_signals(time_steps, original, processed):
    plt.plot(time_steps, original, label='Original')
    plt.plot(time_steps, processed, label='Processed')
    plt.legend()
    plt.show()


__all__ = ["LowPassFilter", "PhaseUnwrap", "plot_signals"]