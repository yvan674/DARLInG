import matplotlib.pyplot as plt


def plot_signals(time_steps, original, processed, title, original_label,
                 processed_label):
    plt.title(title)
    plt.plot(time_steps, original, label=original_label)
    plt.plot(time_steps, processed, label=processed_label)
    plt.legend()
    plt.show()


def plot_many_signals(time_steps, signals, labels, colors=None, title=None,
                      show_legend=True):
    """Plot an arbitrary number of signals on the same figure.

    Args:
        time_steps (np.ndarray): The x-axis.
        signals (list[np.ndarray]): The signals to plot. Should be 1D.
        labels (list[str]): The labels for each signal.
        colors (list[Any]): The color for each signal. Optional.
        title (str): The title of the plot. Optional.
        show_legend (bool): Whether to show the legend.
    """
    if title is not None:
        plt.title(title)

    if colors is not None:
        for signal, label, color in zip(signals, labels, colors):
            plt.plot(time_steps, signal, label=label, color=color)
    else:
        for signal, label in zip(signals, labels):
            plt.plot(time_steps, signal, label=label)

    if show_legend:
        plt.legend()
    plt.show()


__all__ = ["plot_signals", "plot_many_signals"]
