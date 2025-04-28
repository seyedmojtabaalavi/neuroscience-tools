import numpy as np
import matplotlib.pyplot as plt


def plot_wavelet(time, freqs, power_matrix, title="Wavelet Transform", cbar_title=None, label=None,
                 events=None, event_names=None, time_unit="ms", vmin=None, vmax=None,
                 save_path=None, show=True, cmap="cividis",
                 fig=None, ax=None):
    """
    Plot a wavelet transform with optional event markers and support for overlaying.

    Parameters:
    - fig, ax (optional): If provided, plot on existing figure/axis.
    """
    if time is None:
        time = np.arange(power_matrix.shape[1])  # Use index as time

    # Create new figure if none provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    if cbar_title is not None:
        cBarLabel = cbar_title
    else:
        cBarLabel = 'Power'
    if vmin is None or vmax is None:
        vmin = np.min(power_matrix)
        vmax = np.max(power_matrix)
    cax = ax.pcolormesh(time, freqs, power_matrix, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cax, ax=ax, label='Power')
    cbar.set_label(cBarLabel, rotation=-90, labelpad=10)

    # Add event markers
    if events is not None:
        if event_names is None:
            event_names = [f"Event {i+1}" for i in range(len(events))]
            for e, name in zip(events, event_names):
                ax.axvline(e, color='red', linestyle="--", label=name)
        elif event_names == -1:
            event_names = ''
            for e in events:
                ax.axvline(e, color='red', linestyle="--")


    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    # if label is not None:
    if event_names != -1 or events is None:
        ax.legend()
    ax.tick_params(axis='both', direction='out')  # ✅ Ticks out

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and fig is None:
        plt.show()
    elif fig is None:
        plt.close()

    return fig, ax


def plot_raster(spike_times, trial_indices, title="Raster Plot",
                events=None, event_names=None, time_unit="ms",
                save_path=None, show=True, color="black",
                fig=None, ax=None):
    """
    Plot a raster plot with optional event markers and overlay support.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for spikes, trial in zip(spike_times, trial_indices):
        ax.vlines(spikes, trial - 0.4, trial + 0.4, color=color)

    # Add event markers
    if events is not None:
        if event_names is None:
            event_names = [f"Event {i+1}" for i in range(len(events))]
        for e, name in zip(events, event_names):
            ax.axvline(e, color='red', linestyle="--", label=name)

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Trial")
    ax.set_title(title)
    if event_names != -1 or events is None:
        ax.legend()
    ax.tick_params(axis='both', direction='out')  # ✅ Ticks out

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and fig is None:
        plt.show()
    elif fig is None:
        plt.close()

    return fig, ax


def plot_psth(spike_counts, time_bins, title="Peri-Stimulus Time Histogram (PSTH)",
              events=None, event_names=None, time_unit="ms",
              save_path=None, show=True, color="black",
              fig=None, ax=None):
    """
    Plot a PSTH with optional event markers and overlay support.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(time_bins, spike_counts, width=np.diff(time_bins).mean(), color=color, edgecolor=color)

    # Add event markers
    if events is not None:
        if event_names is None:
            event_names = [f"Event {i+1}" for i in range(len(events))]
        for e, name in zip(events, event_names):
            ax.axvline(e, color='red', linestyle="--", label=name)

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Spike Count")
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis='both', direction='out')  # ✅ Ticks out

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and fig is None:
        plt.show()
    elif fig is None:
        plt.close()

    return fig, ax


def plot_psd(freqs, power_spectrum, title="Power Spectral Density (PSD)",
             events=None, event_names=None, time_unit="Hz",
             save_path=None, show=True, color="black",
             log_scale=False, fig=None, ax=None):
    """
    Plot a Power Spectral Density (PSD) with optional event markers and overlay support.

    Parameters:
    - freqs (array): Frequency values.
    - power_spectrum (array): Power values corresponding to frequencies.
    - title (str): Plot title.
    - events (list, optional): List of event times (e.g., stimulus onset).
    - event_names (list, optional): Names corresponding to event markers.
    - time_unit (str): Time unit for x-axis (default: "Hz").
    - save_path (str, optional): File path to save the plot.
    - show (bool): Whether to show the plot.
    - color (str): Line color.
    - log_scale (bool): Whether to use log-log scale.
    - fig, ax (optional): Existing figure and axis for overlaying plots.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(freqs, power_spectrum, color=color, label="PSD")

    # Log scale option
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Add event markers
    if events is not None:
        if event_names is None:
            event_names = [f"Event {i+1}" for i in range(len(events))]
        for e, name in zip(events, event_names):
            ax.axvline(e, color='red', linestyle="--", label=name)

    ax.set_xlabel(f"Frequency ({time_unit})")
    ax.set_ylabel("Power (dB)" if log_scale else "Power")
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis='both', direction='out')  # ✅ Ticks out

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and fig is None:
        plt.show()
    elif fig is None:
        plt.close()

    return fig, ax
