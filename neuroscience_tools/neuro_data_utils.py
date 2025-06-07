import pandas as pd
import pickle
import zstandard as zstd
import numpy as np
import os
try:
    import json
except:
    pass
try:
    # Preferred import for newer versions of scipy
    from scipy.signal.windows import gaussian
except ImportError:
    # Fallback for older versions
    from scipy.signal import gaussian
from scipy import signal
from scipy.signal import welch, hilbert, find_peaks, convolve, windows
import pywt
from tqdm import tqdm
import warnings
import concurrent.futures
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from numba import njit
from scipy.stats import exponnorm
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("⚠️  JAX is not installed. GPU-based functions like superlet will not work.")
try:
    from . import superlets
except:
    pass
import io
from pathlib import Path
from typing import Any

#######################################################################################################################


# ======================================================================================================================
# Functions to be used internally:

def get_optimal_threads():
    """
    Get the optimal number of threads by leaving one core free.
    """
    return max(1, os.cpu_count() - 1)


# 🔒 Use _KNOWN_COLUMNS instead of __KNOWN_COLUMNS to avoid name mangling
_KNOWN_COLUMNS = [
    "StartTrial", "Correct", "Wrong", "Miss", "Abort",
    "FP", "FP_Tin", "CuePos_X", "CuePos_Y", "CueOn", "CueOff",
    "TargetOn", "TargetOff", "Target_Orientation", "TargetPos_X", "TargetPos_Y",
    "Target_Color", "Distractors", "DistractorsPos_X", "DistractorsPos_Y",
    "Distractors_Orientation", "Distractors_Color", "nbDistractors", "Reaction_Time",
    "CueToTarget_Time", "Reward", "ManetteOn", "ManetteOff", "FlashV_On", "FlashV_Off",
    "EndTrial",

    # Additional Neural & Eye Tracking Data
    "FEF_LFP", "LIP_LFP", "FEF_MUA", "LIP_MUA",
    "FEF_Spike", "LIP_Spike", "Eye_Pupil", "Eye_X", "Eye_Y",
    "Eye_plx_Pupil", "Eye_plx_X", "Eye_plx_Y"
]


def is_pandas(d):
    return isinstance(d, (pd.DataFrame, pd.Series))


def reshape_3d_to_2d(data):
    if data.ndim == 3:
        nb_trials, nb_channels, data_points = data.shape
        return data.reshape(nb_trials * nb_channels, data_points)
    return data


def reshape_2d_to_3d(data, nb_trials, nb_channels):
    data_points = data.shape[1]
    return data.reshape(nb_trials, nb_channels, data_points)


def reshape_3d_to_4d(data, nb_trials, nb_channels):
    frequencies, time_points = data.shape[1], data.shape[2]  # Extract frequency & time dimensions
    return data.reshape(nb_trials, nb_channels, frequencies, time_points)


def process_trial_for_compute_firing_rate(args):
    fs, window_size, method, step, factor, trial = args
    if np.isnan(trial).all():
        return np.full((len(trial) // step,), np.nan)
    trial = np.nan_to_num(trial, nan=0)
    if np.all(np.isin(trial, [0, 1])):
        spikes = trial
    else:
        bins = np.arange(0, np.nanmax(trial) + window_size, 1 / fs * factor)
        spikes, _ = np.histogram(trial[~np.isnan(trial)], bins=bins)
    kernel = np.ones(window_size) if method == 'square' else gaussian(window_size, std=window_size / 6)
    kernel /= kernel.sum()
    firing_rate = np.convolve(spikes, kernel, mode='same')[::step]
    return firing_rate


def convert_to_float32(obj):
    """
    Recursively convert all numeric numpy arrays and pandas DataFrames to np.float32 in nested structures.
    """
    import numpy as np
    import pandas as pd

    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.number):
        return obj.astype(np.float32)

    elif isinstance(obj, pd.DataFrame):
        obj = obj.copy()  # avoid modifying original
        numeric_cols = obj.select_dtypes(include=[np.number]).columns
        obj[numeric_cols] = obj[numeric_cols].astype(np.float32)
        return obj

    elif isinstance(obj, dict):
        return {k: convert_to_float32(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_to_float32(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(convert_to_float32(v) for v in obj)

    else:
        return obj


def compute_time_points(data_points, fs=1000, time_unit='ms', time_window=None):
    """
    Compute time points based on data points, sampling frequency, and time window.

    Args:
        data_points (int): Number of data points.
        fs (int): Sampling frequency in Hz.
        time_unit (str, optional): Time unit ('ms', 's', 'us', 'cs', 'm'). Default is 'ms'.
        time_window (tuple, optional): (start, end) time in the given time unit. Default is None.

    Returns:
        np.ndarray: Array of time points corresponding to each data point (float32).
    """
    # Convert time unit to seconds
    time_unit_factors = {'ms': 1e-3, 's': 1, 'us': 1e-6, 'cs': 1e-2, 'm': 60}
    factor = time_unit_factors.get(time_unit, 1e-3)  # Default is 'ms' if an invalid unit is given

    # Compute time step (sampling period)
    dt = np.float32(1.0 / fs)  # Ensure dt is float32

    # If time_window is provided, compute time points accordingly
    if time_window is not None:
        start, end = np.float32(time_window)  # Convert time_window to float32
        time_points = np.linspace(start, end, data_points, dtype=np.float32)
    else:
        # Default: Compute time points starting from zero
        time_points = np.arange(data_points, dtype=np.float32) * dt / np.float32(factor)  # Convert to desired time unit

    return time_points  # ✅ Ensures float32 output


def process_psd(args):
    """
    Compute the Power Spectral Density (PSD) for a single signal, ensuring float32.

    Args:
        signal (np.ndarray): 1D array representing a single trial or channel.
        fs (int): Sampling frequency in Hz.
        freq_range (tuple): Frequency range to compute PSD.
        nperseg (int): Number of points per segment.

    Returns:
        np.ndarray: PSD values within the specified frequency range.
    """
    signal, fs, freq_range, nperseg = args

    f, Pxx = welch(signal, fs=fs, nperseg=min(len(signal), nperseg))

    if freq_range is not None:
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        return Pxx[freq_mask]  # ✅ No need for extra astype(np.float32)

    return Pxx  # ✅ No need for extra astype(np.float32)


def process_wavelet(args):
    """
    Compute the Wavelet Transform for a single signal, ensuring float32.

    Args:
        signal (np.ndarray): 1D array representing a single trial or channel.
        fs (int): Sampling frequency in Hz.
        wavelet (str): Wavelet type.
        scales (np.ndarray): Array of scales to use for the wavelet transform.

    Returns:
        np.ndarray: Wavelet coefficients (power spectrum, float32).
    """
    signal, fs, wavelet, scales = args

    # Compute wavelet transform
    coefficients, _ = pywt.cwt(signal, scales, wavelet, 1 / fs)

    # Compute power spectrum and ensure float32
    return np.square(np.abs(coefficients), dtype=np.float32)


def process_superlet(args):
    """
    Compute the Superlet Transform for a single signal using adaptive superlets.

    Args:
        args: Tuple containing:
            - signal (1D np.ndarray): The input signal (e.g., a trial or channel).
            - freqs (1D np.ndarray): Frequencies of interest.
            - fs (int): Sampling frequency.
            - base_cycle (int): Base number of cycles.
            - min_order (int): Minimum superlet order.
            - max_order (int): Maximum superlet order.
            - mode (str): 'mul' or 'add'.

    Returns:
        np.ndarray: Superlet power (float32), shape (freqs x time).
    """
    signal, freqs, fs, base_cycle, min_order, max_order, mode = args
    signal = jnp.asarray(signal, dtype=jnp.float32)
    freqs = jnp.asarray(freqs, dtype=jnp.float32)

    tfr = superlets.adaptive_superlet_transform(signal, freqs, fs, base_cycle, min_order, max_order, mode=mode)
    return np.array(jnp.abs(tfr), dtype=np.float32)


def apply_filter(data, band, fs=1000, filter_type="iir", order=4):
    """
    Apply a bandpass filter to a given 1D signal, ensuring float32 dtype.

    Parameters:
    - data (np.ndarray): 1D input signal.
    - band (tuple): Frequency range for bandpass filtering (low, high).
    - fs (int, optional): Sampling rate in Hz (default: 1000).
    - filter_type (str, optional): Filter type ('iir' or 'fir', default: 'iir').
    - order (int, optional): Filter order (default: 4).

    Returns:
    - np.ndarray: Filtered signal (float32).
    """
    # Compute Nyquist frequency
    nyquist = np.float32(fs / 2.0)
    low, high = band[0] / nyquist, band[1] / nyquist  # Implicitly float32 if input is float32

    # FIR filter
    if filter_type.lower() == "fir":
        taps = signal.firwin(order + 1, [low, high], pass_zero=False, fs=fs)
        return signal.lfilter(taps, 1.0, data)

    # IIR filter
    elif filter_type.lower() == "iir":
        b, a = signal.butter(order, [low, high], btype="band", output="ba")
        return signal.filtfilt(b, a, data)

    else:
        raise ValueError("Invalid filter_type. Use 'iir' or 'fir'.")


def process_sliding_psd(args):
    """
    Compute the Power Spectral Density (PSD) for a given signal using a sliding window.

    Parameters:
    - args: Tuple containing:
        - signal (np.ndarray): 1D array representing a single trial or channel.
        - fs (int): Sampling frequency in Hz.
        - freq_range (tuple): Frequency range for PSD filtering.
        - window_size (int): Window size in samples.
        - step_size (int): Step size in samples.
        - nperseg (int): Number of points per segment for PSD.

    Returns:
    - np.ndarray: PSD matrix with time-frequency representation (float32).
    - np.ndarray: Frequency points (float32).
    """
    signal, fs, freq_range, window_size, step_size, nperseg = args

    num_windows = (len(signal) - window_size) // step_size + 1
    time_points = np.arange(num_windows, dtype=np.float32) * (step_size / np.float32(fs))

    psd_results = []

    for i in range(num_windows):
        segment = signal[i * step_size: i * step_size + window_size]
        f, Pxx = welch(segment, fs=fs, nperseg=min(len(segment), nperseg))

        if freq_range is not None:
            freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
            psd_results.append(Pxx[freq_mask])  # ✅ No need for redundant astype(np.float32)
        else:
            psd_results.append(Pxx)  # ✅ No need for redundant astype(np.float32)

    return np.array(psd_results, dtype=np.float32).T #, f.astype(np.float32)


def apply_bandstop_filter(data, fs, band, filter_type="iir", order=4):
    """
    Apply a band-stop (notch) filter to remove a given frequency band, ensuring float32.

    Parameters:
    - data (ndarray): 1D array of time-series data.
    - fs (int): Sampling frequency in Hz.
    - band (tuple): (low_freq, high_freq) range of frequencies to remove.
    - filter_type (str): 'iir' (default) or 'fir' filter type.
    - order (int): Filter order.

    Returns:
    - filtered_data (ndarray): Data after band removal (float32).
    """
    low_freq, high_freq = band  # No need to cast to float32 explicitly

    if filter_type.lower() == "iir":
        # Design a bandstop IIR filter
        sos = signal.butter(order, [low_freq, high_freq], btype='bandstop', fs=fs, output='sos')
        return signal.sosfiltfilt(sos, data)

    elif filter_type.lower() == "fir":
        # Design a bandstop FIR filter
        num_taps = int(order * 10 + 1)  # Empirical formula for FIR
        fir_coeffs = signal.firwin(num_taps, [low_freq, high_freq], pass_zero="bandstop", fs=fs)
        return signal.filtfilt(fir_coeffs, 1.0, data)

    else:
        raise ValueError("Invalid filter type. Choose 'iir' or 'fir'.")


def ensure_float32(array):
    """
    Ensures that the given NumPy array is float32. If it's already float32, returns unchanged.

    Args:
        array (np.ndarray): The input array.

    Returns:
        np.ndarray: The array converted to float32 if necessary.
    """
    if isinstance(array, np.ndarray) and array.dtype != np.float32:
        return array.astype(np.float32)
    return array


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# Functions to be used Externally:

class load_zst_file:
    """Class to load a ZST file and enable full autocompletion for its data."""

    def __init__(self, filepath, threads=-2):
        self._data = None  # Store raw dictionary
        # Create a progress bar with a total number of steps
        pbar1 = tqdm(total=100, desc="Loading data ...")

        nbCores = np.arange(1, os.cpu_count() + 1)
        core = nbCores[threads]
        pbar1.update(10)

        with open(filepath, 'rb') as file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(file) as reader:
                data = pickle.load(reader)

        pbar1.update(30)

        if isinstance(data, dict):  # ✅ Now expecting a dictionary
            data = self.__ensure_float32_in_dict(data)  # Store dictionary internally
            self.TooLateRespond = 750
            self.TooEarlyRespond = 330
            # self._data = self.__Recompute_trial_type(data)
            self._data = data
            del data
        else:
            raise ValueError("The loaded file does not contain a dictionary.")

        self.StartTrial = self._data["StartTrial"] if "StartTrial" in self._data else None
        self.Correct = self._data["Correct"] if "Correct" in self._data else None
        self.Wrong = self._data["Wrong"] if "Wrong" in self._data else None
        self.Miss = self._data["Miss"] if "Miss" in self._data else None
        self.Abort = self._data["Abort"] if "Abort" in self._data else None
        self.FP = self._data["FP"] if "FP" in self._data else None
        self.FP_Tin = self._data["FP_Tin"] if "FP_Tin" in self._data else None
        pbar1.update(5)
        self.CuePos_X = self._data["CuePos_X"] if "CuePos_X" in self._data else None
        self.CuePos_Y = self._data["CuePos_Y"] if "CuePos_Y" in self._data else None
        self.CueOn = self._data["CueOn"] if "CueOn" in self._data else None
        self.CueOff = self._data["CueOff"] if "CueOff" in self._data else None
        self.TargetOn = self._data["TargetOn"] if "TargetOn" in self._data else None
        self.TargetOff = self._data["TargetOff"] if "TargetOff" in self._data else None
        self.EyeIn = self._data["Eye_In"] if "Eye_In" in self._data else None
        self.EyeOut = self._data["Eye_Out"] if "Eye_Out" in self._data else None
        pbar1.update(5)
        self.Target_Orientation = self._data["Target_Orientation"] if "Target_Orientation" in self._data else None
        self.TargetPos_X = self._data["TargetPos_X"] if "TargetPos_X" in self._data else None
        self.TargetPos_Y = self._data["TargetPos_Y"] if "TargetPos_Y" in self._data else None
        self.Target_Color = self._data["Target_Color"] if "Target_Color" in self._data else None
        self.Distractors = self._data["Distractors"] if "Distractors" in self._data else None
        self.DistractorsPos_X = self._data["DistractorsPos_X"] if "DistractorsPos_X" in self._data else None
        self.DistractorsPos_Y = self._data["DistractorsPos_Y"] if "DistractorsPos_Y" in self._data else None
        self.Distractors_Orientation = self._data[
            "Distractors_Orientation"] if "Distractors_Orientation" in self._data else None
        pbar1.update(5)
        self.Distractors_Color = self._data["Distractors_Color"] if "Distractors_Color" in self._data else None
        self.nbDistractors = self._data["nbDistractors"] if "nbDistractors" in self._data else None
        self.Reaction_Time = self._data["Reaction_Time"] if "Reaction_Time" in self._data else None
        self.CueToTarget_Time = self._data["CueToTarget_Time"] if "CueToTarget_Time" in self._data else None
        self.CueToDist_First = self._data["CueToDist_First"] if "CueToDist_First" in self._data else None
        self.CueToDist_Last = self._data["CueToDist_Last"] if "CueToDist_Last" in self._data else None
        self.Reward = self._data["Reward"] if "Reward" in self._data else None
        self.ManetteOn = self._data["ManetteOn"] if "ManetteOn" in self._data else None
        self.ManetteOff = self._data["ManetteOff"] if "ManetteOff" in self._data else None
        self.FlashV_On = self._data["FlashV_On"] if "FlashV_On" in self._data else None
        self.FlashV_Off = self._data["FlashV_Off"] if "FlashV_Off" in self._data else None
        self.EndTrial = self._data["EndTrial"] if "EndTrial" in self._data else None
        pbar1.update(5)
        # Additional Neural & Eye Tracking Data
        self.FEF_LFP = self._data["FEF_LFP"] if "FEF_LFP" in self._data else None
        pbar1.update(5)
        self.LIP_LFP = self._data["LIP_LFP"] if "LIP_LFP" in self._data else None
        pbar1.update(5)
        self.FEF_MUA = self._data["FEF_MUA"] if "FEF_MUA" in self._data else None
        pbar1.update(5)
        self.LIP_MUA = self._data["LIP_MUA"] if "LIP_MUA" in self._data else None
        pbar1.update(5)
        self.FEF_Spike = self._data["FEF_Spike"] if "FEF_Spike" in self._data else None
        pbar1.update(5)
        self.LIP_Spike = self._data["LIP_Spike"] if "LIP_Spike" in self._data else None
        pbar1.update(5)
        self.Eye_Pupil = self._data["Eye_Pupil"] if "Eye_Pupil" in self._data else None
        pbar1.update(5)
        self.Eye_X = self._data["Eye_X"] if "Eye_X" in self._data else None
        self.Eye_Y = self._data["Eye_Y"] if "Eye_Y" in self._data else None
        self.Eye_plx_Pupil = self._data["Eye_plx_Pupil"] if "Eye_plx_Pupil" in self._data else None
        self.Eye_plx_X = self._data["Eye_plx_X"] if "Eye_plx_X" in self._data else None
        self.Eye_plx_Y = self._data["Eye_plx_Y"] if "Eye_plx_Y" in self._data else None
        self.Unit_MetaData = self._data["Unit_MetaData"] if "Unit_MetaData" in self._data else None
        self.LFP_MetaData = self._data["LFP_MetaData"] if "LFP_MetaData" in self._data else None

        del self._data
        pbar1.update(5)
        pbar1.close()

    def __ensure_float32_in_dict(self, data):
        """
        Ensures all NumPy arrays inside a dictionary are converted to float32.

        Args:
            data (dict): Dictionary containing NumPy arrays or other data types.

        Returns:
            dict: Updated dictionary with NumPy arrays in float32 format.
        """
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.dtype != np.float32:
                data[key] = value.astype(np.float32)
        return data  # Return the modified dictionary

    def __Recompute_trial_type(self, data):
        """
        Optimally processes electrophysiology data by modifying correctness labels and computing
        reaction times based on reward availability and distractor presence.
        Ensures numerical fields remain float32 and correctness labels remain int8.
        """
        num_trials = len(data["Reward"])

        for trl in range(num_trials):
            if np.isnan(data["Reward"][trl]) and data["Correct"][trl]:  # Check if Reward is NaN
                if data["Reaction_Time"][trl] > self.TooLateRespond:
                    data["Correct"][trl] = np.int8(0)
                    data["Miss"][trl] = np.int8(1)
                elif not data["nbDistractors"][trl]:
                    data["Correct"][trl] = np.int8(0)
                    data["Abort"][trl] = np.int8(1)
                else:
                    nbDistractor = int(data["nbDistractors"][trl] - 1)  # Last distractor index
                    data["Correct"][trl] = np.int8(0)
                    data["Wrong"][trl] = np.int8(1)

                    # Ensure float32 precision
                    data["TargetOn"][trl] = np.float32(data["Distractors"][trl, nbDistractor])
                    data["Reaction_Time"][trl] = np.float32(
                        data["ManetteOn"][trl] - data["Distractors"][trl, nbDistractor])
                    data["CueToTarget_Time"][trl] = np.float32(
                        data["Distractors"][trl, nbDistractor] - data["CueOn"][trl])

        return data

    def Get_RT_Bounds(self, left_restriction=1, right_restriction=4, min_rt=250, max_rt=900):
        """
        Fits an ex-Gaussian distribution to the RT array,
        and returns session-specific lower and upper RT thresholds.

        Parameters:
        - rt_array: array-like, the reaction times for a session
        - min_rt: minimum RT allowed to prevent anticipatory responses (default = 200 ms)

        Returns:
        - lower_bound: RT threshold below which responses are considered anticipatory
        - upper_bound: RT threshold above which responses are considered lapses
        """
        rt_array = self.Reaction_Time[np.where(self.Correct == 1)[0]]

        # Fit the ex-Gaussian model
        K, loc, scale = exponnorm.fit(rt_array)

        # Convert scipy parameters to standard ex-Gaussian terms
        mu = loc
        sigma = scale
        tau = K * scale

        # Define bounds
        lower_bound = max(min_rt, mu - left_restriction * sigma)
        upper_bound = min(max_rt, mu + right_restriction * tau)

        return lower_bound, upper_bound

    def make_only_behavior(self):
        self.FEF_LFP = None
        self.LIP_LFP = None
        self.FEF_MUA = None
        self.LIP_MUA = None
        self.FEF_Spike = None
        self.LIP_Spike = None
        self.Eye_Pupil = None
        self.Eye_X = None
        self.Eye_Y = None
        self.Eye_plx_Pupil = None
        self.Eye_plx_X = None
        self.Eye_plx_Y = None
        self.Unit_MetaData = None
        self.LFP_MetaData = None

    def make_only_behavior_and_Eye(self):
        self.FEF_LFP = None
        self.LIP_LFP = None
        self.FEF_MUA = None
        self.LIP_MUA = None
        self.FEF_Spike = None
        self.LIP_Spike = None
        self.Eye_Pupil = None
        # self.Eye_X = None
        # self.Eye_Y = None
        # self.Eye_plx_Pupil = None
        # self.Eye_plx_X = None
        # self.Eye_plx_Y = None
        self.Unit_MetaData = None
        self.LFP_MetaData = None

    def interpolate_eye_data_with_blink_detection(self, min_size=1.5, padding_ms=100, sampling_rate=1000):

        def flatten(sig):
            return sig.reshape(-1)

        def reshape(sig):
            return sig.reshape(self.Eye_Pupil.shape)

        # Step 1: Flatten
        pupil_flat = flatten(self.Eye_Pupil)
        if self.Eye_X is not None: eye_x_flat = flatten(self.Eye_X)
        if self.Eye_Y is not None: eye_y_flat = flatten(self.Eye_Y)

        # Step 2: Detect blink-related dips
        blink_mask = pupil_flat < min_size
        blink_mask |= np.isnan(pupil_flat)

        # Step 3: Expand blink regions
        padding_samples = int(padding_ms * sampling_rate / 1000)
        expanded_mask = np.copy(blink_mask)
        for i in np.where(blink_mask)[0]:
            start = max(0, i - padding_samples)
            end = min(len(pupil_flat), i + padding_samples + 1)
            expanded_mask[start:end] = True

        # Step 4: Interpolation helper
        def interpolate_signal(signal, mask):
            signal = signal.copy()
            signal[mask] = np.nan
            return pd.Series(signal).interpolate(limit_direction='both').to_numpy()

        # Step 5: Interpolate
        clean_pupil = interpolate_signal(pupil_flat, expanded_mask)
        clean_x = interpolate_signal(eye_x_flat, expanded_mask) if self.Eye_X is not None else None
        clean_y = interpolate_signal(eye_y_flat, expanded_mask) if self.Eye_Y is not None else None

        self.Eye_Pupil = reshape(clean_pupil)
        self.Eye_X = reshape(clean_x)
        self.Eye_Y = reshape(clean_y)


    def Recompute_by_rtBounds(self, lower_bound=None, upper_bound=None):

        if lower_bound is None or upper_bound is None:
            lower_bound, upper_bound = self.Get_RT_Bounds()

        ToAbort_index = np.where(
            (self.Correct == 1) & (self.nbDistractors == 0) & (self.Reaction_Time < lower_bound))[0]
        self.Correct[ToAbort_index] = 0
        self.Abort[ToAbort_index] = 1

        ToAbort_index = np.where(
            (self.Correct == 1) & (self.nbDistractors >= 0) & (self.Reaction_Time > upper_bound))[0]
        self.Correct[ToAbort_index] = 0
        self.Abort[ToAbort_index] = 1

        ToWrong_index = np.where((self.Correct == 1) & (self.nbDistractors > 0) & (self.Reaction_Time < lower_bound))[
            0]
        self.Correct[ToWrong_index] = 0
        self.Wrong[ToWrong_index] = 1
        for i in ToWrong_index:
            last_Dist = self.Distractors[i, int(self.nbDistractors[i] - 1)]
            last_DistPos_X = self.DistractorsPos_X[i, int(self.nbDistractors[i] - 1)]
            last_DistPos_Y = self.DistractorsPos_Y[i, int(self.nbDistractors[i] - 1)]
            self.TargetOn[i] = last_Dist
            self.TargetPos_X = last_DistPos_X
            self.TargetPos_Y = last_DistPos_Y

        ToPassLeft3_index = \
            np.where((self.Wrong == 1) & (self.nbDistractors == 3) & (self.Reaction_Time > upper_bound))[0]
        self.Wrong[ToPassLeft3_index] = 0
        self.Abort[ToPassLeft3_index] = 1
        ToPassLeft3_index = \
        np.where((self.Wrong == 1) & (self.nbDistractors == 3) & (self.Reaction_Time < lower_bound))[0]
        for i in ToPassLeft3_index:
            self.Distractors[i, int(self.nbDistractors[i] - 1)] = np.nan
            self.DistractorsPos_X[i, int(self.nbDistractors[i] - 1)] = np.nan
            self.DistractorsPos_Y[i, int(self.nbDistractors[i] - 1)] = np.nan
            self.nbDistractors[i] -= 1
            last_Dist = self.Distractors[i, int(self.nbDistractors[i] - 1)]
            last_DistPos_X = self.DistractorsPos_X[i, int(self.nbDistractors[i] - 1)]
            last_DistPos_Y = self.DistractorsPos_Y[i, int(self.nbDistractors[i] - 1)]
            self.TargetOn[i] = last_Dist
            self.TargetPos_X = last_DistPos_X
            self.TargetPos_Y = last_DistPos_Y

        ToPassLeft2_index = \
            np.where((self.Wrong == 1) & (self.nbDistractors == 2) & (self.Reaction_Time > upper_bound))[0]
        self.Wrong[ToPassLeft2_index] = 0
        self.Abort[ToPassLeft2_index] = 1
        ToPassLeft2_index = \
            np.where((self.Wrong == 1) & (self.nbDistractors == 2) & (self.Reaction_Time < lower_bound))[0]
        for i in ToPassLeft2_index:
            self.Distractors[i, int(self.nbDistractors[i] - 1)] = np.nan
            self.DistractorsPos_X[i, int(self.nbDistractors[i] - 1)] = np.nan
            self.DistractorsPos_Y[i, int(self.nbDistractors[i] - 1)] = np.nan
            self.nbDistractors[i] -= 1
            last_Dist = self.Distractors[i, int(self.nbDistractors[i] - 1)]
            last_DistPos_X = self.DistractorsPos_X[i, int(self.nbDistractors[i] - 1)]
            last_DistPos_Y = self.DistractorsPos_Y[i, int(self.nbDistractors[i] - 1)]
            self.TargetOn[i] = last_Dist
            self.TargetPos_X = last_DistPos_X
            self.TargetPos_Y = last_DistPos_Y

        ToPassLeft1_index = \
            np.where((self.Wrong == 1) & (self.nbDistractors == 1) & (self.Reaction_Time < lower_bound))[0]
        self.Wrong[ToPassLeft1_index] = 0
        self.Abort[ToPassLeft1_index] = 1
        ToPassLeft1_index = \
            np.where((self.Wrong == 1) & (self.nbDistractors == 1) & (self.Reaction_Time > upper_bound))[0]
        self.Wrong[ToPassLeft1_index] = 0
        self.Abort[ToPassLeft1_index] = 1


# -----------------------------------------------------------------------------------------------------------------------

def save_zst_file(data, filepath, compression_level=3, threads=None):
    """
    Save data to a .zst file with zstandard compression and convert all numeric data to float32.

    Args:
        data (dict, pd.DataFrame, or np.ndarray): The data to be saved.
        filepath (str or Path): Path to save the .zst file.
        compression_level (int): Compression level (1–22).
        threads (int, optional): Number of threads to use for compression.
    """
    if data is None:
        raise ValueError("Input data is None. Cannot save an empty file.")

    # ✅ Convert all numeric data to float32 recursively
    data = convert_to_float32(data)

    # ✅ Ensure output directory exists
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    threads = threads or max(1, os.cpu_count() - 1)

    # ✅ Save with zstandard compression
    with open(filepath, 'wb') as file:
        cctx = zstd.ZstdCompressor(level=compression_level, threads=threads)
        with cctx.stream_writer(file) as writer:
            pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------------------------------------------------------------------------------
# def get_pickle_size(obj):
#     return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

def save_general_object_zst_auto_chunk(obj, save_dir, prefix="data", chunk_size_bytes=8 * 1024**3):
    """
    Save any Python object into compressed .zst chunks with metadata.

    Args:
        obj: Any serializable Python object.
        save_dir: Directory to save chunks and metadata.
        prefix: File prefix.
        chunk_size_bytes: Max size of each chunk (default: 250MB).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Serialize the object
    pickled_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Split into chunks
    num_chunks = (len(pickled_data) + chunk_size_bytes - 1) // chunk_size_bytes
    compressor = zstd.ZstdCompressor(level=10)

    for i in range(num_chunks):
        chunk = pickled_data[i * chunk_size_bytes : (i + 1) * chunk_size_bytes]
        chunk_path = save_dir / f"{prefix}_chunk_{i:04d}.zst"
        with open(chunk_path, "wb") as f:
            f.write(compressor.compress(chunk))

    # 3. Save metadata
    meta = {
        "prefix": prefix,
        "total_chunks": num_chunks,
        "protocol": pickle.HIGHEST_PROTOCOL,
        "chunk_size_bytes": chunk_size_bytes,
        "compressor": "zstd",
    }
    meta_path = save_dir / f"{prefix}_meta.zst"
    with open(meta_path, "wb") as f:
        f.write(compressor.compress(pickle.dumps(meta)))

    print(f"✅ Saved {num_chunks} chunks + metadata to '{save_dir}'")


# ----------------------------------------------------------------------------------------------------------------------

def load_zst(filepath):
    """
    Load data from a .zst file compressed with pickle + zstandard.

    Args:
        filepath (str): Path to the .zst file.

    Returns:
        The unpickled Python object (np.ndarray, dict, pd.DataFrame, etc.).
    """
    with open(filepath, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            decompressed = reader.read()

    return pickle.loads(decompressed)

# -----------------------------------------------------------------------------------------------------------------------

def load_general_object_zst_auto_chunk(load_dir, prefix="data"):
    """
    Load and reconstruct object saved with save_general_object_zst_auto_chunk.

    Args:
        load_dir: Directory containing chunks and metadata.
        prefix: File prefix.

    Returns:
        Reconstructed Python object.
    """
    load_dir = Path(load_dir)

    # Load metadata
    meta_path = load_dir / f"{prefix}_meta.zst"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    with open(meta_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            meta = pickle.load(reader)

    # Read all chunks in order
    data = bytearray()
    for i in range(meta["total_chunks"]):
        chunk_path = load_dir / f"{prefix}_chunk_{i:04d}.zst"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing chunk: {chunk_path}")
        with open(chunk_path, "rb") as f:
            with zstd.ZstdDecompressor().stream_reader(f) as reader:
                while True:
                    chunk = reader.read(16384)
                    if not chunk:
                        break
                    data.extend(chunk)

    # Deserialize
    obj = pickle.loads(data)
    print(f"✅ Successfully loaded object with type: {type(obj).__name__}")
    return obj


# -----------------------------------------------------------------------------------------------------------------------

def compute_firing_rate(data, fs=1000, window=30, method='Gaussian', overlap='full', multi_processing=-2,
                        time_unit='ms'):
    """
    Compute firing rates from spike train data using a sliding window approach.

    Args:
        data (np.ndarray): 4D array (trials, channels, units, time points).
        fs (int, optional): Sampling frequency in Hz (default is 1000 Hz).
        window (int, optional): Window size in milliseconds (default is 30 ms).
        method (str, optional): Smoothing method ('Gaussian' or other methods).
        overlap (str or int, optional): 'full' (default, step=1), 'half' (step=window/2), or custom integer step.
        multi_processing (int, optional): -1 for all cores, -2 (default) for all but one, 0 for no multiprocessing.
        time_unit (str, optional): Unit for time points ('ms', 'us', 's', 'cs', 'm'). Default is 'ms'.

    Returns:
        np.ndarray: Firing rate array in float32.
        np.ndarray: Time points in float32.
    """
    # Ensure correct time unit scaling
    time_unit_factors = {'ms': 1, 'us': 1e-3, 's': 1000, 'cs': 10, 'm': 60000}
    factor = np.float32(time_unit_factors.get(time_unit, 1))

    # Compute window size & step size in samples
    window_size = int((window / 1000) * fs)
    step = 1 if overlap == 'full' else (
        window_size // 2 if overlap == 'half' else (overlap if isinstance(overlap, int) else window_size)
    )

    # Ensure 4D structure (trials, channels, units, time)
    if data.ndim == 4:
        nb_trials, nb_channels, nb_units, data_points = data.shape
    elif data.ndim == 3:  # Missing unit dimension → Add unit dim (assume single-unit case)
        nb_trials, nb_channels, data_points = data.shape
        data = np.expand_dims(data, axis=2)  # Add unit dimension
        nb_units = 1
    elif data.ndim == 2:  # (channels, time) → Reshape for processing
        nb_channels, data_points = data.shape
        data = np.expand_dims(data, axis=(0, 2))  # Add trial & unit dim
        nb_trials, nb_units = 1, 1
    elif data.ndim == 1:  # Single time series
        data_points = data.shape[0]
        data = np.expand_dims(data, axis=(0, 1, 2))  # Add all necessary dimensions
        nb_trials, nb_channels, nb_units = 1, 1, 1
    else:
        raise TypeError("The input data must be 1D, 2D, 3D, or 4D. Check your input format.")

    # Compute time points
    time_points = np.arange(0, data_points, step, dtype=np.float32) / fs * np.float32(1000)

    # Flatten trials & units for parallel processing
    Data = np.nan_to_num(data.reshape(nb_trials * nb_channels * nb_units, data_points), nan=0.0, copy=False)

    # Prepare input arguments for multiprocessing
    inp = [(fs, window_size, method, step, factor, row) for row in Data]

    if multi_processing != 0 and data.ndim > 1:
        num_workers = max(1, os.cpu_count() + multi_processing)  # Adjust worker count based on available CPUs

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_trial_for_compute_firing_rate, inp))

        firing_rates = np.array(results, dtype=np.float32)

    elif multi_processing == 0 and data.ndim > 1:
        results = [process_trial_for_compute_firing_rate(args) for args in inp]
        firing_rates = np.array(results, dtype=np.float32)

    else:
        firing_rates = np.array(process_trial_for_compute_firing_rate((fs, window_size, method, step, factor, Data)),
                                dtype=np.float32)

    # Reshape back to 4D (trials, channels, units, time)
    firing_rates = firing_rates.reshape(nb_trials, nb_channels, nb_units, -1)

    return firing_rates, time_points


# -----------------------------------------------------------------------------------------------------------------------

def align_data_around_event(data, event_times, before, after, fs=1000, nan_padding=True):
    """
    Aligns data around an event time, creating a windowed segment centered at the event.

    Supports both:
    - LFP data (3D: trials, channels, time)
    - Spiking data (4D: trials, channels, units, time)

    Args:
        data (np.ndarray): 3D (trials, channels, time) for LFP, or 4D (trials, channels, units, time) for spikes.
        event_times (np.ndarray): 1D array of event times (in milliseconds) for each trial.
        before (int): Time before the event (in milliseconds).
        after (int): Time after the event (in milliseconds).
        fs (int, optional): Sampling frequency in Hz (default: 1000 Hz).
        nan_padding (bool, optional): Whether to pad with NaN if data points are insufficient (default: True).

    Returns:
        np.ndarray: Data aligned to event times (float32).
        np.ndarray: Time vector with zero at the event time (float32).
    """
    # Convert time to samples
    before_samples = int(before * fs / 1000)
    after_samples = int(after * fs / 1000)
    total_samples = before_samples + after_samples

    # Determine input shape (LFP or Spiking data)
    if data.ndim == 3:  # LFP data (trials, channels, time)
        nb_trials, nb_channels, data_points = data.shape
        is_spiking = False
    elif data.ndim == 4:  # Spiking data (trials, channels, units, time)
        nb_trials, nb_channels, nb_units, data_points = data.shape
        is_spiking = True
    else:
        raise ValueError("Data must be 3D (LFP: trials, channels, time) or 4D (Spikes: trials, channels, units, time).")

    # Initialize aligned data with NaNs
    aligned_shape = (nb_trials, nb_channels, nb_units, total_samples) if is_spiking else (
    nb_trials, nb_channels, total_samples)
    aligned_data = np.full(aligned_shape, np.nan, dtype=np.float32)

    # Time vector centered at the event
    time_vector = np.linspace(-before, after, total_samples, dtype=np.float32)

    for trial_idx, event_time in enumerate(event_times):
        event_idx = round(event_time * fs / 1000)  # Convert event time to sample index

        start_idx = max(0, event_idx - before_samples)
        end_idx = min(data_points, event_idx + after_samples)

        buffer_start = max(0, before_samples - event_idx)
        buffer_end = buffer_start + (end_idx - start_idx)

        if is_spiking:
            aligned_data[trial_idx, :, :, buffer_start:buffer_end] = data[trial_idx, :, :, start_idx:end_idx]
        else:
            aligned_data[trial_idx, :, buffer_start:buffer_end] = data[trial_idx, :, start_idx:end_idx]

    return aligned_data, time_vector

# -----------------------------------------------------------------------------------------------------------------------

def align_wavelet_data_around_event(wavelet_data, event_times, before, after, fs=1000, nan_padding=True):
    """
    Aligns a list of wavelet-transformed data arrays around event times.

    Each element in `wavelet_data` is a 3D array (channels, frequencies, time).
    The time dimension may differ across trials.

    Args:
        wavelet_data (List[np.ndarray]): List of 3D arrays (channels, frequencies, time), one per trial.
        event_times (np.ndarray): Event times in milliseconds for each trial (len must match wavelet_data).
        before (int): Time before event in ms.
        after (int): Time after event in ms.
        fs (int): Sampling frequency in Hz.
        nan_padding (bool): Whether to pad with NaNs (default: True).

    Returns:
        List[np.ndarray]: List of aligned wavelet data (channels, frequencies, aligned_time).
        np.ndarray: Common time vector (float32).
    """
    aligned_data = []
    time_vector = np.linspace(-before, after, int((before + after) * fs / 1000), dtype=np.float32)
    total_samples = len(time_vector)

    for trial_idx, (trial_wav, evt_ms) in enumerate(zip(wavelet_data, event_times)):
        ch, f, t = trial_wav.shape
        aligned_trial = np.full((ch, f, total_samples), np.nan, dtype=np.float32)

        evt_idx = int(round(evt_ms * fs / 1000))
        before_samples = int(before * fs / 1000)
        after_samples = int(after * fs / 1000)

        start_idx = max(0, evt_idx - before_samples)
        end_idx = min(t, evt_idx + after_samples)

        buffer_start = max(0, before_samples - evt_idx)
        buffer_end = buffer_start + (end_idx - start_idx)

        # Copy valid range
        aligned_trial[:, :, buffer_start:buffer_end] = trial_wav[:, :, start_idx:end_idx]

        aligned_data.append(aligned_trial)

    return aligned_data, time_vector
# -----------------------------------------------------------------------------------------------------------------------

def compute_psd(data, fs=1000, freq_range=(0, 90), multi_processing=-2, nperseg=2000):
    """
    Compute the Power Spectral Density (PSD) of LFP data, ensuring float32 dtype.

    Args:
        data (np.ndarray): 1D, 2D, or 3D array where the last dimension is the data points.
        fs (int, optional): Sampling frequency in Hz (default is 1000 Hz).
        freq_range (tuple, optional): Frequency range to compute PSD (default is (0, 90) Hz).
        multi_processing (int, optional): -1 for all cores, -2 (default) for all but one, 0 for no multiprocessing.
        nperseg (int, optional): Number of points per segment for PSD computation.

    Returns:
        np.ndarray: PSD with the same shape as input but in the frequency domain (float32).
        np.ndarray: 1D array of frequency points (float32).
    """
    # Ensure input is float32 **ONCE**
    if not isinstance(data, np.ndarray) or data.dtype != np.float32:
        data = np.asarray(data, dtype=np.float32)

    # Determine input dimensions
    if data.ndim == 3:
        nb_trials, nb_channels, data_points = data.shape
        data = data.reshape(nb_trials * nb_channels, data_points)  # Flatten for processing
        mode_3D = True
    elif data.ndim == 2:
        nb_channels, data_points = data.shape
        mode_3D = False
    elif data.ndim == 1:
        data_points = data.shape[0]
        mode_3D = False
    else:
        raise TypeError("The input data is not properly shaped. Check your input.")

    # Compute frequency range
    f = np.fft.rfftfreq(data_points, d=1 / fs).astype(np.float32)  # ✅ Ensure float32
    if freq_range is not None:
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[freq_mask]  # ✅ Filtered frequency points

    # Prepare arguments for parallel processing
    args = [(trial, fs, freq_range, nperseg) for trial in data]

    # Compute PSD using multiprocessing if enabled
    if multi_processing == 0:
        results = [process_psd(arg) for arg in tqdm(args, total=len(args), desc="PSD computation")]
    else:
        num_workers = max(1, os.cpu_count() - 1 if multi_processing == -2 else os.cpu_count())
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_psd, args),
                                total=len(args), desc="PSD computation"))

    results = np.array(results, dtype=np.float32)  # ✅ Ensure final output is float32

    # Reshape back to 3D if necessary
    if mode_3D:
        results = results.reshape(nb_trials, nb_channels, -1)

    return results, f


# -----------------------------------------------------------------------------------------------------------------------

def compute_wavelet(data, fs=1000, wavelet='cmor1.5-1.0', multi_processing=-2, freq_range=None, freq_res=None,
                    time_unit='ms', time_window=None):
    """
    Compute Continuous Wavelet Transform (CWT) of LFP data, ensuring float32 while minimizing unnecessary conversions.

    Args:
        data (np.ndarray): 1D, 2D, or 3D array where the last dimension is the data points.
        fs (int, optional): Sampling frequency in Hz (default is 1000 Hz).
        wavelet (str, optional): Wavelet type (default is 'cmor1.5-1.0').
        multi_processing (int, optional): -1 for all cores, -2 (default) for all but one, 0 for no multiprocessing.
        freq_range (tuple, optional): Frequency range (default is (1, 120) Hz).
        freq_res (float, optional): Frequency resolution (default is 1 Hz).
        time_unit (str, optional): Time unit ('ms', 's', etc.).
        time_window (tuple, optional): Time window for the computation.

    Returns:
        np.ndarray: Wavelet-transformed data (float32).
        np.ndarray: Frequencies (float32).
        np.ndarray: Time points (float32).
    """

    def make_even(n):
        """Ensure the number is even."""
        n = int(n)
        return n if n % 2 == 0 else n + 1

    # Ensure input is float32 **ONCE**
    data = np.asarray(data, dtype=np.float32)

    # Set default frequency range and resolution
    freq_range = freq_range or (1, 120)
    freq_res = freq_res or 1

    # Compute frequency and scale values
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]),
                        make_even((freq_range[1] - freq_range[0]) / freq_res)).astype(np.float32)

    scales = (1 / (pywt.scale2frequency(wavelet, 1) * freqs) * fs).astype(np.float32)

    # Determine data dimensionality
    mode_3D = data.ndim == 3
    if mode_3D:
        nb_trials, nb_channels, data_points = data.shape
        data = data.reshape(nb_trials * nb_channels, data_points)  # Flatten for processing

    # Compute frequency and time vectors
    frequencies = (pywt.scale2frequency(wavelet, scales) * fs).astype(np.float32)
    times = compute_time_points(data.shape[-1], fs=fs, time_unit=time_unit, time_window=time_window)

    # Prepare arguments for parallel processing
    args = [(row, fs, wavelet, scales) for row in data]

    # Compute Wavelet Transform using multiprocessing if enabled
    if multi_processing == 0:
        results = [process_wavelet(arg) for arg in tqdm(args, total=len(args), desc="Wavelet computation")]
    else:
        num_workers = max(1, os.cpu_count() - 1 if multi_processing == -2 else os.cpu_count())
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_wavelet, args),
                                total=len(args), desc="Wavelet computation"))

    results = np.array(results, dtype=np.float32)

    # Reshape back to 3D if necessary
    if mode_3D:
        results = results.reshape(nb_trials, nb_channels, *results.shape[1:])

    return results, frequencies, times.astype(np.float32)


# -----------------------------------------------------------------------------------------------------------------------

def remove_common_artifact(data, mode='centered'):
    """
    Removes common noise/artifact using spatial referencing.

    Parameters:
    - data (np.ndarray):
        - 1D (Time): Returns unchanged.
        - 2D (Channels x Time): Applies chosen spatial referencing.
        - 3D (Trials x Channels x Time): Applies per trial.
    - mode (str): Type of referencing.
        - 'centered' (default): Subtract mean of previous and next channels.
        - 'bipolar_forward': Subtract next channel (top-down).
        - 'bipolar_backward': Subtract previous channel (bottom-up).

    Returns:
    - np.ndarray: Processed data with reduced channels (if applicable), dtype float32.
    """

    # Ensure input is float32 once
    data = np.asarray(data, dtype=np.float32)

    # Validate mode
    valid_modes = ['centered', 'bipolar_forward', 'bipolar_backward']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")

    # If 1D, return as-is
    if data.ndim == 1:
        return data

    # 2D: Channels x Time
    elif data.ndim == 2:
        num_channels = data.shape[0]

        if num_channels < 2:
            raise ValueError("Need at least 2 channels for referencing.")

        if mode == 'centered':
            if num_channels < 3:
                warnings.warn("Centered mode needs at least 3 channels. Returning empty array.")
                return np.empty((0, data.shape[1]), dtype=np.float32)
            return data[1:-1] - (data[:-2] + data[2:]) / 2

        elif mode == 'bipolar_forward':
            return data[:-1] - data[1:]

        elif mode == 'bipolar_backward':
            return data[1:] - data[:-1]

    # 3D: Trials x Channels x Time
    elif data.ndim == 3:
        num_channels = data.shape[1]

        if num_channels < 2:
            raise ValueError("Need at least 2 channels for referencing.")

        if mode == 'centered':
            if num_channels < 3:
                warnings.warn("Centered mode needs at least 3 channels. Returning empty array.")
                return np.empty((data.shape[0], 0, data.shape[2]), dtype=np.float32)
            return data[:, 1:-1, :] - (data[:, :-2, :] + data[:, 2:, :]) / 2

        elif mode == 'bipolar_forward':
            return data[:, :-1, :] - data[:, 1:, :]

        elif mode == 'bipolar_backward':
            return data[:, 1:, :] - data[:, :-1, :]

    else:
        raise ValueError("Input data must be 1D, 2D, or 3D.")


# -----------------------------------------------------------------------------------------------------------------------

def compute_amplitude_modulation(signal, smoothing_sigma=5):
    """
    Compute the amplitude modulation (AM) using the Hilbert Transform
    and apply Gaussian smoothing, ensuring float32 dtype.

    Parameters:
    - signal (np.ndarray): 1D array of raw signal data.
    - smoothing_sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
    - smoothed_am (np.ndarray): Smoothed amplitude modulation signal (float32).
    """
    # Ensure input is float32 **ONCE**
    signal = np.asarray(signal, dtype=np.float32)

    # Handle edge cases
    if np.isnan(signal).all():
        print("Warning: The input signal contains only NaNs!")
        return np.full_like(signal, np.nan)

    if np.all(signal == 0):
        print("Warning: The input signal contains only zeros!")
        return np.full_like(signal, np.nan)

    # Compute analytic signal and amplitude envelope
    amplitude_envelope = np.abs(hilbert(signal, dtype=np.complex64))  # ✅ Ensures correct dtype

    # Check for invalid amplitude envelope
    if np.isnan(amplitude_envelope).all():
        print("Error: Hilbert Transform returned only NaNs!")
        return np.full_like(signal, np.nan)

    if np.all(amplitude_envelope == 0):
        print("Error: The amplitude envelope is all zeros! Something went wrong.")
        return np.full_like(signal, np.nan)

    # Apply Gaussian smoothing
    smoothed_am = gaussian_filter1d(amplitude_envelope, sigma=smoothing_sigma)

    return smoothed_am


# -----------------------------------------------------------------------------------------------------------------------

def peak_based_envelope(signal, smoothing_sigma=5):
    """
    Extracts and smooths the amplitude envelope using peak detection and interpolation,
    ensuring float32 dtype.

    Parameters:
    - signal (np.ndarray): Input array (1D, 2D, or 3D+) where the last dimension is time.
    - smoothing_sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
    - smoothed_envelope (np.ndarray): Smoothed envelope, same shape as input signal.
    """
    # Ensure input is float32 **ONCE**
    signal = np.asarray(signal, dtype=np.float32)
    shape = signal.shape
    reshaped_signal = signal.reshape(-1, shape[-1])  # Flatten all but the last axis

    # Initialize envelope
    envelope = np.zeros_like(reshaped_signal)

    # Process each channel/trial independently
    for i in range(reshaped_signal.shape[0]):
        sig = reshaped_signal[i]
        peaks, _ = find_peaks(sig)  # Detect peaks

        if len(peaks) < 2:  # Edge case: Too few peaks for interpolation
            envelope[i] = sig  # Default to original signal
        else:
            # Interpolate envelope
            envelope[i] = np.interp(
                np.arange(len(sig)),  # Interpolation index
                peaks,
                sig[peaks]
            )

            # Apply Gaussian smoothing
            envelope[i] = gaussian_filter1d(envelope[i], sigma=smoothing_sigma)

    # Reshape back to original input dimensions
    return envelope.reshape(shape)


# -----------------------------------------------------------------------------------------------------------------------

def match_data_points(data1, data2):
    """
    Resample data1 to have the same number of data points as data2 while maintaining float32.

    Parameters:
    - data1 (ndarray): The data to be resampled. Can be 1D, 2D, or 3D.
    - data2 (ndarray): The reference data that defines the target number of points.

    Returns:
    - ndarray: Resampled data1 with the same number of data points as data2 (float32).
    """
    # Ensure inputs are float32 **ONCE**
    data1 = np.asarray(data1, dtype=np.float32)
    num_points_old = data1.shape[-1]
    num_points_new = data2.shape[-1]

    # Generate index mapping from old to new
    old_indices = np.linspace(0, 1, num_points_old)
    new_indices = np.linspace(0, 1, num_points_new)

    # Handle different dimensions efficiently
    if data1.ndim == 1:
        return np.interp(new_indices, old_indices, data1, left=data1[0], right=data1[-1])

    # Resample along the last dimension
    resampled_data = np.apply_along_axis(
        lambda arr: np.interp(new_indices, old_indices, arr, left=arr[0], right=arr[-1]),
        axis=-1,
        arr=data1
    )

    return resampled_data


# -----------------------------------------------------------------------------------------------------------------------

def concatenate_trials_without_nan(data):
    """
    Convert trial-based data into continuous session data while removing NaNs and preserving float32 dtype.

    Supports:
    - 3D input: (trials, channels, time points)
    - 2D input: (trials, time points) → treated as (trials, 1 channel, time)

    Parameters:
    - data (np.ndarray): Input array of shape (trials, channels, time) or (trials, time).

    Returns:
    - np.ndarray: Output of shape (channels, concatenated time points) in float32.
    """
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 2:
        # (trials, time) → reshape to (trials, 1, time)
        data = data[:, np.newaxis, :]
    elif data.ndim != 3:
        raise ValueError("Input data must be either 2D (trials × time) or 3D (trials × channels × time).")

    # Concatenate across trials for each channel, removing NaNs
    concatenated_data = [
        data[:, ch, :].ravel()[~np.isnan(data[:, ch, :].ravel())]
        for ch in range(data.shape[1])
    ]

    return np.array(concatenated_data, dtype=np.float32)


# -----------------------------------------------------------------------------------------------------------------------

def reconstruct_trials_from_time_info(continuous_data, trial_start_times, trial_durations, sampling_rate):
    """
    Reconstruct trials from continuous LFP or wavelet data using start times and durations.

    Parameters:
        continuous_data: np.ndarray
            - LFP shape: (n_channels, total_time)
            - Wavelet shape: (n_channels, n_freqs, total_time)
        trial_start_times: array-like, in milliseconds
        trial_durations: array-like, in milliseconds
        sampling_rate: int, in Hz

    Returns:
        - If LFP: np.ndarray of shape (n_trials, n_channels, max_trial_length)
        - If wavelet: list of arrays, each of shape (n_channels, n_freqs, trial_len)
    """
    continuous_data = np.asarray(continuous_data, dtype=np.float32)
    trial_start_times = np.asarray(trial_start_times, dtype=np.float32)
    trial_durations = np.asarray(trial_durations, dtype=np.float32)

    # Ensure input is 2D (channels x time)
    continuous_data = np.atleast_2d(continuous_data).astype(np.float32)

    is_wavelet = continuous_data.ndim == 3

    if is_wavelet:
        n_channels, n_freqs, total_time_points = continuous_data.shape
    else:
        n_channels, total_time_points = continuous_data.shape

    num_trials = len(trial_start_times)

    # Convert ms to sample indices
    trial_start_indices = np.round(trial_start_times * sampling_rate / 1000).astype(int)
    trial_durations_samples = np.round(trial_durations * sampling_rate / 1000).astype(int)
    trial_end_indices = trial_start_indices + trial_durations_samples
    max_trial_length = np.max(trial_durations_samples)

    if is_wavelet:
        reconstructed_data = []
        for i in range(num_trials):
            start_idx, end_idx = trial_start_indices[i], trial_end_indices[i]
            trial = continuous_data[:, :, start_idx:end_idx]
            reconstructed_data.append(trial.astype(np.float32))
    else:
        reconstructed_data = np.full(
            (num_trials, n_channels, max_trial_length), np.nan, dtype=np.float32
        )
        for i in range(num_trials):
            start_idx, end_idx = trial_start_indices[i], trial_end_indices[i]
            trial_len = end_idx - start_idx
            reconstructed_data[i, :, :trial_len] = continuous_data[:, start_idx:end_idx]

    return reconstructed_data



# -----------------------------------------------------------------------------------------------------------------------

def bandpass_filter(data, band, fs=1000, filter_type="iir", order=4, multi_processing=-2):
    """
    Apply a bandpass filter to the input data using parallel processing, ensuring float32 dtype.

    Parameters:
    - data (np.ndarray): Input data where the last dimension represents time points.
    - band (tuple): Frequency range for bandpass filtering (low, high).
    - fs (int, optional): Sampling rate in Hz (default: 1000).
    - filter_type (str, optional): Filter type ('iir' or 'fir', default: 'iir').
    - order (int, optional): Filter order (default: 4).
    - multi_processing (int, optional): Number of CPU cores to use (-2 = all but one, -1 = all, 0 = no multiprocessing).

    Returns:
    - np.ndarray: Filtered data with the same shape as input (float32).
    """

    # Ensure input is float32 **ONCE**
    data = np.asarray(data, dtype=np.float32)

    # Validate and set number of workers
    max_cores = os.cpu_count()
    if multi_processing == -2:
        num_workers = max(1, max_cores - 1)
    elif multi_processing == -1:
        num_workers = max_cores
    elif multi_processing == 0:
        num_workers = 1  # No multiprocessing
    elif isinstance(multi_processing, int) and 1 <= multi_processing <= max_cores:
        num_workers = multi_processing
    else:
        raise ValueError(f"Invalid number of workers: {multi_processing}. Must be between 1 and {max_cores}.")

    # Handle different data dimensions
    if data.ndim == 1:
        return apply_filter(data, band, fs, filter_type, order)  # 1D data, no parallel processing needed
    elif data.ndim == 2:
        flattened_data = data  # 2D case: (channels, time points)
    elif data.ndim == 3:
        trials, channels, time_points = data.shape
        flattened_data = data.reshape(trials * channels, time_points)  # Flatten trials to 2D
    else:
        raise ValueError("Data must be 1D, 2D, or 3D. Higher dimensions are not supported.")

    # Prepare arguments for parallel processing
    args = [(trial, band, fs, filter_type, order) for trial in flattened_data]  # ✅ No need to re-cast to float32

    # If multiprocessing is disabled, process sequentially
    if multi_processing == 0:
        print("Processing sequentially (no multiprocessing)...")
        filtered_results = [apply_filter(*arg) for arg in tqdm(args, desc="Filtering Progress")]
    else:
        # Use parallel processing
        print(f"Applying {filter_type.upper()} filter with {num_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            filtered_results = list(tqdm(executor.map(apply_filter, *zip(*args)),
                                         total=len(flattened_data), desc="Filtering Progress"))

    # Convert results back to a NumPy array
    filtered_data = np.array(filtered_results, dtype=np.float32)  # ✅ Ensure float32

    # Reshape back to 3D if needed
    if data.ndim == 3:
        filtered_data = filtered_data.reshape(trials, channels, time_points)

    return filtered_data


# -----------------------------------------------------------------------------------------------------------------------


def compute_sliding_psd(data, fs=1000, freq_range=(1, 120), window_size=200, step_size=50, nperseg=128,
                        multi_processing=-2):
    """
    Compute a time-frequency representation using PSD with a sliding window, ensuring float32 dtype.

    Args:
        data (np.ndarray): 1D, 2D, or 3D array where the last dimension is data points.
        fs (int, optional): Sampling frequency in Hz (default is 1000 Hz).
        freq_range (tuple, optional): Frequency range (default is (1, 120) Hz).
        window_size (int, optional): Window size in samples.
        step_size (int, optional): Step size in samples.
        nperseg (int, optional): Number of points per segment for PSD.
        multi_processing (int, optional): -1 for all cores, -2 (default) for all but one, 0 for no multiprocessing.

    Returns:
        np.ndarray: Time-frequency representation (similar to wavelet coefficients).
        np.ndarray: Frequency points.
        np.ndarray: Time points.
    """

    # Ensure input is float32 **ONCE**
    data = np.asarray(data, dtype=np.float32)

    # Handle different input shapes
    if data.ndim == 3:
        nb_trials, nb_channels, data_points = data.shape
        data = data.reshape(nb_trials * nb_channels, data_points)  # Flatten trials & channels for parallel processing
        mode_3D = True
        sample_data = data[0, :int(window_size*(fs/1000))]
    elif data.ndim == 2:
        nb_channels, data_points = data.shape
        mode_3D = False
        sample_data = data[0, :int(window_size * (fs / 1000))]
    elif data.ndim == 1:
        data_points = data.shape[0]
        mode_3D = False
        sample_data = data[:int(window_size * (fs / 1000))]
    else:
        raise TypeError("The input data must be 1D, 2D, or 3D.")

    # Compute time points
    time_points = np.arange(0, data_points - window_size, step_size, dtype=np.float32) / fs
    f, _ = welch(sample_data, fs=fs, nperseg=min(len(sample_data), nperseg))
    del sample_data
    # Prepare arguments for multiprocessing
    args = [(trial, fs, freq_range, window_size, step_size, nperseg) for trial in data]

    # Compute PSD with or without multiprocessing
    if multi_processing == 0:
        # print("Processing sequentially (no multiprocessing)...")
        results = [process_sliding_psd(arg) for arg in tqdm(args, total=len(args), desc="Sliding PSD computation")]
    else:
        num_workers = max(1, os.cpu_count() - 1 if multi_processing == -2 else os.cpu_count())
        # print(f"Applying Sliding PSD with {num_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_sliding_psd, args),
                                total=len(args), desc="Sliding PSD computation"))

    # Convert results to array
    results = np.array(results, dtype=np.float32)

    # Reshape back to 3D if necessary
    if mode_3D:
        results = results.reshape(nb_trials, nb_channels, *results.shape[1:])

    return results, f, time_points


# -----------------------------------------------------------------------------------------------------------------------

def remove_artifacts_lfp(data, threshold=5.0):
    """
    Removes artifacts from LFP data by applying z-score thresholding and interpolating missing values,
    ensuring float32 dtype. Supports input shapes: (tr, ch, dp), (ch, pt*), (dp,), (dp*,).

    Parameters:
    - data (np.ndarray): LFP data array with shape (tr, ch, dp) or (ch, pt*) or (dp,) or (dp*,).
    - threshold (float): Z-score threshold for artifact detection (default: 5.0).

    Returns:
    - np.ndarray: Cleaned LFP data with the same shape as input (float32).
    """
    # Ensure input is float32 **ONCE**
    data = np.asarray(data, dtype=np.float32)

    original_shape = data.shape
    flattened_data = data.reshape(-1, original_shape[-1]) if data.ndim > 1 else data

    # Compute Z-score
    mean = np.mean(flattened_data, axis=-1, keepdims=True)
    std = np.std(flattened_data, axis=-1, keepdims=True)
    z_scored = (flattened_data - mean) / (std + 1e-8)

    # Identify artifacts
    mask = np.abs(z_scored) > threshold
    cleaned_data = np.where(mask, np.nan, flattened_data)

    # Interpolation function
    def interpolate_nans(arr):
        """ Interpolate NaN values in a 1D array using linear interpolation. """
        nans = np.isnan(arr)
        if np.all(nans):  # If all values are NaN, return zeros
            return np.zeros_like(arr)
        valid_x = np.where(~nans)[0]
        valid_y = arr[~nans]
        interpolator = interp1d(valid_x, valid_y, kind='linear', bounds_error=False, fill_value='extrapolate')
        arr[nans] = interpolator(np.where(nans)[0])
        return arr

    # Apply interpolation along the last axis (time axis)
    cleaned_data = np.apply_along_axis(interpolate_nans, axis=-1, arr=cleaned_data)

    # Reshape back to the original format
    return cleaned_data.reshape(original_shape)


# -----------------------------------------------------------------------------------------------------------------------

def band_remove_filter(data, band, fs=1000, filter_type="iir", order=4, multiprocessing=-2):
    """
    Apply a band-stop filter to remove a frequency band from the input signal, ensuring float32.

    Parameters:
    - data (ndarray): Input array (any number of dimensions, last axis is time).
    - band (tuple): (low_freq, high_freq) range of frequencies to remove.
    - fs (int): Sampling frequency in Hz. Default: 1000 Hz.
    - filter_type (str): 'iir' (default) or 'fir'.
    - order (int): Filter order. Default: 4.
    - multiprocessing (int): Number of CPU cores:
        - -2 (default): Use all cores minus 1.
        - -1: Use all available CPU cores.
        -  0: Run sequentially (no multiprocessing).
        - Any positive integer: Number of CPU cores to use.

    Returns:
    - filtered_data (ndarray): Data after band removal (float32).
    """

    # Ensure input is float32 **ONCE**
    data = np.asarray(data, dtype=np.float32)

    # Determine the number of workers
    max_cores = os.cpu_count()
    if multiprocessing == -1:
        num_workers = max_cores
    elif multiprocessing == -2:
        num_workers = max(1, max_cores - 1)
    elif multiprocessing == 0:
        num_workers = 0  # No multiprocessing
    elif isinstance(multiprocessing, int) and 1 <= multiprocessing <= max_cores:
        num_workers = multiprocessing
    else:
        raise ValueError(f"Invalid number of workers: {multiprocessing}. Must be between 1 and {max_cores}.")

    # Handle input shape
    original_shape = data.shape
    num_samples = original_shape[-1]

    if data.ndim == 3:
        num_trials, num_channels = original_shape[:2]
        reshaped_data = data.reshape(-1, num_samples)  # Flatten trials & channels
        mode_3D = True
    elif data.ndim == 2:
        reshaped_data = data  # 2D case (channels, time)
        mode_3D = False
    elif data.ndim == 1:
        reshaped_data = data  # 1D case
        mode_3D = False
    else:
        raise TypeError("Input data must be 1D, 2D, or 3D.")

    # Prepare arguments for parallel processing
    args = [(x, fs, band, filter_type, order) for x in reshaped_data]

    # Parallel processing
    if num_workers > 0 and reshaped_data.ndim > 1:
        print(f"Applying band-stop filter with {num_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            filtered_results = list(tqdm(executor.map(apply_bandstop_filter, *zip(*args)),
                                         total=len(args), desc="Filtering Progress"))
    else:
        # Run sequentially
        print("Processing sequentially (no multiprocessing)...")
        filtered_results = [apply_bandstop_filter(*arg) for arg in
                            tqdm(args, total=len(args), desc="Filtering Progress")]

    # Convert results back to NumPy array
    filtered_data = np.array(filtered_results, dtype=np.float32)

    # Reshape back to original
    if mode_3D:
        filtered_data = filtered_data.reshape(num_trials, num_channels, num_samples)

    return filtered_data


# -----------------------------------------------------------------------------------------------------------------------

def compute_trial_condition_rate(
        start_trials: np.ndarray,
        trial_durations: np.ndarray,
        trial_condition_mask: np.ndarray,
        window_length: float = 60000.0,  # in ms
        window_type: str = "gaussian",  # Default changed to "hanning"
        output_fs: float = 1.0,  # Hz
        data_dur: float = None  # Total duration of data in ms (optional)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time-resolved rate of trials that match a given condition (in ms).

    Parameters:
    - start_trials (np.ndarray): Trial start times in ms.
    - trial_durations (np.ndarray): Trial durations in ms.
    - trial_condition_mask (np.ndarray): Boolean mask (0 or 1) for matching trials.
    - window_length (float): Smoothing window length in ms (default: 60000 ms).
    - window_type (str): "hanning", "gaussian", or "square".
    - output_fs (float): Output sampling frequency in Hz (default: 1 Hz).
    - data_dur (float or None): Total duration of data in ms (optional).

    Returns:
    - time_points (np.ndarray): Time points in ms.
    - trial_condition_rate (np.ndarray): Smoothed rate signal.
    """

    # Filter matching trials
    matching_trials = trial_condition_mask.astype(bool)
    start_trials = start_trials[matching_trials].astype(np.float32)
    trial_durations = trial_durations[matching_trials].astype(np.float32)

    # Define time resolution based on output_fs
    resolution = int(1000 / output_fs)

    # Handle case where no trials match
    if len(start_trials) == 0:
        default_dur = data_dur if data_dur is not None else 10000  # 10 sec default
        time_points = np.arange(0, default_dur, resolution, dtype=np.float32)
        return time_points, np.zeros_like(time_points, dtype=np.float32)

    # Define time axis
    time_min = 0
    time_max = data_dur if data_dur is not None else np.max(start_trials + trial_durations)
    time_points = np.arange(time_min, time_max + resolution, resolution, dtype=np.float32)

    # Build binary trial presence vector
    trial_events = np.zeros_like(time_points, dtype=np.float32)
    for t_start, t_duration in zip(start_trials, trial_durations):
        trial_mask = (time_points >= t_start) & (time_points < t_start + t_duration)
        trial_events[trial_mask] = 1.0

    # Create smoothing window
    window_size = int(window_length / resolution)
    window_size = max(window_size, 1)

    if window_type == "gaussian":
        window = gaussian(window_size, std=window_size / 6)
    elif window_type == "square":
        window = np.ones(window_size, dtype=np.float32)
    elif window_type == "hanning":
        window = windows.hann(window_size, sym=True)
    else:
        raise ValueError("Unsupported window type. Use 'hanning', 'gaussian', or 'square'.")

    window /= np.sum(window)

    # Convolve to get smoothed rate
    trial_condition_rate = convolve(trial_events, window, mode='same', method='auto')

    return time_points, trial_condition_rate.astype(np.float32)

# -----------------------------------------------------------------------------------------------------------------------
def compute_superlet(data, fs=1000, base_cycle=3, min_order=1, max_order=10,
                     mode='mul', multi_processing=-2, freq_range=None, freq_res=None,
                     time_unit='ms', time_window=None):
    """
    Compute Superlet Transform of LFP data using multiprocessing and return power.

    Args:
        data (np.ndarray): 1D, 2D, or 3D array where the last dimension is time.
        fs (int, optional): Sampling frequency (Hz). Defaults to 1000.
        base_cycle (int): Number of cycles at base order.
        min_order (int): Minimum superlet order.
        max_order (int): Maximum superlet order.
        mode (str): 'mul' or 'add' for superlet scaling.
        multi_processing (int): -1 for all cores, -2 for all but one, 0 for no multiprocessing.
        freq_range (tuple, optional): (low, high) in Hz.
        freq_res (float, optional): Frequency resolution in Hz.
        time_unit (str): Time unit for returned time array.
        time_window (tuple, optional): Time range.

    Returns:
        np.ndarray: Superlet power (float32).
        np.ndarray: Frequencies.
        np.ndarray: Time points.
    """
    # Ensure float32 once
    data = np.asarray(data, dtype=np.float32)

    # Frequency setup
    freq_range = freq_range or (1, 120)
    freq_res = freq_res or 1
    freqs = np.linspace(freq_range[0], freq_range[1],
                        int((freq_range[1] - freq_range[0]) / freq_res)).astype(np.float32)

    # Reshape if 3D
    mode_3D = data.ndim == 3
    if mode_3D:
        nb_trials, nb_channels, data_points = data.shape
        data = data.reshape(nb_trials * nb_channels, data_points)

    # Time vector
    times = compute_time_points(data.shape[-1], fs=fs, time_unit=time_unit, time_window=time_window)

    args = [(row, freqs, fs, base_cycle, min_order, max_order, mode) for row in data]

    # Compute
    if multi_processing == 0:
        results = [process_superlet(arg) for arg in tqdm(args, total=len(args), desc="Superlet computation")]
    else:
        num_workers = max(1, os.cpu_count() - 1 if multi_processing == -2 else os.cpu_count())
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_superlet, args),
                                total=len(args), desc="Superlet computation"))

    results = np.array(results, dtype=np.float32)

    # Restore shape
    if mode_3D:
        results = results.reshape(nb_trials, nb_channels, *results.shape[1:])

    return results, freqs.astype(np.float32), times.astype(np.float32)
# -----------------------------------------------------------------------------------------------------------------------

def align_wavelet_around_event(data, event_times, before, after, fs=1000, nan_padding=True):
    """
    Aligns 4D wavelet data (trials, channels, freqs, time) around event times.

    Args:
        data (np.ndarray): 4D array with shape (trials, channels, freqs, time)
        event_times (np.ndarray): 1D array of event times in milliseconds (one per trial)
        before (int): Time before event (ms)
        after (int): Time after event (ms)
        fs (int): Sampling frequency in Hz (default 1000)
        nan_padding (bool): Whether to pad with NaNs if window exceeds bounds (default: True)

    Returns:
        aligned_data (np.ndarray): Shape (trials, channels, freqs, aligned_time)
        time_vector (np.ndarray): Time vector centered around event (float32)
    """
    assert data.ndim == 4, "Input data must be 4D: (trials, channels, freqs, time)"
    n_trials, n_channels, n_freqs, n_timepoints = data.shape

    event_times = np.asarray(event_times)
    before_samples = int(before * fs / 1000)
    after_samples = int(after * fs / 1000)
    total_samples = before_samples + after_samples

    aligned_data = np.full(
        (n_trials, n_channels, n_freqs, total_samples),
        np.nan,
        dtype=np.float32
    )

    time_vector = np.linspace(-before, after, total_samples, dtype=np.float32)

    for trial_idx, event_time in enumerate(event_times):
        event_idx = int(round(event_time * fs / 1000))
        start_idx = max(0, event_idx - before_samples)
        end_idx = min(n_timepoints, event_idx + after_samples)

        buffer_start = max(0, before_samples - event_idx)
        buffer_end = buffer_start + (end_idx - start_idx)

        aligned_data[trial_idx, :, :, buffer_start:buffer_end] = data[trial_idx, :, :, start_idx:end_idx]

    return aligned_data, time_vector