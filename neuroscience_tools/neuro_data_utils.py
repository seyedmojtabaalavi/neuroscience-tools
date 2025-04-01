import pandas as pd
import pickle
import zstandard as zstd
import numpy as np
import os
try:
    # Preferred import for newer versions of scipy
    from scipy.signal.windows import gaussian
except ImportError:
    # Fallback for older versions
    from scipy.signal import gaussian
from scipy import signal
from scipy.signal import welch, hilbert, find_peaks, convolve
import pywt
from tqdm import tqdm
import warnings
import concurrent.futures
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from numba import njit

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
        taps = signal.firwin(order + 1, [low, high], pass_zero=False, fs=fs, dtype=np.float32)
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

    return np.array(psd_results, dtype=np.float32).T, f.astype(np.float32)



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
                    data["Reaction_Time"][trl] = np.float32(data["ManetteOn"][trl] - data["Distractors"][trl, nbDistractor])
                    data["CueToTarget_Time"][trl] = np.float32(data["Distractors"][trl, nbDistractor] - data["CueOn"][trl])

        return data


# -----------------------------------------------------------------------------------------------------------------------

def save_zst_file(data, filepath, compression_level=3, threads=None, dataframe=False):
    """
    Save data (dictionary, pandas DataFrame, or numpy array) to a .zst file compressed with zstandard.

    Args:
        data (dict, pd.DataFrame, or np.ndarray): The data to be saved.
        filepath (str): Path to save the .zst file.
        compression_level (int): Compression level (1-22, where 1 is fastest, 22 is maximum compression).
        threads (int, optional): Number of threads to use for compression. Defaults to max available - 1.
        dataframe (bool, optional): Whether to convert data to a pandas DataFrame before saving (default: False).
    """
    if data is None:
        raise ValueError("Input data is None. Cannot save an empty file.")

    # ✅ Ensure float32 for NumPy arrays
    if isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=np.float32)
        if dataframe:
            data = pd.DataFrame(data)

    # ✅ Ensure float32 for numeric columns in Pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        if dataframe:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].astype(np.float32)

    # ✅ Ensure float32 for NumPy arrays inside dictionaries
    elif isinstance(data, dict):
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].astype(np.float32)
        if dataframe:
            data = pd.DataFrame(data)

    else:
        raise ValueError("Input data must be a dictionary, pandas DataFrame, or numpy array.")

    # ✅ Determine optimal number of threads if not provided
    threads = threads or max(1, os.cpu_count() - 1)

    # ✅ Ensure save directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # ✅ Save with zstandard compression
    with open(filepath, 'wb') as file:
        cctx = zstd.ZstdCompressor(level=compression_level, threads=threads)
        with cctx.stream_writer(file) as writer:
            pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------------------------------------------------------------

def compute_firing_rate(data, fs=1000, window=30, method='Gaussian', overlap='full', multi_processing=-2, time_unit='ms'):
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
        firing_rates = np.array(process_trial_for_compute_firing_rate((fs, window_size, method, step, factor, Data)), dtype=np.float32)

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
    aligned_shape = (nb_trials, nb_channels, nb_units, total_samples) if is_spiking else (nb_trials, nb_channels, total_samples)
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
    Convert trial-based data (trials, channels, time points) into continuous session data (channels, time points),
    ensuring float32 dtype.

    Parameters:
    - data (np.ndarray): A 3D array of shape (trials, channels, time points).

    Returns:
    - np.ndarray: A 2D array of shape (channels, concatenated time points) in float32.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array (trials, channels, time points).")

    # Ensure input is float32 **ONCE**
    data = np.asarray(data, dtype=np.float32)

    # Remove NaN values and concatenate across trials for each channel
    concatenated_data = [data[:, ch, :].ravel()[~np.isnan(data[:, ch, :].ravel())] for ch in range(data.shape[1])]

    # Convert to a 2D array (channels, concatenated time points)
    return np.array(concatenated_data, dtype=np.float32)



# -----------------------------------------------------------------------------------------------------------------------

def reconstruct_trials_from_time_info(continuous_data, trial_start_times, trial_durations, sampling_rate):
    """
    Reconstruct trials from a continuous signal using trial start times and durations, ensuring float32 dtype.

    Parameters:
    - continuous_data (ndarray): Shape (channels, total_time_points), the concatenated data.
    - trial_start_times (list or ndarray): Start time of each trial in milliseconds.
    - trial_durations (list or ndarray): Duration of each trial (relative to start) in milliseconds.
    - sampling_rate (int): Sampling rate in Hz.

    Returns:
    - reconstructed_data (ndarray): Shape (num_trials, num_channels, max_trial_length), with NaN padding (float32).
    """
    # Ensure inputs are float32 **ONCE**
    continuous_data = np.asarray(continuous_data, dtype=np.float32)
    trial_start_times = np.asarray(trial_start_times, dtype=np.float32)
    trial_durations = np.asarray(trial_durations, dtype=np.float32)

    num_channels, total_time_points = continuous_data.shape
    num_trials = len(trial_start_times)

    # Convert times to sample indices (ensure integer indices)
    trial_start_indices = np.round(trial_start_times * sampling_rate / 1000).astype(int)
    trial_durations_samples = np.round(trial_durations * sampling_rate / 1000).astype(int)
    trial_end_indices = trial_start_indices + trial_durations_samples

    # Find the max trial length (in samples) for consistent padding
    max_trial_length = np.max(trial_durations_samples)

    # Initialize output array with NaNs in float32
    reconstructed_data = np.full((num_trials, num_channels, max_trial_length), np.nan, dtype=np.float32)

    # Extract and assign data efficiently
    for i in range(num_trials):
        start_idx, end_idx = trial_start_indices[i], trial_end_indices[i]
        trial_length = end_idx - start_idx
        reconstructed_data[i, :, :trial_length] = continuous_data[:, start_idx:end_idx]

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
    elif data.ndim == 2:
        nb_channels, data_points = data.shape
        mode_3D = False
    elif data.ndim == 1:
        data_points = data.shape[0]
        mode_3D = False
    else:
        raise TypeError("The input data must be 1D, 2D, or 3D.")

    # Compute time points
    time_points = np.arange(0, data_points - window_size, step_size, dtype=np.float32) / fs

    # Prepare arguments for multiprocessing
    args = [(trial, fs, freq_range, window_size, step_size, nperseg) for trial in data]

    # Compute PSD with or without multiprocessing
    if multi_processing == 0:
        print("Processing sequentially (no multiprocessing)...")
        results = [process_sliding_psd(arg) for arg in tqdm(args, total=len(args), desc="Sliding PSD computation")]
    else:
        num_workers = max(1, os.cpu_count() - 1 if multi_processing == -2 else os.cpu_count())
        print(f"Applying Sliding PSD with {num_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_sliding_psd, args),
                                total=len(args), desc="Sliding PSD computation"))

    # Convert results to array
    results = np.array(results, dtype=np.float32)

    # Reshape back to 3D if necessary
    if mode_3D:
        results = results.reshape(nb_trials, nb_channels, *results.shape[1:])

    return results, results[0][1].astype(np.float32), time_points



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
        filtered_results = [apply_bandstop_filter(*arg) for arg in tqdm(args, total=len(args), desc="Filtering Progress")]

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
        trial_condition_mask: np.ndarray,  # ✅ Renamed for clarity
        window_length: float = 30.0,
        window_type: str = "gaussian",
        unit_time: str = "sec",
        resolution: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time-resolved rate of trials that match a given condition.

    Parameters:
    - start_trials (np.ndarray): Start times of trials in milliseconds or seconds.
    - trial_durations (np.ndarray): Durations of trials in milliseconds or seconds.
    - trial_condition_mask (np.ndarray): Boolean mask (0 or 1) indicating whether a trial matches the condition.
    - window_length (float): Length of the smoothing window in seconds (default: 30.0 sec).
    - window_type (str): Type of the smoothing window ("gaussian" or "square").
    - unit_time (str): "ms" (convert to sec) or "sec" (default).
    - resolution (float): Time step resolution in seconds (default: 0.1 sec).

    Returns:
    - time_points (np.ndarray): Time points corresponding to the computed rates.
    - trial_condition_rate (np.ndarray): Time-resolved rate of trials matching the condition.
    """
    # Convert to seconds if needed
    if unit_time == "ms":
        start_trials = start_trials.astype(np.float32) / 1000.0
        trial_durations = trial_durations.astype(np.float32) / 1000.0
        window_length /= 1000.0
        resolution /= 1000.0
    else:
        start_trials = start_trials.astype(np.float32)
        trial_durations = trial_durations.astype(np.float32)

    # Filter trials that match the condition
    matching_trials = trial_condition_mask.astype(bool)
    start_trials = start_trials[matching_trials]
    trial_durations = trial_durations[matching_trials]

    # Handle edge case: No trials match the condition
    if len(start_trials) == 0:
        time_points = np.arange(0, 1, resolution, dtype=np.float32)  # Dummy time axis
        return time_points, np.zeros_like(time_points, dtype=np.float32)

    # Define time axis
    time_min = np.min(start_trials)
    time_max = np.max(start_trials + trial_durations)
    time_points = np.arange(time_min, time_max, resolution, dtype=np.float32)

    # Create event array (vectorized approach)
    trial_events = np.zeros_like(time_points, dtype=np.float32)
    for t_start, t_duration in zip(start_trials, trial_durations):
        trial_mask = (time_points >= t_start) & (time_points < t_start + t_duration)
        trial_events += trial_mask.astype(np.float32)  # ✅ Vectorized update

    # Create smoothing window
    window_size = int(window_length / resolution)
    if window_type == "gaussian":
        window = gaussian(window_size, std=window_size / 6)
    elif window_type == "square":
        window = np.ones(window_size, dtype=np.float32)
    else:
        raise ValueError("Unsupported window type. Use 'gaussian' or 'square'.")

    window /= np.sum(window)  # Normalize

    # Convolve to smooth rate
    trial_condition_rate = convolve(trial_events, window, mode='same', method='auto')

    return time_points, trial_condition_rate.astype(np.float32)

# -----------------------------------------------------------------------------------------------------------------------

