from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BIDS_ROOT = r"C:\Users\alexander.lepauvre\Documents\GitHub\cogitate-videogame-ieeg-bids\data\bids"
SOURCE_ROOT = r"C:\Users\alexander.lepauvre\Documents\GitHub\cogitate-videogame-ieeg-bids\data\source"
PD_ROW_ENTRIES = "TRIGGER_MANAGER_PHOTODIODE"
TRIGGERED_EVENTS = ["GameStimulusOn", "GameFillerOn", "GameBlankOn", "GameProbeOn", "ReplayTaskOn",
                    "ReplayNonTaskOn", "ReplayFillerOn", "AnimationPeakEnd"]
BEH_LOG_FN_TEMPLATE = "{}1_{}_X_Trigger.csv"
IEEG_FN_TEMPLATE = "{}_ECoG_V2_{}.EDF"

LEVELS_ORDER = ("VGR0", "VGR1", "VGR2", "ReplayR1")
LOGS_ORDER = ("0", "1", "2", "A")


def list_files(
    root: Union[str, Path],
    pattern: str,
) -> List[Path]:
    """
    Recursively list files under `root` that match a glob `pattern`.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory to search (recursively).
    pattern : str
        Glob pattern relative to `root`. Examples:
        - f\"{subject}_ECoG_V2_*.EDF\"
        - \"**/*.EDF\" (discouraged if you can be more specific)

    Returns
    -------
    list of pathlib.Path
        Sorted list of matching file paths.

    Raises
    ------
    FileNotFoundError
        If `root` does not exist or is not a directory.
    """
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    files = sorted(root.rglob(pattern))
    return files


def sort_files(files, order):
    """
    Sort files by stage/run tokens appearing in either their folder path
    or filename. Works for both iEEG files (subfolders) and log files
    (same folder).

    Parameters
    ----------
    files : list[pathlib.Path]
        List of files to sort.
    order : list[str]
        Desired order of tokens (e.g., ["VGR0", "VGR1", "VGR2", "ReplayR1"]
        for iEEG, or ["0", "1", "2", "A"] for log files).

    Returns
    -------
    list[pathlib.Path]
        Files sorted according to the first matching token. Files without
        any matching token are placed at the end.
    """
    order_lower = [tok.lower() for tok in order]

    def file_index(path):
        hay = path.as_posix().lower()
        for i, tok in enumerate(order_lower):
            if f"{tok.lower()}" in hay:
                return i
        return len(order)  # unknowns at the end

    return sorted(files, key=lambda p: (file_index(p), p.name))


def load_ieeg_signal(
    ieeg_files: Sequence[Union[str, Path]],
    levels_order: Optional[list] = None,
) -> mne.io.BaseRaw:
    """
    Load iEEG EDF files and concatenate them in chronological order.

    Files are sorted primarily by `meas_date` (from EDF header), with
    filename as a deterministic tiebreaker when dates are missing or equal.

    Parameters
    ----------
    ieeg_files : sequence of str or pathlib.Path
        Paths to EDF files to load and concatenate.

    Returns
    -------
    mne.io.BaseRaw
        Concatenated Raw object.

    Raises
    ------
    ValueError
        If `ieeg_files` is empty.
    IOError
        If any file cannot be read.
    """
    if not ieeg_files:
        raise ValueError("No iEEG files provided to load_ieeg_signal().")
    raws_list: List[Tuple[float, str, mne.io.BaseRaw]] = []
    # Sort the iEEG files per level:
    ieeg_files = sort_files(ieeg_files, order=levels_order)
    for f in ieeg_files:
        # Load raw:
        raws_list.append(mne.io.read_raw_edf(str(f), preload=False, verbose=False))
    return mne.concatenate_raws(raws_list, verbose=False)


def binarize_pd(
    pd_signal: np.ndarray,
    thresh: Optional[float] = None,
) -> np.ndarray:
    """
    Binarize a photodiode signal using a fixed threshold.

    If `thresh` is not provided, the midpoint between min and max is used.
    NaNs are ignored for range estimation (midpoint computed on finite data).

    Parameters
    ----------
    pd_signal : np.ndarray
        1-D array of PD samples.
    thresh : float, optional
        Threshold above which samples are considered "on". If None,
        uses (min + max) / 2 of finite values.

    Returns
    -------
    np.ndarray
        Boolean array of same shape as `pd_signal` where True indicates "on".

    Raises
    ------
    ValueError
        If `pd_signal` is empty or all values are NaN.
    """
    if pd_signal.size == 0:
        raise ValueError("pd_signal is empty.")

    pd_signal = np.asarray(pd_signal).ravel()

    finite = pd_signal[np.isfinite(pd_signal)]
    if finite.size == 0:
        raise ValueError("pd_signal contains only NaNs.")

    if thresh is None:
        thresh = (finite.min() + finite.max()) / 2.0

    binary = pd_signal > float(thresh)
    return binary


def get_pd_timestamps(
    pd_signal: np.ndarray,
    fs: float,
    edge: str = "rising",
) -> np.ndarray:
    """
    Extract timestamps (seconds) of photodiode transitions.

    By default, returns rising edges (0→1 transitions) of the binarized PD
    signal. Falling edges can be selected via `edge="falling"`.

    Parameters
    ----------
    pd_signal : np.ndarray
        1-D array of PD samples.
    fs : float
        Sampling rate in Hz.
    edge : {"rising", "falling"}, optional
        Which edges to return. Default is "rising".

    Returns
    -------
    np.ndarray
        1-D array of onset timestamps in seconds.

    Raises
    ------
    ValueError
        If `fs` <= 0 or `edge` is invalid.
    """
    if fs <= 0:
        raise ValueError(f"Sampling rate must be positive, got {fs}.")

    edge = edge.lower()
    if edge not in {"rising", "falling"}:
        raise ValueError("edge must be one of {'rising', 'falling'}.")

    pd_binary = binarize_pd(pd_signal).astype(np.int8)

    # Compute first differences with a 0 prepended to align indices.
    diffs = np.diff(pd_binary, prepend=0)

    if edge == "rising":
        idx = np.flatnonzero(diffs == 1)
    else:  # falling
        idx = np.flatnonzero(diffs == -1)

    timestamps = idx / float(fs)
    return timestamps


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def vg2bids(
    source_root: Union[str, Path],
    bids_root: Union[str, Path],
    subject: str,
    pd_channel: str = "DC1",
) -> Mapping[str, object]:
    """
    Minimal vg2bids-like routine: collect PD timestamps and basic metadata.

    This function:
      1) Finds iEEG EDF files under `<source_root>/scans` matching the template
         `IEEG_FN_TEMPLATE.format(subject, "*")`.
      2) Loads and concatenates them in chronological order.
      3) Extracts the specified PD channel.
      4) Returns PD onset timestamps (seconds), sampling rate, and file list.

    Parameters
    ----------
    source_root : str or pathlib.Path
        Root directory for raw data (expects a `scans/` subdirectory).
    bids_root : str or pathlib.Path
        Root directory for BIDS output (currently unused here but kept for
        API compatibility and future extensions).
    subject : str
        Subject identifier used to expand the EDF filename template.
    pd_channel : str, optional
        Name of the PD channel to extract (default "DC1").

    Returns
    -------
    Mapping[str, object]
        A dict containing:
        - "raw": mne.io.BaseRaw, the concatenated recording.
        - "pd_timestamps_s": np.ndarray, PD rising-edge onsets (seconds).
        - "sfreq": float, sampling rate in Hz.
        - "ieeg_files": list[str], the EDF files used.

    Raises
    ------
    FileNotFoundError
        If no matching iEEG files are found.
    KeyError
        If the requested `pd_channel` is not present in the data.
    """

    #  ==============================================================
    # Photodiode:
    scans_dir = Path(source_root, subject, "scans")
    pattern = IEEG_FN_TEMPLATE.format(subject, "*")

    ieeg_files = list_files(scans_dir, pattern)
    if not ieeg_files:
        raise FileNotFoundError(
            f"No iEEG files found for subject '{subject}' "
            f"using pattern '{pattern}' under {scans_dir}"
        )

    raw = load_ieeg_signal(ieeg_files, levels_order=LEVELS_ORDER)

    if pd_channel not in raw.ch_names:
        raise KeyError(
            f"PD channel '{pd_channel}' not found. "
            f"Available channels include e.g. {raw.ch_names[:10]}..."
        )

    raw_pd = raw.copy().pick([pd_channel]).load_data()
    fs = float(raw.info["sfreq"])
    pd_ts = get_pd_timestamps(np.squeeze(raw_pd.get_data()), fs, edge="rising")

    #  ==============================================================
    # Log files:
    logs_dir = Path(source_root, subject, "resources", "BEH", subject, "1", "AnalyzerOutput")
    pattern = BEH_LOG_FN_TEMPLATE.format(subject, "*")

    log_files = list_files(logs_dir, pattern)
    if not log_files:
        raise FileNotFoundError(
            f"No log files found for subject '{subject}' "
            f"using pattern '{pattern}' under {logs_dir}"
        )
    
    # Load log files:
    logs = pd.concat([pd.read_csv(f, sep=";") 
                      for f in sort_files(log_files, order=LOGS_ORDER)]).reset_index(drop=True)

    # Extract the triggered events (a bit messy but oh well):
    logs_clean = logs[((logs["sender"] == "TRIGGER_MANAGER_PHOTODIODE")
                            & (logs["triggerState"] == "TRIGGER_SENT")
                            & (logs["triggerEvent"] != "LevelBegin")
                            & (logs["triggerEvent"] != "LevelEnd"))
                           | ((logs["sender"] == "TRIGGER_MANAGER_PHOTODIODE")
                              & (logs["triggerState"] == "TRIGGER_INFO")
                              & ((logs["triggerEvent"] == "LevelBegin")
                                 | (logs["triggerEvent"] == "LevelEnd")))][["triggerEvent", "triggerCode", "world", "level", "timeMS"]].reset_index(drop=True)
    
    # =================================================================
    # Compare the time stamps:
    fig, ax = plt.subplots(2)
    ax[0].plot(np.diff(pd_ts), color="r", label=r"$\Delta(t_{photo})$")
    ax[0].plot(np.diff(logs_clean["timeMS"].to_numpy()*0.001), color="b", label=r"$\Delta(t_{log})$")
    ax[0].legend()
    ax[0].set_xlabel("Event #")
    ax[0].set_ylabel(r'$\Delta(t)$ (seconds)')
    ax[0].set_title("Difference in events intervals between logs and pd")
    ax[1].hist(np.diff(pd_ts) - np.diff(logs_clean["timeMS"].to_numpy()*0.001))
    ax[1].set_xlabel(r'$\Delta(t_{photo}) - \Delta(t_{log})$')
    ax[1].set_ylabel("# of events")
    ax[1].set_title("Difference between intervals of logs and photodiode")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    subjects_list = ["SF104"]
    for sub in subjects_list:
        vg2bids(SOURCE_ROOT, BIDS_ROOT, sub, pd_channel="DC1")



