"""Data preparation for VG Replay."""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import tempfile
import glob
import mne

from general_helper_functions.pathHelperFunctions import find_files
from Preprocessing.SubjectInfo import SubjectInfo
from data_preparation.mne_bids_converter import remove_elecname_leading_zero
from data_preparation.DataPreparationParameters import DataPreparationParameters
from data_preparation.mne_bids_converter import save_to_BIDs
import re


PD_ROW_ENTRIES = "TRIGGER_MANAGER_PHOTODIODE"
PD_RELVEVANT_TRIGGERS = ["GameStimulusOn", "GameFillerOn", "GameBlankOn", "GameProbeOn", "ReplayTaskOn",
                         "ReplayNonTaskOn", "ReplayFillerOn", "AnimationPeakEnd"]
FILLER_DURATION = 0.250
STIM_DURATION = 0.250


def find_correct_file(files_list):
    """
    This function reads the name of files and returns only those that do not have rh (right hemisphere) or lh (left
    hemisphere) in them, as in cases of bilaterly implant, there will be one file for electrodes of the one hemisphere
    another for the other hemisphere and a final one with both combined. we want only the latter
    :param files_list: list of files containing a date in them
    :return:
    """
    # Removing any file that contain the rh and lh names as we are interested in only the ones that have both:
    relevant_files_list = [
        file for file in files_list if "rh" not in file and "lh" not in file]

    return relevant_files_list[0]


def load_signal(root_path, file_naming_pattern, subject_info, file_extension=None, debug=False):
    """
    This function lists the EDF files in the path and loads them with MNE.
    NOTE: if you have several files in the directory, they will be loaded and concantenated to another!!!
    :param root_path: (string or path) Path to the root data. This function is recursive and will look for the data
    across the root directory down the structure. It will load all the files that have matching naming conventions to
    what is passed. Careful therefore not to have different files matching the naming convention in the directory that
    you don't want to load. In that case, you should give the root directory at a lower level to avoid loading spurious
    stuff.
    :param file_naming_pattern: (string) naming pattern of the file to load. String with wildcard accepted
    :param subject_info: (SubjectInfo custom object) object containing info about the specific participant running
    :param file_extension: (string) extension of the file, with the dot first!
    :param debug: true or false flag to be in debug mode or not. In debug mode, you will only load a few channels
    :return: raw: mne raw object containing the signal
    """

    # Priting info to let the user know it might take a while:
    print('-' * 40)
    print('The edf file(s) is/are being loaded, it might take a little while')

    data_files_list = find_files(
        root_path, naming_pattern=file_naming_pattern, extension=file_extension)
    if len(data_files_list) == 0:
        raise ReferenceError("No files with " + file_naming_pattern + " naming pattern were found in the root:"
                             + str(root_path))

    # Because of how the files are named, the replay ends up at the beginning of the list eventhough it was ran last.
    # To avoid misalignment issue, sorting things out now:
    replay_files = [file for file in data_files_list if "Replay" in file]
    game_files = [file for file in data_files_list if "Replay" not in file]
    data_files_list = game_files + replay_files

    # Creating a dictionary to store the raws in:
    raws = {}

    # looping through the files and loading them:
    for ind, file in enumerate(data_files_list):
        full_file = file
        print('Now loading this file ', full_file)
        if debug:
            raws[ind] = mne.io.read_raw_edf(full_file,
                                            verbose='error', preload=False)
            # Dropping all channels except the trigger channel and the trigger reference channel
            channels_to_drop = [ch for ch in raws[ind].info['ch_names'] if
                                ch != subject_info.TRIGGER_CHANNEL and ch != subject_info.TRIGGER_REF_CHANNEL]
            raws[ind].drop_channels(channels_to_drop)
            # Loading only the few channels to cut down loading time
            raws[ind].load_data()

        else:
            raws[ind] = mne.io.read_raw_edf(full_file,
                                            verbose='error', preload=True)

    # Concatenating the files:
    raw = raws[0]

    # Looping through the dict to raws to append to the raw:
    for key in raws:
        # If we are at key 0 of the dict, then this is the first and it has already been add to raw
        if key == 0:
            pass
        else:
            # But if we are at keys other than 0, we concatenate
            mne.concatenate_raws([raw, raws[key]])

    return raw


def extract_pd_signal(raw, subject_info):
    """
    This function extracts the photodiode signal in an easy to manage format:
    :param subject_info: (SubjectInfo object) custom made object containing info about the participant
    :param raw: mne raw object
    :return: pd_signal: (dict) dictionary of np arrays with "time" key being the photodiode time in seconds, "amp" being
    the amplitude of the photodiode signal
    """
    # If there is a reference trigger channel, using it to correct PD amplitude:
    if subject_info.TRIGGER_REF_CHANNEL != '':
        pd_signal = ({
            "time": np.array([raw.times]),
            "Amp": np.array([raw.get_data(picks=subject_info.TRIGGER_CHANNEL)[0]])
            - np.array([raw.get_data(picks=subject_info.TRIGGER_REF_CHANNEL)[0]])
        })
    else:  # Otherwise, just extract the photodiode channel:
        pd_signal = ({
            "time": np.array([raw.times]),
            "Amp": np.array([raw.get_data(picks=subject_info.TRIGGER_CHANNEL)[0]])
        })

    return pd_signal


def manual_adjust_threshold(subject_info, pd_signal):
    """
    This function enables the user to adjust the photodiode threshold manually, if it turns out to not be fitting this
    particular data set:
    :param pd_signal: photodiode signal
    :param subject_info: subject information object
    :return: subject_info updated
    """
    # Adjusting photodiode threshold:
    # Asking the user whether there are things they wish to change:
    pd_yes_no = input("The photodiode threshold is set to "
                      + str(subject_info.PD_THRESHOLD) + ". Would you like to modify it? [Yes or No]")

    # If the user wished to adjust the pd threshold, plotting the signal:
    if pd_yes_no == "Yes" or pd_yes_no == "yes":
        print("Close the figure to be able to enter the threshold.")
        fig = plt.figure(figsize=(8, 6))
        plt.plot(pd_signal["Amp"][0], 'g')
        plt.show()
        new_pd_threshold = input(
            "Type the photodiode threshold you wish to apply: ")
        subject_info.PD_THRESHOLD = float(new_pd_threshold)

    return subject_info


def extract_trigger_onset(pd_raw, subject_info, data_preparation_parameters):
    """
    This function detects the onset of the photodiode triggers
    :param pd_raw: dictionary of np array: time: np arraylen(datapoints) of PD timestamp, Amp: len(datapoints)
    np array of PD amplitude
    :param subject_info: object containing information specific to a subject
    :param data_preparation_parameters: (DataPreparation object) custom made object containing info about data
    preparation
    :return: PD_onsets: pandas series containing the timestamps of the detect photodiode onsets
    """

    # Binarizing the photodiode signal:
    binary_pd = np.squeeze(pd_raw.get_data() > subject_info.PD_THRESHOLD).astype(
        int)  # np array dim: len(datapoints)

    # Finding the onset of the photodiode peaks:
    pd_onsets_ind = np.where(np.diff(binary_pd) == 1)[0]

    # Make sure that you don't get false positives from noisy signal so that the same signal triggers the threshold
    # twice one photodiode signal is 3 ref rates ms and there are srate samples per second
    pd_onsets_ind_no_fp = pd_onsets_ind
    minimum_ind_diff = int(
        (data_preparation_parameters.ref_rate_ms * 4.5 * pd_raw.info["sfreq"]) / 1000)
    indices_to_remove = []
    for i in range(1, len(pd_onsets_ind)):
        onset_ind = pd_onsets_ind[i]
        prev_onset_ind = pd_onsets_ind[i - 1]
        if (onset_ind - prev_onset_ind) < minimum_ind_diff:
            indices_to_remove.append(i)

    pd_onsets_ind_no_fp = np.delete(pd_onsets_ind_no_fp, indices_to_remove)

    # Using the pd_onsets_ind, getting the time of the onsets of photodiode from pd_raw.time
    pd_onsets_time = pd_raw.times[pd_onsets_ind_no_fp]

    # Creating a dictionary of np array: "sample" is the sample corresponding to the onset, pd_onsets_time is the time
    pd_onsets = {"Sample_num": pd_onsets_ind_no_fp, "Time": pd_onsets_time}

    # Computing the size of the bars for the plot:
    min_amp = subject_info.PD_THRESHOLD - 0.5 * subject_info.PD_THRESHOLD
    max_amp = subject_info.PD_THRESHOLD + 0.5 * subject_info.PD_THRESHOLD

    # Get the path where to save data
    save_path = subject_info.path["preprocessing_sub-" + subject_info.SUBJ_ID
                                  + "_ses-" + subject_info.session + "_ieeg_trigger_alignment_figures"]

    # Plot raw signal with threshold and onsets (zoom in and check if they make sense)
    plt.figure(figsize=(8, 6))
    plt.plot(pd_raw.get_data()[0], 'g')
    plt.vlines(pd_onsets["Sample_num"], min_amp,
               max_amp, 'r', zorder=10, linewidth=2)
    plt.title('Raw photodiode signal with thresholds and onset.')
    plt.xlabel('Sample nr')
    plt.ylabel('Amplitude')

    if data_preparation_parameters.show_check_plots:
        plt.show()
        manual_remove_or_add_trigger_detections = \
            input(
                'Do you want to add or remove trigger onset detections? [Yes or No]')

        # Plot it again so that the correct one is saved.
        plt.plot(pd_raw["Amp"][0], 'g')
        plt.vlines(pd_onsets["Sample_num"], min_amp,
                   max_amp, 'r', zorder=10, linewidth=2)
        plt.title('Raw photodiode signal with thresholds and onset.')
        plt.xlabel('Sample nr')
        plt.ylabel('Amplitude')

    plt.savefig(os.path.join(save_path, "Raw_photodiode_signal.png"))
    plt.close()
    # Dumping the subject info and the analysis parameters with the figures:
    subject_info.save(save_path)
    data_preparation_parameters.save(save_path, subject_info.files_prefix)

    return pd_onsets


def handle_duplicates(full_logs):
    """
    This function reads in the log files and identify where there were repetitions and flags it
    :param full_logs: (pandas data frame) full logs from the first experiment
    :return: full_logs: (pandas data frame) data frame with duplication flag appended
    """
    # Finding the duplicates. What should be unique is the combination of block, miniblock and trial within those.
    # However, a trial consists most of the time of three events: stimulus, fixation and jitter, which will have the
    # same combination of the above. Hence the need for this specific subset
    full_logs['duplicate'] = full_logs.duplicated(
        subset=['block', 'miniBlock', 'trial', 'eventType'], keep='last')
    full_logs = full_logs.reset_index(drop=True)
    # To that, need to add the responses if there were any inbetween the repetition, because those won't be caught by
    # the above:
    for row_n, row in full_logs.iterrows():
        if row['eventType'] == "Response":
            if full_logs.loc[row_n - 1, 'duplicate'] == 1 and full_logs.loc[row_n + 1, 'duplicate'] == 1:
                full_logs.loc[row_n, 'duplicate'] = True

    return full_logs


def load_logs(root_path, file_naming_pattern, file_extension=None):
    """
    This function loads the log files and prepares them for the alignment with the photodiode flashes. This function
    will load recursively all the files with the matching naming pattern and extension along the root directory. So make
    sure you don't have multiple different files in the directory you will be searching to avoid loading spurious files
    :param root_path: (string or Pathlib path) root path to look for the files
    :param file_naming_pattern: (string) naming pattern of the files to load. You can use wildcards
    :param file_extension: (string) extension of the file to load. With the dot!
    :return: full_logs: (pandas data frame) log files of the experiment with duplicates removed
    """

    # Getting the list of the log files:
    files_list = find_files(
        root_path, naming_pattern=file_naming_pattern, extension=file_extension)

    # Make sure they are in order:
    files_list.sort()

    # Preparing to load the logs:
    full_logs = pd.DataFrame()
    # Looping through file list to load the logs:
    for files in files_list:
        full_logs = full_logs.append(pd.read_csv(files))

    full_logs = handle_duplicates(full_logs)

    return full_logs


def manual_trigger_fix(pd_onsets_signal, pd_onsets_logs, sr):
    """
    This function plots the signal and asks for user input. This function assumes that the log file contains all the
    information you need. If something doesn't align, this must therefore be because of missing triggers in your signal
    In this function, we therefore plot the signal. The user can then identify where things misalign and feed in the
    index of the triggers where the misalignment starts. The function will retrofit it into the timeline of the
    photodiode using the diff in the log time line. This will be repeated up until alignment is achieved!
    :param pd_onsets_logs: (pd.DataFrame) full log files with removed unecessary entries and duplicates
    :param pd_onsets_signal: (dict of np arrays)
    :param sr: (int) sampling rate of the signal
    :return:
    """

    def query_input():
        does_it_looks_good = \
            input(
                "Do you wish to manually insert triggers [Yes] or proceed keep the signal as is [No]?")
        index_to_add = None
        index_to_remove = None

        if does_it_looks_good != 'No' and does_it_looks_good != 'no':
            continue_looping = True
            # Asking for user input:
            index_to_add = input("Give in the index of a trigger you would like to add (or leave empty if you wish to "
                                 "remove one)?")
            if index_to_add == '':
                index_to_remove = input(
                    "Give in the index of a trigger you would like to remove (or leave empty if you "
                    "wish to remove one)?")
        else:
            continue_looping = False

        return continue_looping, index_to_add, index_to_remove

    # First, setting a few flags, because we will loop until the user says otherwise:
    keep_looping = True

    # Looping until we are told to stop
    while keep_looping:
        # computing the interval of the triggers in the file and in the photodiode:
        interval_pd = np.diff(pd_onsets_logs)
        interval_log = np.diff(pd_onsets_signal)

        # We can now plot those on top of another:
        fig, ax = plt.subplots()
        ax.plot(interval_pd, 'r', label='Photodiode')
        ax.plot(interval_log, 'b', label='Log file')
        plt.legend()
        plt.show()

        keep_looping, add_index, remove_index = query_input()

        if keep_looping:
            if add_index != '':
                add_index = int(add_index)
                try:
                    new_pd_ts = pd_onsets_signal[add_index] + \
                                (pd_onsets_logs[add_index + 1]
                                 - pd_onsets_logs[add_index])
                    new_pd_sample = pd_onsets_signal['Sample_num'][add_index] + \
                        (int(
                                        pd_onsets_logs[add_index + 1] - pd_onsets_logs[
                                            add_index]) * sr)
                    pd_onsets_signal = {
                        'Sample_num': np.insert(pd_onsets_signal['Sample_num'], add_index + 1, new_pd_sample),
                        'Time': np.insert(pd_onsets_signal['Time'], add_index + 1, new_pd_ts)
                    }
                except KeyError:
                    print(
                        "The index you set was outside the bounds of the array. Choose another index")
                except ValueError:
                    print("Are you sure you entered a number? Try again")
            elif remove_index != '':
                try:
                    remove_index = int(remove_index)
                    # If asked to remove a specific event from the photodiode timeline, simply removing it!
                    pd_onsets_signal = {
                        'Sample_num': np.delete(pd_onsets_signal['Sample_num'], remove_index + 1),
                        'Time': np.delete(pd_onsets_signal['Time'], remove_index + 1)
                    }
                except KeyError:
                    print(
                        "The index you set was outside the bounds of the array. Choose another index")
                except ValueError:
                    print("Are you sure you entered a number? Try again")
            else:
                print(
                    "You must pass an index to be able to adjust the alignment. If you are happy as is, reply correctly"
                    "to the input")

    return pd_onsets_signal


def check_alignment(pd_onsets_signal, pd_onsets_logs, sr, subject_info):
    """
    This function compares the onsets of the photodiode detected in the signal and what was logged during the experiment
    This acts as a sanity check because if the info isn't consistent between both, something went wrong. Chances are
    that the photodiode missed or detected one flash too much, but only close inspection enables obtaining answers.
    :param pd_onsets_logs: (1D np array of floats) containing onset of photodiode flashes
    :param pd_onsets_signal: (1D np array) containing the onset of photodiode logged during the experiment
    :param sr: (float or int) sampling rate of the signal
    :param subject_info: (subject info object) contains info about the path of where the things should be saved
    :return:
    """

    # First checking that the two arrays have the same size:
    if len(pd_onsets_signal) != len(pd_onsets_logs):
        # If the length differ, something is wrong. Asking the user for manual input to adjust if needed as needed:
        print("The number of detected triggers is not consistent with the log file. You will need to manually adjust")
        pd_onsets_signal = manual_trigger_fix(
            pd_onsets_signal, pd_onsets_logs, sr)

    # Once the two signals are of the same length, we can proceed to the actual alignment:
    pd_diff_signal = np.diff(pd_onsets_signal)
    pd_diff_log = np.diff(pd_onsets_logs)

    save_path = subject_info.path["preprocessing_sub-" + subject_info.SUBJ_ID
                                  + "_ses-V2_ieeg_trigger_alignment_figures"]
    # Now, plotting them on top of another:
    plt.figure()
    plt.plot(pd_diff_signal, "r", label="signal")
    plt.plot(pd_diff_log, "b", label="log")
    # Saving the plot:
    plt.savefig(os.path.join(save_path, "Photodiode_log_alignment.png"))

    return None


def read_csv(files):
    """
    Open csv file, iterate over the rows and map values to a list of dictionaries containing key/value pairs
    :param files: ()name of the file to open
    :return: log: pandas dataframe of the columns of interest
    """
    dict_of_lists = []
    for file in files:
        reader = csv.DictReader(open(file), delimiter=';')
        for line in reader:
            dict_of_lists.append(line)

    logs = pd.DataFrame.from_dict(dict_of_lists)

    return logs


def combine_logs(log_1, log_2, sort_col=None):
    """
    This function combines the info from the signal photodiode triggers to the info found in the logs. We need to do a
    bit of gymnastic to get things together
    :param log_1:
    :param log_2:
    :param sort_col:
    :return:
    """

    # Combining the two logs:
    combined_logs = pd.concat([log_1, log_2], axis=0, ignore_index=True)

    if sort_col is not None:
        # We now need to sort the log by the specified column:
        combined_logs = combined_logs.sort_values(
            by=sort_col).reset_index(drop=True)

    return combined_logs


def plot_temporal_precision(combined_logs, subject_info):
    """
    This function plots the difference between observed and expected stimuli duration. This is a sanity check to make
    sure that the experiment was behaving as expected.
    :param combined_logs: (pandas dataframe) contains the logs info and the photodiode time stamps
    :param subject_info: (subject info object) contains info about path to save the plot
    :return: None
    """

    # Extracting the stimuli onsets from the log
    stim_onsets = \
        combined_logs[combined_logs["triggerEvent"].isin(
            ["GameStimulusOn", "GameBlankOn", "ReplayTaskOn", "ReplayNonTaskOn"])]
    # Extracting the stimuli offset:
    stim_offsets = combined_logs.loc[combined_logs["triggerEvent"]
                                     == "AnimationPeakEnd"]

    # Taking the diff between the two:
    stim_dur_inaccuracy = [(x1 - x2) - STIM_DURATION
                           for (x1, x2) in zip(list(stim_offsets["pd_time"]), list(stim_onsets["pd_time"]))]

    save_path = subject_info.path["preprocessing_sub-" + subject_info.SUBJ_ID
                                  + "_ses-V2_ieeg_trigger_alignment_figures"]

    # Plotting the inaccuracies:
    fig, ax = plt.subplots()
    # Plotting the difference between the planned and observed stimuli duration:
    ax.scatter(range(len(stim_dur_inaccuracy)), [
               val * 1000 for val in stim_dur_inaccuracy])
    # Plotting horizontal lines are reference
    ax.hlines([-16, 16], xmin=0, xmax=len(stim_dur_inaccuracy), color="r")
    ax.set_xlabel("trial")
    ax.set_ylabel("Duration inaccuracy (msec)")
    ax.set_title("Video game stimuli duration inaccuracies")
    plt.savefig(os.path.join(save_path, "Stimuli duration inaccuracies.png"))

    return None


def format_logs_annot(combined_logs):
    """

    :param combined_logs:
    :return:
    """

    # Extracting only the photodiode events of interest:
    combined_logs_light = \
        combined_logs[combined_logs["triggerEvent"].isin(
            PD_RELVEVANT_TRIGGERS + ["_"])].reset_index(drop=True)

    # Declaring the time and events data frames:
    annotations = pd.DataFrame()
    timestamps = []

    # Looping through all the rows of the log:
    for row_ind, row in combined_logs_light.iterrows():
        # If the row has an underscore in the specified col, we skip it:
        if row["triggerEvent"] == "_":
            pass
        # But if it is one of the event for which we have the event log, we need to handle it specifically:
        elif row["triggerEvent"] == "GameStimulusOn" or row["triggerEvent"] == "GameBlankOn" \
                or row["triggerEvent"] == "ReplayTaskOn" or row["triggerEvent"] == "ReplayNonTaskOn":

            # If in this row, the trigger marks something for which we have an "event" logged, need to find the
            # corresponding event that is closest:
            event_found = False
            ctr = 0
            # Using a while loop to loop until we find what we are looking for:
            while event_found is not True and ctr < 100:
                ctr = ctr + 1
                # Going downward
                if combined_logs_light["triggerEvent"].iloc[row_ind + ctr] == "_":
                    # If we find the event log, keeping it to generate the annotation:
                    current_event_log = combined_logs_light.iloc[row_ind + ctr]
                    event_found = True
                # But also upward, as the trigger log can be before or after the event log:
                elif combined_logs_light["triggerEvent"].iloc[row_ind - ctr] == "_":
                    current_event_log = combined_logs_light.iloc[row_ind - ctr]
                    event_found = True

            # Since we are at a stimulus onset, there must also be a stimulus offset ("AnimationPeakEnd") at some
            # point after that, which we need to get the event duration:
            offset_found = False
            ctr = 0
            while offset_found is not True and ctr < 100:
                ctr = ctr + 1
                if combined_logs_light["triggerEvent"].iloc[row_ind + ctr] == "AnimationPeakEnd":
                    # The first animation peak end we find must be from the correct one:
                    stim_duration = combined_logs_light["pd_time"].iloc[row_ind
                                                                        + ctr] - row["pd_time"]
                    offset_found = True

            if "current_event_log" not in locals():
                raise ValueError("The event corresponding to the trigger wasn't found. Row = " + str(row_ind)
                                 + ", ctr=" + str(ctr))
            if "stim_duration" not in locals():
                raise ValueError("The stim offset trigger wasn't found. Row = " + str(row_ind)
                                 + ", ctr=" + str(ctr))
            # Based on that event log, we can append the event description to the dataframe:
            annotations = annotations.append({
                "event": "StimulusOn",
                "timestamp": row["pd_time"],
                "duration": stim_duration,
                "world": "World_" + str(row["world"]),
                "level": "Level_" + str(row["level"]),
                "stimID": current_event_log["stimID"],
                "stimType": "Blank" if current_event_log["stimType"] == "None" else current_event_log["stimType"],
                "stimName": current_event_log["stimName"],
                "stimDirection": current_event_log["stimDirection"],
                "isProbed": "Probed" if current_event_log["isProbed"] == "True" else "Unprobed",
                "response": np.nan if current_event_log["response"] == "N/A" else current_event_log["response"],
                "responseEvaluation": np.nan if current_event_log["response"] == "N/A"
                else current_event_log["responseEvaluation"],
            }, ignore_index=True)
            del stim_duration

            # If the stimulus was probed, then we need to find the photodiode of the probe:
            if current_event_log["isProbed"] == "True":
                ctr = 0
                probe_found = False
                while probe_found is False and ctr < 100:
                    ctr = ctr + 1
                    if combined_logs_light["triggerEvent"].iloc[row_ind + ctr] == "GameProbeOn":
                        probe_ts = combined_logs_light["pd_time"].iloc[row_ind + ctr]
                        probe_found = True
                if "probe_ts" not in locals():
                    raise ValueError("The GameProbeOn trigger wasn't found. Row = " + str(row_ind)
                                     + ", ctr=" + str(ctr))

                # We can now generate the event for the probe:
                annotations = annotations.append({
                    "event": "ProbeOn",
                    "timestamp": probe_ts,
                    "duration": float(current_event_log["responseTS"]) * 0.001
                    - float(current_event_log["probeTS"]) * 0.001,
                    "world": "World_" + str(row["world"]),
                    "level": "Level_" + str(row["level"]),
                    "stimID": current_event_log["stimID"],
                    "stimType": current_event_log["stimType"],
                    "stimName": current_event_log["stimName"],
                    "stimDirection": current_event_log["stimDirection"],
                    "isProbed": np.nan,
                    "response": current_event_log["response"],
                    "responseEvaluation": current_event_log["responseEvaluation"],
                }, ignore_index=True)
                del probe_ts

            # Additionally, if there was a response from the participant, we need to add it to the events:
            if current_event_log["isProbed"] == "True":
                annotations = annotations.append({
                    "event": "Response",
                    "timestamp": row["pd_time"] + float(current_event_log["responseTS"]) * 0.001
                    - current_event_log["timeSec"],
                    "duration": 0,
                    "world": "World_" + str(row["world"]),
                    "level": "Level_" + str(row["level"]),
                    "stimID": current_event_log["stimID"],
                    "stimType": current_event_log["stimType"],
                    "stimName": current_event_log["stimName"],
                    "stimDirection": current_event_log["stimDirection"],
                    "isProbed": np.nan,
                    "response": current_event_log["response"],
                    "responseEvaluation": current_event_log["responseEvaluation"],
                }, ignore_index=True)

            # If we are in the replay level, there are responses in the absence of probes. Need to take that into
            # account to enter the responses:
            if row["world"] == "L" and "NoResponse" not in row["response"]:
                annotations = annotations.append({
                    "event": "Response",
                    "timestamp": row["pd_time"] + float(current_event_log["responseTS"]) * 0.001
                    - current_event_log["timeSec"],
                    "duration": 0,
                    "world": "World_" + str(row["world"]),
                    "level": "Level_" + str(row["level"]),
                    "stimID": current_event_log["stimID"],
                    "stimType": current_event_log["stimType"],
                    "stimName": current_event_log["stimName"],
                    "stimDirection": current_event_log["stimDirection"],
                    "isProbed": np.nan,
                    "response": current_event_log["response"],
                    "responseEvaluation": current_event_log["responseEvaluation"],
                }, ignore_index=True)
            del current_event_log

        # If the event is not a probe nor non of the above, then it must be a filler, in which case we create the
        # annotation accordingly:
        elif row["triggerEvent"] != "GameProbeOn":
            annotations = annotations.append({
                "event": "FillerOn",
                "timestamp": row["pd_time"],
                "duration": FILLER_DURATION,
                "world": "World_" + str(row["world"]),
                "level": "Level_" + str(row["level"]),
                "stimID": np.nan,
                "stimType": np.nan,
                "stimName": np.nan,
                "stimDirection": np.nan,
                "isProbed": np.nan,
                "response": np.nan,
                "responseEvaluation": np.nan,
            }, ignore_index=True)

    # Now, sorting the rows of the data frame by time:
    annotations = annotations.sort_values(by="timestamp", ignore_index=True)

    # Then, extracting time and duration as np arrays:
    timestamps = annotations["timestamp"].to_numpy().astype(float)
    durations = annotations["duration"].to_numpy().astype(float)
    # Removing the two above from the table:
    del annotations["timestamp"], annotations["duration"]
    # Converting the table to list of slash separated strings:
    events_str = annotations.to_string(
        header=False, index=False, index_names=False).split('\n')
    events_descriptions = ["/".join(ele.split()) for ele in events_str]

    # Convert all of that to annotations:
    events_annotations = mne.Annotations(onset=timestamps,
                                         duration=durations,
                                         description=events_descriptions)

    return events_annotations


def data_preparation():
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument('--subjectID', type=str, default=None,
                        help="Subject ID, for instance SE110")
    parser.add_argument('--AnalysisParametersFile', type=str, default=None,
                        help="Analysis parameters file (file name + path)")
    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Instantiating parameters jsons
    # Creating the analysis parameters:
    print("Reading analysis info from analysis parameter config file")
    data_preparation_parameters = DataPreparationParameters(
        args.AnalysisParametersFile)

    # Creating the subject info based on the json file created:
    print("Reading subject info from subject info config file or preparing to create subject info file")
    subject_info = SubjectInfo(args.subjectID, data_preparation_parameters)
    # Get the site as some steps will differ a little based on it
    site = re.split('(\d+)', subject_info.SUBJ_ID)[0]
    # ------------------------------------------------------------------------------------------------------------------
    # Loading and preparing the photodiode signal:
    raw = load_signal(data_preparation_parameters.raw_root + os.sep + subject_info.SUBJ_ID,
                      data_preparation_parameters.ecog_files_naming, subject_info,
                      file_extension=data_preparation_parameters.ecog_files_extension,
                      debug=data_preparation_parameters.debug)

    print(raw.info)

    # Keeping only the photodiode channel:
    raw_pd = raw.copy().pick_channels([subject_info.TRIGGER_CHANNEL])

    # Adjusting the photodiode threshold if needed:
    subject_info = manual_adjust_threshold(subject_info, raw_pd)

    # Finally, the subject info in the json needs to be updated:
    subject_info.update_json()

    # ------------------------------------------------------------------------------------------------------------------
    # Extracting the photodiode triggers:
    # Computing triggers onsets:
    pd_onsets = extract_trigger_onset(
        raw_pd, subject_info, data_preparation_parameters)

    # ------------------------------------------------------------------------------------------------------------------
    # Checking alignment of detected triggers to log file:
    # This is a sanity check
    # First, loading the log files:
    log_file_names = find_files(data_preparation_parameters.raw_root + os.sep + subject_info.SUBJ_ID,
                                naming_pattern=data_preparation_parameters.beh_files_naming,
                                extension=data_preparation_parameters.beh_files_extension)
    # Loading and appending all these logs:
    logs = read_csv(log_file_names)

    # For consistency with the raw mne objects, convert time to sec:
    logs["timeSec"] = logs["timeMS"].astype(float) * 0.001
    del logs["timeMS"]

    # Extract the relevant info from the log files. It is a bit of a weird one but for the levels onset, we need
    # to extract the trigger info line as opposed to the trigger sent line otherwise:
    pd_trigger_logs = logs[((logs["sender"] == PD_ROW_ENTRIES)
                            & (logs["triggerState"] == "TRIGGER_SENT")
                            & (logs["triggerEvent"] != "LevelBegin")
                            & (logs["triggerEvent"] != "LevelEnd"))
                           | ((logs["sender"] == PD_ROW_ENTRIES)
                              & (logs["triggerState"] == "TRIGGER_INFO")
                              & ((logs["triggerEvent"] == "LevelBegin")
                                 | (logs["triggerEvent"] == "LevelEnd")))].reset_index(drop=True)

    # Align the log file entries to the detected onsets to investigate any discrepancies:
    check_alignment(pd_onsets["Time"], pd_trigger_logs["timeSec"].to_numpy().astype(
        float), raw.info["sfreq"], subject_info)

    # ------------------------------------------------------------------------------------------------------------------
    # Annotating the signal:
    # Following alignment between log files and signal triggers, we use the log file info to get the info of event
    # associated with each trigger to create the BIDS:
    # First things first, adding the pd time to the pd trigger log:
    pd_trigger_logs["pd_time"] = pd_onsets["Time"]

    # Then, combining the trigger log with the event part of the log:
    combined_logs = combine_logs(
        pd_trigger_logs, logs.loc[logs["triggerEvent"] == "_"], sort_col="timeSec")

    # Plotting the temporal accuray:
    plot_temporal_precision(combined_logs, subject_info)

    # Formatting the combined logs for conversion to annotations:
    events_annotations = format_logs_annot(combined_logs)

    # Finally, annotating the signal:
    raw.set_annotations(events_annotations)

    # ------------------------------------------------------------------------------------------------------------------
    # Setting channels type:
    try:
        elec_recon_files = find_files(data_preparation_parameters.raw_root + os.sep + subject_info.SUBJ_ID,
                                      naming_pattern=data_preparation_parameters.elec_loc[
                                          list(data_preparation_parameters.elec_loc.keys())[0]],
                                      extension=data_preparation_parameters.elec_loc_extension)
        # If there were more than 1 elec recon file, use whichever was the latest:
        if len(elec_recon_files) > 1:
            elec_recon_file = find_correct_file(elec_recon_files)
        else:
            elec_recon_file = elec_recon_files[0]
        print(elec_recon_file)
        # There are slight differences in format of elec recon between the different sites, which is adressed here:
        if site == "SF":
            # Loading the file:
            elec_coord_raw = np.genfromtxt(elec_recon_file, dtype=str, delimiter=' ',
                                           comments=None, encoding='utf-8')
            elec_coord_raw[:, 0] = remove_elecname_leading_zero(
                elec_coord_raw[:, 0])
            # Convert to a dataframe for ease of use:
            elec_coord = pd.DataFrame({
                "name": elec_coord_raw[:, 0],
                "x": elec_coord_raw[:, 1],
                "y": elec_coord_raw[:, 2],
                "z": elec_coord_raw[:, 3],
                "type": elec_coord_raw[:, 4],
            })
        elif site == "SE":
            # Loading the file:
            elec_coord_raw = np.genfromtxt(elec_recon_file, dtype=str, delimiter=',',
                                           comments=None, encoding='utf-8')
            # Keeping only the columns of interest
            elec_coord_raw = elec_coord_raw[1:, [0, 1, 3, 4, 5]]
            # Convert to a dataframe for ease of use:
            elec_coord = pd.DataFrame({
                "name": elec_coord_raw[:, 0],
                "x": elec_coord_raw[:, 2],
                "y": elec_coord_raw[:, 3],
                "z": elec_coord_raw[:, 4],
                "type": elec_coord_raw[:, 1],
            })

        # Looping through all channels in the signal, to set their types:
        for ind, ch in enumerate(raw.ch_names):
            # Finding the channel in the table:
            if ch in elec_coord["name"].to_list():
                # Set the channel type accordingly:
                if elec_coord.loc[elec_coord['name'] == ch, "type"].item() == "D":
                    raw.set_channel_types({ch: "seeg"})
                    print("{0}: {1}".format(ch, "seeg"))
                else:
                    raw.set_channel_types({ch: "ecog"})
                    print("{0}: {1}".format(ch, "ecog"))
            # If the electrode is not in the table but the letter match one of the naming convention, setting the:
            # channel type accordingly
            elif re.findall("[a-zA-Z]+", ch)[0] in list(data_preparation_parameters.additional_channel_conv.keys()):
                raw.set_channel_types({
                    ch: data_preparation_parameters.additional_channel_conv[re.findall("[a-zA-Z]+", ch)[0]]
                })
                print("{0}: {1}".format(ch, data_preparation_parameters.additional_channel_conv[re.findall("[a-zA-Z]+",
                                                                                                           ch)[0]]))
            else:  # Otherwise, set the channel as bad, we don't know what that is:
                raw.info['bads'].append(ch)
                print("{0}: {1}".format(ch, "bad"))
    except IndexError:
        # If the site is SE, then most of the electrodes are depth, therefore, setting all the channels to seeg. The
        # other type are handled below:
        if site == "SE":
            raw.set_channel_types({ch: "seeg" for ch in raw.ch_names})
        for elec_name_conv in data_preparation_parameters.additional_channel_conv[site].keys():
            raw.set_channel_types({ch_name: data_preparation_parameters.additional_channel_conv[elec_name_conv]
                                   for ch_name in raw.ch_names if elec_name_conv == ch_name[0:len(elec_name_conv)]})

    # Setting the trigger channel to stim. No matter what stands in the electrodes reconstruction files or what the
    # naming conventions are, the trigger channel is the stim channel. By doing it last, we ensure that it won't be
    # misattributed:
    raw.set_channel_types({subject_info.TRIGGER_CHANNEL: 'stim'})

    # ------------------------------------------------------------------------------------------------------------------
    # Convert to BIDS:
    # The game and the replay needs to be separated:
    # Find the time stamp of the replay onset:
    replay_onset = combined_logs["pd_time"].iloc[np.where(
        combined_logs["world"] == "L")[0][0]]

    # Chopping the file accordingly:
    game_raw = raw.copy().crop(
        tmin=0.0, tmax=replay_onset - 1 / raw.info["sfreq"])
    replay_raw = raw.copy().crop(tmin=replay_onset, tmax=raw.times[-1])
    del raw
    # Saving the game to BIDS:
    try:
        # Creating a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Generate a file name
        fname = temp_dir + os.sep + "_raw.fif"
        # Saving the data in that temp dir:
        game_raw.save(fname)
        save_to_BIDs(mne.io.read_raw_fif(fname, preload=False), elec_recon_file=elec_recon_file,
                     bids_root=data_preparation_parameters.BIDS_root,
                     subject_id=subject_info.SUBJ_ID, session=data_preparation_parameters.session,
                     task="VG", data_type="ieeg",
                     line_freq=data_preparation_parameters.line_freq)
    finally:  # Removing the temporary directory:
        files = glob.glob(temp_dir + os.sep + '*')
        for f in files:
            os.remove(f)
        os.rmdir(temp_dir)

    # Saving the replay to BIDS:
    try:
        # Creating a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Generate a file name
        fname = temp_dir + os.sep + "_raw.fif"
        # Saving the data in that temp dir:
        replay_raw.save(fname)
        print(data_preparation_parameters.BIDS_root)
        save_to_BIDs(mne.io.read_raw_fif(fname, preload=False), elec_recon_file=elec_recon_file,
                     bids_root=data_preparation_parameters.BIDS_root,
                     subject_id=subject_info.SUBJ_ID, session=data_preparation_parameters.session,
                     task="Replay", data_type="ieeg",
                     line_freq=data_preparation_parameters.line_freq)
    finally:  # Removing the temporary directory:
        files = glob.glob(temp_dir + os.sep + '*')
        for f in files:
            os.remove(f)
        os.rmdir(temp_dir)

    # Finally, the subject info in the json needs to be updated:
    subject_info.update_json()
    print("Done. Ready to preprocessing EDFPreprocessing.py!")


if __name__ == "__main__":
    data_preparation()
