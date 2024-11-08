# <--------------------------------------------------------------------------------------------------> #
# <------------------------------------------MODULE IMPORTS------------------------------------------> #
# <--------------------------------------------------------------------------------------------------> #


import numpy as np

import pandas as pd

import datetime as dt

import wfdb

from typing import Tuple, List, Dict


# <-------------------------------------------------------------------------------------------------->#
# <-------------------------------------------DUKE FUNCTIONS----------------------------------------->#
# <-------------------------------------------------------------------------------------------------->#


def get_Duke_mapping_df(
    input_dir: str,
    mapping_filename: str,
) -> pd.DataFrame:
    '''
    We'd get each site to provide a file/info. on mapping to standard names for channels etc.
    
    This function just reads it in and sets the 'signal_id' as the index for efficient lookup
    of channel metadata.
    
    Args:
        input_dir (str): Directory path where the mapping file is saved.
        mapping_filename (str): The name of the mapping file to read.
        
    Returns:
        Sickbay_Fast_Lookup_df (pd.DataFrame): The DF used to lookup the channel information from the 'signal_id'. 
        
    '''
    
    # Read in the Sickbay dataframe to map from the 'signal_id's to the other meta data fields. (Looks like this file came from: sickbay.data.get_master_signal_list())
    Sickbay_Channel_Mapping_df = pd.read_excel(f'{input_dir}/{mapping_filename}')

    # Get just the columns we need for lookup, and set the 'signal_id' as the index for fast (hashmap-like structure) lookup.
    Sickbay_Fast_Lookup_df = Sickbay_Channel_Mapping_df[['signal_id', 'class', 'sampling_rate', 'units']].set_index('signal_id')
    
    return Sickbay_Fast_Lookup_df


def read_Duke_WF(
    input_dir: str, 
    input_filename: str,
) -> Tuple[str, str, str, pd.DataFrame]:
    '''
    We'd get each site to write a function like this, that reads a WF data file and 
    returns the signal data along with some necessary metadata.
    
    Args:
        input_dir (str): Directory path where the data file is saved.
        input_filename (str): Name of the file to be processed.

    Returns:
        input_file_ID (str): The patient identifier for this file.
        input_date (str): The start date for the data (YYYYMMDD).
        input_time (str): The start time for the data (HHMMSS).
        signal_df (pd.DataFrame): The DataFrame that contains the signal data.
    '''
    
    # Remove the file extension.
    filename = input_filename.split('.')[0]

    # Split the filename by the underscores '_'.
    meta_info_parts = filename.split('_')

    # Extract the Sickbay ID from the split filename.
    input_file_ID = meta_info_parts[0]

    # Extract the start date from the split filename.
    input_date = meta_info_parts[1]

    # Extract the start time from the split filename.
    input_time = meta_info_parts[2]

    # Read in the input csv.
    signal_df = pd.read_csv(f'{input_dir}/{input_filename}')
    
    return input_file_ID, input_date, input_time, signal_df


# <-------------------------------------------------------------------------------------------------->#
# <----------------------------------------METADATA EXTRACTION--------------------------------------->#
# <-------------------------------------------------------------------------------------------------->#


def get_channel_cols(signal_df: pd.DataFrame) -> List[str]:
    '''
    Function to extract the columns in the DataFrame that contain the signal data.
    
    Args:
        signal_df (pd.DataFrame): The DataFrame that contains the signal data.

    Returns:
        channel_cols (List[str]): The list of the channel_IDs that are present in the data. 
    '''
    
    # Define any columns that are not channel columns.
    non_channel_cols = [TIMING_COL]

    # Make sure that the timing column is present in the signal dataframe.
    if TIMING_COL not in signal_df.columns:
        raise ValueError("The timing column is not present in the input dataframe!")

    # Check for and drop any columns that are all NaNs, for some reason.
    signal_df.dropna(axis=1, how='all', inplace=True)

    # Get the columns with WF data in them, i.e. all other columns that are not used for timing.
    channel_cols = [channel for channel in signal_df.columns if channel not in non_channel_cols]

    # Check that there is actually some useful data in the dataframe.
    if len(channel_cols)==0:
        raise ValueError("There are no signal columns, with any data, in the input dataframe!")
        
    return channel_cols


def get_signal_meta_dicts(
    channel_cols: List[str], 
    mapping_df: pd.DataFrame, 
) -> Tuple[Dict, Dict, Dict]:
    '''
    Function to extract the relevant channel metadata, for the present channels, from the mapping 
    dataframe, and pack them into dictionaries for later lookup.
    
    Args:
        channel_cols (List[str]): The list of the channel_IDs that are present in the data. 
        mapping_df (pd.DataFrame): The DF used to lookup the channel information from the 'signal_id'.
        
    Returns:
        sig_names_dict (Dict): Dictionary to lookup the channel signal names.
        samp_freqs_dict (Dict): Dictionary to lookup the channel signal sampling frequencies.
        units_dict (Dict): Dictionary to lookup the channel signal units.
    '''
        
    # Create empty dicts. for the signal names, sampling frequencies and the units.
    sig_names_dict, samp_freqs_dict, units_dict = {}, {}, {}

    # Loop over the channel IDs and extract the name, fs and units from each.
    for channel_ID in channel_cols:

        # The way we've setup the 'Sickbay_Fast_Lookup_df', the cols are 'class', 'sampling_rate', 'units' with the 'signal_id' as the index.
        channel_name, channel_samp_freq, channel_unit = mapping_df.loc[int(channel_ID)].values

        # Add the channel name to the relevant dict.
        sig_names_dict[channel_ID] = channel_name

        # Add the channel sampling frequency to the relevant dict.
        samp_freqs_dict[channel_ID] = channel_samp_freq

        # Add the channel units to the relevant dict.
        units_dict[channel_ID] = channel_unit

    return sig_names_dict, samp_freqs_dict, units_dict


# <-------------------------------------------------------------------------------------------------->#
# <---------------------------------------TIMING/SAMPLING CHECKS------------------------------------->#
# <-------------------------------------------------------------------------------------------------->#


def check_timing_coherancy(
    signal_df: pd.DataFrame,
    expected_fs: int,
    hz_tolerance: int,
) -> int: 
    '''
    Here we check if the inferred sampling frequency from the data file 'TIMING_COL'
    is within tolerance (5Hz at the moment?) of what the mapping dataframe expects.
    
    Args:
        signal_df (pd.DataFrame): The DataFrame that contains the signal data.
        expected_fs (int): The expected sampling frequency of the data in the file
        hz_tolerance (int): The maximum acceptable difference between the expected and measured sampling frequencies.
    Returns:
        fs (int): The nominal sampling frequency for the data. 
    '''
    
    # Calculate the median time between samples across the datafile.
    median_timing_interval = signal_df[TIMING_COL].diff().median()

    # Covert this to an integer (necessary for np.timedelta) representing the median sampling frequency, in whatever units is passed.
    median_sampling_freq = int(1/median_timing_interval)

    # Convert to a np.timedelta, with appropriate units and divide by the timedelta for 1 second, to convert to Hz.
    measured_fs = np.timedelta64(median_sampling_freq, EXPECTED_TIMING_UNITS)/np.timedelta64(1,'s')

    # Add check, if within 5Hz of quoted sample freq., then go with quoted sampling freq.
    if abs(measured_fs - expected_fs) < hz_tolerance:
        fs = expected_fs # [wfdb.io.Record PARAM]

    else:
        raise ValueError("The measured sampling frequency does not match the quoted sampling frequency!")
        
    return fs


def add_timing_index(
    signal_df: pd.DataFrame, 
    input_date: str, 
    input_time: str,
    fs: int,
) -> Tuple[dt.datetime.date, dt.datetime.time]:
    '''
    Function to convert the timing col. in the signal dataframe (inplace) to a datetime index. 
    This makes things easy as we segment the data matrix into chunks. 
    Also returns the base date/time after conversion to dt.datetime objects. 
    
    Args:
        signal_df (pd.DataFrame): The DataFrame that contains the signal data.
        input_date (str): The start date for the data (as a string).
        input_time (str): The start time for the data (as a string).
        fs (int): The nominal sampling frequency for the data.

    Returns:
        base_date (dt.datetime.date): The start date for the data (as a datetime object).
        base_time (dt.datetime.time): The start time for the data (as a datetime object).
    '''

    # Convert the string representing the start time to a datetime time object.
    base_time = dt.datetime.strptime(input_time, '%H%M%S').time() # [wfdb.io.Record PARAM]

    # Convert the string representing the start date to a datetime date object.
    base_date = dt.datetime.strptime(input_date, "%Y%m%d").date() # [wfdb.io.Record PARAM]

    # Create a 'pd.DatetimeIndex' from the start datetime using the assumed sampling frequency.
    timestamp_index = pd.date_range(start=dt.datetime.combine(base_date, base_time), periods=signal_df.shape[0], freq=pd.Timedelta(seconds=1/fs))

    # Set the 'pd.DatetimeIndex' as the index for our signal dataframe.
    signal_df.insert(0, TIMING_INDEX_COL, timestamp_index)

    # Drop the original timing column.
    signal_df.drop([TIMING_COL], axis=1, inplace=True)
    
    return base_date, base_time


# <-------------------------------------------------------------------------------------------------->#
# <--------------------------------------------SEGMENTATION------------------------------------------>#
# <-------------------------------------------------------------------------------------------------->#


def check_missing_data_blocks(
    signal_df: pd.DataFrame,
    channel_cols: List[str],
) -> Tuple[List, List]:
    '''
    Check for rows in the data matrix where ALL the signals are missing.
    
    Args:
        signal_df (pd.DataFrame): The DataFrame that contains the signal data.
        channel_cols (List[str]): The list of the channel_IDs that are present in the data.

    Returns:
        data_block_start_indices (List[int]): The indices where chunks of data begin.
        NaN_block_start_indices (List[int]): The indices where chunks of NaNs begin.
    '''

    # Create a mask where any signal columns are non-NaN (i.e. atleast one channel has some data).
    data_mask = signal_df[channel_cols].notna().any(axis=1)

    # Do a diff between consecutive rows of the mask, this will give '-1' where a block of NaNs starts and '1' where a block of data starts.
    data_NaN_transitions = data_mask.astype(int).diff()

    # Indices where blocks of data start.
    data_block_start_indices = data_NaN_transitions[data_NaN_transitions==1].index.tolist()

    # Indices where blocks of NaNs start.
    NaN_block_start_indices = data_NaN_transitions[data_NaN_transitions==-1].index.tolist()

    # The diff method doesnt explicitly capture the first row, double check it has some data.
    if signal_df.iloc[0][channel_cols].notna().any():
        data_block_start_indices = [0] + data_block_start_indices

    # For the case (likely most of the time) when the zeroth row does have data, append a dummy 'end of data' index.
    if len(NaN_block_start_indices) == (len(data_block_start_indices)-1):
        NaN_block_start_indices.append(signal_df.shape[0]+1)

    return data_block_start_indices, NaN_block_start_indices


def check_oversized_chunks(
    data_block_start_indices: List[int], 
    NaN_block_start_indices: List[int],
    fs: int,
) -> Tuple[List, List]:
    '''
    Check for chunks of data that are larger than our desired segment size,
    adjust the final start/stop indices for segmentation appropriately. 
    
    Args:
        data_block_start_indices (List[int]): The indices where chunks of data begin.
        NaN_block_start_indices (List[int]): The indices where chunks of NaNs begin.
        fs (int): The nominal sampling frequency for the data. 

    Returns:
        segment_start_indices (List[int]): The final indices where segments of data begin.
        segment_stop_indices (List[int]): The final indices where segments of data end.
    '''
    
    # Set the maximum size of each segment.
    max_segment_size = fs*MAX_SEGMENT_LENGTH_SECONDS

    # Create new lists to store the final segment start and stop indices.
    segment_start_indices = []
    segment_stop_indices = []

    # Iterate through the original start and stop indices for the data blocks.
    for start, stop in zip(data_block_start_indices, NaN_block_start_indices):

        # Check if we've passed the end of the current data block.
        while start < stop:

            # Calculate the current chunk size.
            chunk_size = stop - start

            # If the chunk size exceeds the threshold, split it up.
            if chunk_size > max_segment_size:

                # Split the chunk at the 'max_segment_size'.
                new_stop = start + max_segment_size  
                # Keep the last start index.
                segment_start_indices.append(start)
                # Append the new stop index from the split.
                segment_stop_indices.append(new_stop)
                # Move the start index forward for the next iteration.
                start = new_stop  

            # If the chunk size is within the threshold, keep it as is.
            else:

                # Keep the start index.
                segment_start_indices.append(start)
                # Keep the stop index.
                segment_stop_indices.append(stop)
                # Move to the next chunk by breaking out of the 'while' loop.
                break 
                
    return segment_start_indices, segment_stop_indices


# <-------------------------------------------------------------------------------------------------->#
# <----------------------------------------SEGMENT DATA PREP.---------------------------------------->#
# <-------------------------------------------------------------------------------------------------->#


def prepare_segment_data(
    *,
    signal_df: pd.DataFrame, 
    channel_cols: List[str], 
    master_record_name: str, 
    segment_start_indices: List[int], 
    segment_stop_indices: List[int],
) -> Tuple[List, List, List, List]:
    '''
    Loop over the start/stop indices and for each segment create the segment name,
    start time, data matrix and present channels entries. 
    
    Args:
        signal_df (pd.DataFrame): The DataFrame that contains the signal data.
        channel_cols (List[str]): The list of the channel_IDs that are present in the data.
        master_record_name (str): The name for the master record, from which the segment record names are derived.
        segment_start_indices (List[int]): The final indices where segments of data begin.
        segment_stop_indices (List[int]): The final indices where segments of data end.

    Returns:
        segment_record_names_list (List): The list of the records that represent the segments in the MultiRecord.
        data_segment_start_times_list (List): The list of the start times for each segment.
        data_segment_list (List): The list of data matrixes that represent the data across all channels present in each segment. 
        segment_channels_list (List): The list of channels that are actually present in each segment. 
    '''
    
    # Create a list holding the 'record_name's for each segment, which are the 'master_record_name' with a 4-digit index after an underscore.
    segment_record_names_list = []

    # Create a list to hold the start timestamps for each segment.
    data_segment_start_times_list = []

    # Create a list to hold the data matrices for each segment.
    data_segment_list = []

    # Create a list to keep track of which channels are present in each segment.
    segment_channels_list = []

    # Loop over the starting indices for each segment of data.
    for i in range(len(segment_start_indices)):

        # Get the segment name and append it to the list (forcing the 'segment ID' to be a 4-digit index after an underscore).
        segment_name = f"{master_record_name}_{i:04d}"
        segment_record_names_list.append(segment_name)

        # Chunk the dataframe into a segment containing the ith block of data.
        data_segment_df = signal_df.iloc[segment_start_indices[i]:segment_stop_indices[i]]

        # Extract the start timestamp of the segment from the segmented DF.
        block_start_time = data_segment_df[TIMING_INDEX_COL].values[0]
        data_segment_start_times_list.append(block_start_time)

        # Get a list of the channels that are not all NaNs, for this segment. Make 'channel_cols' a np.array to use boolean indexing with the non-empty cols.
        good_channel_cols = list(np.array(channel_cols)[data_segment_df[channel_cols].notna().any().values])
        segment_channels_list.append(good_channel_cols)

        # Set the matrix of waveform data values, in physical units, as the values from the signal columns in the segmented DF.
        data_values = data_segment_df[good_channel_cols].values # (p_signal) [wfdb.io.Record PARAM]
        data_segment_list.append(data_values)

    if not len(segment_record_names_list)==len(data_segment_list)==len(data_segment_start_times_list):
        raise ValueError("Something went wrong in the segmentation process!")

    return segment_record_names_list, data_segment_start_times_list, data_segment_list, segment_channels_list


# <-------------------------------------------------------------------------------------------------->#
# <---------------------------------------CREATE SEGMENT RECORDS------------------------------------->#
# <-------------------------------------------------------------------------------------------------->#


def create_segment_records(
    *,
    segment_record_names_list: List, 
    data_segment_start_times_list: List, 
    data_segment_list: List, 
    segment_channels_list: List, 
    units_dict: Dict, 
    sig_names_dict: Dict, 
    fs: int,
) -> List[wfdb.io.Record]:
    '''
    Loop over the start/stop indices and create the segment name, start time, data matrix, 
    present channel entries and finally the wfdb.io.Record object for each segment.
    
    Args:
        segment_record_names_list (List): The list of the records that represent the segments in the MultiRecord.
        data_segment_start_times_list (List): The list of the start times for each segment.
        data_segment_list (List): The list of data matrixes that represent the data across all channels present in each segment. 
        segment_channels_list (List): The list of channels that are actually present in each segment. 
        units_dict (Dict): Dictionary to lookup the channel signal units.
        sig_names_dict (Dict): Dictionary to lookup the channel signal names.
        fs (int): The nominal sampling frequency for the data. 
        
    Returns:
        segments_record_list (List[wfdb.io.Record]): The collated list of the record objects that represent each segment. 
    '''
    
    # Create the list that holds the segment 'wfdb.io.Record's.
    segments_record_list = [] # [wfdb.io.multiRecord PARAM]

    # Loop over the segments to create the list of segment records.
    for i in range(len(segment_record_names_list)):

        # See: class wfdb.io.Record from https://wfdb.readthedocs.io/en/latest/io.html#module-wfdb.io

        # NB from the wfdb.wrsamp() function:
        # For more control over attributes, create a `Record` object, manually set its attributes, and call its `wrsamp` instance method. 
        # If you choose this more advanced method, see also the `set_defaults`, `set_d_features`, and `set_p_features` instance methods to help populate attributes.

        # Here I opt to create the 'wfdb.io.Record' object, passing the 'p_signal' etc. and then call 'set_d_features' and 'set_defaults' and then call the wrsamp method.
        # The above is exactly what would happen if you just use the wfdb.wrsamp() function directly.

        """

        # MUST-HAVE FIELDS FOR SICKBAY DATA (ends up in the 'Record.__dict__'): [TOTAL = 8]

        p_signal (np.ndarray): An (MxN) 2d numpy array, where M is the signal length, of the physical signal values intended to be written.
        record_name (str): The name of the WFDB record to be read, without any file extensions. 
        fs (float): The sampling frequency of the record.
        base_datetime (datetime.datetime): The datetime at the beginning of the record.
        fmt (list): A list of strings giving the WFDB format of each file used to store each channel. 
        units (list): A list of strings giving the units of each signal channel.
        sig_name (list): A list of strings giving the signal name of each signal channel.

        MANDATORY BUT AUTOMATICALLY CALCULATED FROM THE DATA: [TOTAL = 2]
        n_sig (int): Total number of signals (N) (calculated from data automatically during set_d_features(do_adc=1) method).
        sig_len (int): The total length of the signal (M) (calculated from data automatically during set_d_features(do_adc=1) method).

        """
        print(f"Working on segment #{i}")

        # Get the list of channels that are present in this segment.
        segment_channels = segment_channels_list[i]

        # Get the physical signal matrix from the list of data segments.
        segment_p_signal = data_segment_list[i]

        # Get the segment record name from the list.
        segment_record_name = segment_record_names_list[i]

        # Get the record start time for the segment, and convert it to a datetime.datetime. 
        segment_base_datetime = pd.to_datetime(data_segment_start_times_list[i]).to_pydatetime()

        # Get the units for the channels that appear in the segment.
        segment_units = [units_dict.get(channel) for channel in segment_channels]

        # Get the names for the channels that appear in the segment.
        segment_sig_name = [sig_names_dict.get(channel) for channel in segment_channels]

        # We want the same binarized format for all channels, expand the list to match the number of channels.
        segment_fmt = BINARY_FMT * len(segment_channels)

        # Create the WFDB Record Object.
        segment_Record = wfdb.io.Record(
            p_signal=segment_p_signal, 
            record_name=segment_record_name, 
            fs=fs, 
            base_datetime=segment_base_datetime,
            fmt=segment_fmt, 
            units=segment_units, 
            sig_name=segment_sig_name, 
        )

        # Follow some additional steps before writing (NB: these are done automatically if you call the wfdb.wrsamp() function directly).

        # Compute optimal fields to store the digital signal, carry out adc, and set the fields (also automatically calculates 'n_sig' and 'sig_len' from the data).
        segment_Record.set_d_features(do_adc=1)

        # Set default values of any missing field dependencies (sets: 'filename', 'adc_res', 'adc_zero', 'block_size').
        segment_Record.set_defaults()

        # Sanity checks:

        #if Sickbay_Record.__dict__['sig_len'] != signal_df.shape[0]:
        #    raise ValueError("The signal length does not match the length of the input dataframe!")

        '''
        if segment_Record.__dict__['n_sig'] != len(channel_cols):
            raise ValueError("The number of signals does not match the number of signal columns in the input dataframe!")

        if segment_Record.__dict__['p_signal'].shape[1] != len(channel_cols):
            raise ValueError("The channel dimension does not match the number of signal columns in the input dataframe!")

        if len(segment_Record.__dict__['sig_name']) != len(channel_cols):
            raise ValueError("The number of channel names does not match the number of signal columns in the input dataframe!")

        if len(segment_Record.__dict__['units']) != len(channel_cols):
            raise ValueError("The number of units does not match the number of signal columns in the input dataframe!")

        if len(segment_Record.__dict__['fmt']) != len(channel_cols):
            raise ValueError("The number of formats does not match the number of signal columns in the input dataframe!")
        '''

        # Append the segment record to the list.
        segments_record_list.append(segment_Record)

    return segments_record_list


# <-------------------------------------------------------------------------------------------------->#
# <----------------------------------------CREATE MULTI RECORD--------------------------------------->#
# <-------------------------------------------------------------------------------------------------->#


def create_multirecord(
    *,
    segments_record_list: List[wfdb.io.Record], 
    master_record_name: str,
    segment_record_names_list: List, 
    data_segment_start_times_list: List,
    channel_cols: List,   
    sig_names_list: List, 
    units: List,
    fs: int,
) -> wfdb.io.MultiRecord:
    '''
    Create the final wfdb.io.MultiRecord object with the list of segment records and other metadata for the 
    'master record' that represents all the data for this patient/session.
    
    Args:
        segments_record_list (List[wfdb.io.Record]): The list containing the records for all segments that comprise the MultiRecord.
        master_record_name (str): The name for the master record, from which the segment record names are derived.
        segment_record_names_list (List): The list of the records that represent the segments in the MultiRecord.
        data_segment_start_times_list (List): The list of the start times for each segment.
        channel_cols (List[str]): The list of the channel_IDs that are present in the data.
        sig_names_list (List[str]): The list of all channel signal names in the MultiRecord.
        units (list): A list of strings giving the units of each signal channel.
        fs (int): The nominal sampling frequency for the data. 
        
    Returns:
        Final_MultiRecord (wfdb.io.MultiRecord): The final MultiRecord object with all data and metadata for this patient/session. 
    '''

    # Create the WFDB MultiRecord Object.
    Final_MultiRecord = wfdb.io.MultiRecord(
        # MultiRecord only stuff:
        segments=segments_record_list,
        layout='variable',
        # 'seg_len'=... (Set automatically by MultiRecord constructor)
        # 'n_seg'=... (Set automatically by MultiRecord constructor)
        # Parallel to Record:
        record_name=master_record_name,
        seg_name=segment_record_names_list,
        n_sig=len(channel_cols),
        fs=fs,
        sig_len=None,
        #Maybe do a check to make sure this is the same as the base date and base time from the beginning of script?
        base_datetime=pd.to_datetime(data_segment_start_times_list[0]).to_pydatetime(), # Should be equal to dt.datetime.combine(base_date, base_time)
        sig_name=sig_names_list,
    )

    # Set the total signal length as the sum of the lengths of all segments.
    # Does this change if we include the missing segments as the '~' segments?
    Final_MultiRecord.__dict__['sig_len'] = np.sum(Final_MultiRecord.__dict__['seg_len'])

    # Set the total number of signals as the number of overall channels that appear anywhere in the record.
    Final_MultiRecord.__dict__['n_sig'] = len(Final_MultiRecord.__dict__['sig_name'])

    # Set the units to be the list of units for all channels that appear anywhere in the record.
    Final_MultiRecord.__dict__['units'] = units

    return Final_MultiRecord


# <-------------------------------------------------------------------------------------------------->#
# <--------------------------------------------END OF FILE------------------------------------------->#
# <-------------------------------------------------------------------------------------------------->#