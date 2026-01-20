import numpy as np
import pandas as pd
import pdb
import pickle
import datetime as dt
import logging

def test_consistency_of_annotation_table(t, verbose=True):
    """A collection of tests to make sure that the annotation
    table has no obvious mistakes
    
    Parameters:
    -----------
    t : pandas Dataframe
    verbose : bool, print conflicting lines
    
    Returns:pandas Dataframe
    -----------
    void
    
    """
    
    # -------------------
    # One aninal, one channel. 
    # -----
    # Each animal is associated with one recording channel. 
    # Thus each animal must only have one channel entry.
    gpb_ID = t.groupby('ID')
    for n_i, gp_i in gpb_ID:
        assert(len(gp_i['Channel'].unique())==1)
    
    # Make sure there is no known overlap in recordings.
    bool_conflict = (t['Conflict overlap recording']=='x').any()
    if bool_conflict:
        print('Conflict of overlapping recordings detected. Please fix')
        print(t_conflict)
    assert(bool_conflict == False)
        
    # Shared channels between animals are only allowed to
    # overlap on a given day if "missing data" is marked.
    # This may happen when an animal is terminated on the same day recording starts in another animal
    t = t[t['Missing data'].isnull()]
    gpb_ID = t.groupby(['Date', 'Channel', 'Machine'])
    for n_i, gp_i in gpb_ID:
        if verbose and len(gp_i['ID'].unique())>1:
            print(gp_i)
        assert(len(gp_i['ID'].unique())==1)
        

def actions_to_dataframe(actions):
    ls_actions = []
    for name_action, action in actions.items():
        # extract attribues
        attr = action.attributes
        dct = {}
        dct['name'] = action.id
        dct['datetime'] = action.datetime
        dct['type'] = attr['type']
        
        if 'data' in attr.keys():
            dct['data_path'] = {}
            for key, val in attr['data'].items():
                dct['data_path'][key] = action.data_path(key)
        if 'info' in action.modules.keys():     
            dct = {**dct, **dict(action.modules['info'])}
        ls_actions.append(dct)
    df = pd.DataFrame(ls_actions)
    return df


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

# Initialize logger
logger = logging.getLogger(__name__)

def define_epg_phases(df: pd.DataFrame, p_phase: dict, p_columnnames: dict) -> dict:
    """
    Defines experimental phases (Baseline, Latent, Chronic) based on stimulation 
    and seizure events.

    Logic Definitions:
    - Latent 0: Fixed period immediately following PPS stimulation (e.g., 3 days).
    - Chronic Seizure: The first seizure occurring >= 7 days after PPS stimulation.
    - Latent 2: Fixed period ending immediately before the first chronic seizure.
    - Latent 1: Centered period between L0 and L2, only if sufficient gap exists.
    - Chronic 0-2: Sequential fixed periods starting from the first chronic seizure.

    Parameters:
    -----------
    df : pd.DataFrame
        Annotation table containing columns for Date, ID, Seizure, Surgery, etc.
    p_phase : dict
        Configuration dict containing 'len' (dt.timedelta) for each phase 
        (e.g., p_phase['Latent_0']['len']).
    p_columnnames : dict
        Mapping for dataframe column names (keys: 'date', 'pps_stim', 'surgery', etc.).

    Returns:
    --------
    dict
        Dictionary of start/stop datetime objects for each identified phase.
        Returns None if critical data (like stimulation) is missing.
    """

    # 1. Validation and Setup
    # -----------------------------------------------------------
    if df.empty:
        logger.warning("define_epg_phases received an empty DataFrame.")
        return None

    required_keys = ['date', 'pps_stim', 'surgery']
    if not all(k in p_columnnames for k in required_keys):
        raise KeyError(f"p_columnnames missing required keys: {required_keys}")

    # Ensure dataframe is sorted by date for logical processing
    df = df.sort_values(p_columnnames['date'])
    
    # Extract Animal ID for logging
    try:
        animal_idx = df['ID'].unique()[0]
    except (KeyError, IndexError):
        animal_idx = "Unknown"

    res = {}

    # 2. Define Global Boundaries (Recording & Stimulation)
    # -----------------------------------------------------------
    try:
        # End of recording
        date_last = df[p_columnnames['date']].max()
        res['recording_last'] = date_last

        # Stimulation Window
        bool_stim = ~df[p_columnnames['pps_stim']].isnull()
        df_stim = df[bool_stim]

        if df_stim.empty:
            logger.warning(f"Animal {animal_idx}: No stimulation found. Skipping phase definition.")
            return None

        # Stimulation Start (Beginning of first stim day)
        date_start_stim = df_stim[p_columnnames['date']].min()
        res['stimulation_start'] = dt.datetime.combine(date_start_stim, dt.time.min)

        # Stimulation Stop (End of last stim day)
        date_stop_stim = df_stim[p_columnnames['date']].max()
        res['stimulation_stop'] = dt.datetime.combine(date_stop_stim, dt.time.max)

    except Exception as e:
        logger.error(f"Error defining stimulation boundaries for {animal_idx}: {e}")
        return None

    # 3. Define Baseline Phases
    # -----------------------------------------------------------
    # Surgery date
    bool_surg = ~df[p_columnnames['surgery']].isnull()
    df_surg = df[bool_surg]

    if len(df_surg) == 1:
        date_surg = dt.datetime.combine(df_surg.iloc[0][p_columnnames['date']], dt.time.min)
        res['surgery'] = date_surg
        
        # BL_0: Starts 1 day after surgery
        date_aftersurg = date_surg + dt.timedelta(days=1)
        res['BL_0_start'] = dt.datetime.combine(date_aftersurg, dt.time.min)
        res['BL_0_stop'] = date_aftersurg + p_phase['BL_0']['len'] - dt.timedelta(seconds=1)

        # BL_1: Ends right before stimulation starts
        res['BL_1_start'] = res['stimulation_start'] - p_phase['BL_1']['len']
        res['BL_1_stop'] = res['stimulation_start'] - dt.timedelta(seconds=1)
    else:
        logger.info(f"Animal {animal_idx}: Surgery info missing or ambiguous. Skipping Baseline phases.")


    # 4. Define Latent Start & Latent 0 (Immediate Post-Stim)
    # -----------------------------------------------------------
    # Latent Start: Immediately the moment after stimulation ends
    res['latent_start'] = res['stimulation_stop'] + dt.timedelta(seconds=1)

    # Latent 0: Defined as the first X days (usually 3) after stimulation
    res['Latent_0_start'] = res['latent_start']
    res['Latent_0_stop'] = res['latent_start'] + p_phase['Latent_0']['len'] - dt.timedelta(seconds=1)


    # 5. Define Latent Stop (Chronic Seizure Logic)
    # -----------------------------------------------------------
    # Logic: The latent phase ends at the first "Chronic" seizure.
    # Definition: A chronic seizure is any seizure occurring >= 7 days post-stimulation.
    
    acute_cutoff_date = res['stimulation_stop'] + dt.timedelta(days=7)
    
    # Filter for seizures that meet the chronic definition
    # Note: We compare .date() to ensure we aren't tripped up by time-of-day
    bool_chronic_seizure = (
        (df[p_columnnames['date']] > acute_cutoff_date.date()) & 
        (~df['Seizure'].isnull())
    )
    df_chronic = df[bool_chronic_seizure]

    if not df_chronic.empty:
        # Found a chronic seizure. Latent period ends at the START of that day.
        first_chronic_date = df_chronic.iloc[0][p_columnnames['date']]
        res['latent_stop'] = dt.datetime.combine(first_chronic_date, dt.time.min)
    else:
        # No chronic seizures found (animal never seized or only had acute seizures)
        res['latent_stop'] = np.timedelta64('NaT')


    # 6. Define Latent 1 & Latent 2 (Retrospective Calculation)
    # -----------------------------------------------------------
    # These depend on the existence of a 'latent_stop'
    
    if not pd.isnull(res['latent_stop']):
        
        # Latent 2: Defined as the X days (usually 3) BEFORE the first chronic seizure
        res['Latent_2_stop'] = res['latent_stop'] - dt.timedelta(seconds=1)
        res['Latent_2_start'] = res['latent_stop'] - p_phase['Latent_2']['len']

        # Latent 1: Defined as the gap between Latent 0 and Latent 2
        # Only created if the gap is larger than the required Latent 1 length.
        
        l0_stop = res['Latent_0_stop']
        l2_start = res['Latent_2_start']
        
        # Check that Latent 2 doesn't technically start before Latent 0 ended
        # (This can happen if the first chronic seizure is very early, e.g., Day 8)
        if l2_start > l0_stop:
            gap_duration = l2_start - l0_stop
            required_l1_len = p_phase['Latent_1']['len']
            
            if gap_duration > required_l1_len:
                # Center Latent 1 in the available gap
                gap_center = l0_stop + (gap_duration / 2)
                half_len = required_l1_len / 2
                
                res['Latent_1_start'] = gap_center - half_len
                res['Latent_1_stop'] = gap_center + half_len - dt.timedelta(seconds=1)


        # 7. Define Chronic Phases (Sequential)
        # -----------------------------------------------------------
        # Chronic 0 starts exactly when Latent period stops (first chronic seizure day)
        
        current_start = res['latent_stop']
        
        # Iterate through phases 0, 1, 2 dynamically to reduce code repetition
        for i in range(3):
            phase_name = f'Chronic_{i}'
            
            # Safety check: does config exist?
            if phase_name not in p_phase:
                continue

            phase_len = p_phase[phase_name]['len']
            phase_stop = current_start + phase_len - dt.timedelta(seconds=1)

            # Only assign if the phase fits within the recording limits
            if (current_start + phase_len) < res['recording_last']:
                res[f'{phase_name}_start'] = current_start
                res[f'{phase_name}_stop'] = phase_stop
                
                # Update start for the next phase (immediately follows previous stop)
                current_start = phase_stop + dt.timedelta(seconds=1)
            else:
                # If Chronic_0 doesn't fit, Chronic_1 definitely won't. Break loop.
                break

    return res

def match_condition_on_row(df, p):
    """
    Determine rows that match conditions p
    
    Usage:
    Extract a certain subset of rows. For example,
    those corresponding to three days after a surgery.
    
    Example for p:
    p = {        
        'col':'Surgery',   # column to look at
        'ref':'x',         # reference value
        'ord': argmax,     # which numpy function to decide
                           # between multiple values matching ref
        'ord_col':'datetime',  # values in this column
        'wndw':[24, 96],   # window, here hours
        'wndw_col': 'hours_since_start', # column to evaluate window on
    }

    """
    df_ref = df[df[p['col']] == p['ref']]
    assert hasattr(np, p['ord'])
    if len(df_ref)>0:
        pos = getattr(np, p['ord'])(df_ref[p['ord_col']])
    else:
        return np.zeros(len(df), dtype=bool)
    
    # get reference row
    row_ref = df_ref.iloc[pos]
    # get reference time
    assert 'hours_since_start' in df.keys()
    hours_ref = row_ref['hours_since_start']
    
    # get time difference to all rows
    hours_delta = df[p['wndw_col']]-hours_ref
    
    # retrieve bool whithin time window.
    bool_delta = (hours_delta>=p['wndw'][0]) & (hours_delta<p['wndw'][1])
    return bool_delta


def determine_start_end_of_blobs(a, min_len=None):
    m = np.concatenate(( [True], ~a, [True] ))  # Mask
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits
    if min_len:
        ss_bool = (ss[:,1] - ss[:,0]) >= min_len
        ss = ss[ss_bool]
        # if no value is present modify output
        if ~np.any(ss_bool):
            ss = ss.flatten()
    return ss
