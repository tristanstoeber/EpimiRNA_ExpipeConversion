import numpy as np
import pandas as pd
import pdb
import pickle
import datetime as dt

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

def define_epg_phases(df, p_phase, p_columnnames):
    
    df = df.sort_values(p_columnnames['date'])

    idx = df['ID'].unique()[0]
    
    res = {}

    # get end of recording
    date_last = df[p_columnnames['date']].max()
    res['recording_last'] = date_last


    # define boundaries of stimulation
    # --------------------------------
    # get all entries with a mark at Stimulation
    bool_stim = ~df[p_columnnames['pps_stim']].isnull()
    # get rows with stimulation
    df_stim = df[bool_stim]

    # get first day of stimulation
    if not np.any(bool_stim):
        print(
            'Animal: ' + str(idx) + 
            ' - No stimulation information found.' +
            ' Can not associated labels. Skip')
        return None
    else:
        # get datestring 
        date_start = df_stim[p_columnnames['date']].min()
        # take beginning of the day as start timepoint
        res['stimulation_start'] = dt.datetime.combine(date_start, dt.time.min)

        # get last day of stimulation
        date_stop = df_stim[p_columnnames['date']].max()
        # take end of the day as stop timepoint
        res['stimulation_stop'] = dt.datetime.combine(date_stop, dt.time.max)


        # define baseline phases
        #-----------------------

        # get day of surgery
        bool_surg = ~df[p_columnnames['surgery']].isnull()
        df_surg = df[bool_surg]
        assert len(df_surg) == 1
        date_surg = dt.datetime.combine(df_surg.iloc[0][p_columnnames['date']], dt.time.min)
        res['surgery'] = date_surg

        # get day after surgery
        date_aftersurg = date_surg + dt.timedelta(days=1)

        # baseline 0
        res['BL_0_start'] = dt.datetime.combine(date_aftersurg, dt.time.min)
        res['BL_0_stop'] = date_aftersurg+p_phase['BL_0']['len']-dt.timedelta(seconds=1)

        # baseline 1
        res['BL_1_start'] = res['stimulation_start'] - p_phase['BL_1']['len']
        res['BL_1_stop'] = res['stimulation_start'] - dt.timedelta(seconds=1)

        # define EPG phases
        # ---------------------

        # find rows with dates after stimulation
        row_ids_afterstim = []
        for i, row_i in df.iterrows():
            date_i = dt.datetime.combine(row_i[p_columnnames['date']], dt.time.min)
            if date_i > res['stimulation_stop']:
                row_ids_afterstim.append(i)

        # determine seizure free rows after stimulation
        df_afterstim = df.loc[row_ids_afterstim]

        # get seizure free entries
        bool_noseizure = df_afterstim['Seizure'].isnull()

        min_len = p_phase['Criteria_LatentPhase']['min_len']
        if isinstance(min_len, dt.timedelta):
            min_len = min_len.days
        ls_size_noseizure = determine_start_end_of_blobs(
            bool_noseizure, min_len=min_len)

        if len(ls_size_noseizure.shape) >= 2:
            # get start of latent period
            pos_start = ls_size_noseizure[0, 0]
            date_start = dt.datetime.combine(
                df_afterstim.iloc[pos_start][p_columnnames['date']],
                dt.time.min)
            # take beginning of the day as start timepoint
            res['latent_start'] = date_start

            # get stop of latent period
            # only if a new seizure happen before recording ends, can we assign the 
            # end of the latent period, latent 1,2 and chronic 0-2.
            pos_stop = ls_size_noseizure[0, 1]
            if pos_stop < len(df_afterstim):
                date_stop = (
                    df_afterstim.iloc[pos_stop][p_columnnames['date']] +
                    dt.timedelta(days=1))
                # take end of the day as end timepoint
                res['latent_stop'] = date_stop
            else:
                res['latent_stop'] = np.timedelta64('NaT')

            # define latent_0
            res['Latent_0_start'] = res['latent_start']
            res['Latent_0_stop'] = (
                res['latent_start'] +
                p_phase['Latent_0']['len'] -
                dt.timedelta(seconds=1))

            if not pd.isnull(res['latent_stop']):

                # define latent 2
                res['Latent_2_start'] = (
                    res['latent_stop'] - p_phase['Latent_2']['len'])
                
                    
                res['Latent_2_stop'] = res['latent_stop'] - dt.timedelta(seconds=1)

                # define latent_1, centered between latent 0 and latent 2
                # but only if distance between latent_0 and latent_2 is large enoug
                if res['Latent_2_start']-res['Latent_0_stop'] > p_phase['Latent_2']['len']:
                    latent_half = (res['latent_stop']-res['latent_start'])/2.
                    len_half = p_phase['Latent_1']['len']/2
                    res['Latent_1_start'] = res['latent_start'] + latent_half - len_half
                    res['Latent_1_stop'] = (
                        res['latent_start'] + latent_half + len_half - dt.timedelta(seconds=1))

                # define chronic 0
                if ((res['latent_stop'] +
                     p_phase['Chronic_0']['len']) <
                    res['recording_last']):
                        res['Chronic_0_start'] = res['latent_stop']
                        res['Chronic_0_stop'] = (
                            res['latent_stop'] +
                            p_phase['Chronic_0']['len'] -
                            dt.timedelta(seconds=1))

                # define chronic_1
                if ((res['latent_stop'] +
                     p_phase['Chronic_0']['len'] +
                     p_phase['Chronic_1']['len']) <
                    res['recording_last']):
                        res['Chronic_1_start'] = (
                            res['latent_stop'] +
                            p_phase['Chronic_0']['len'])
                        res['Chronic_1_stop'] = (
                            res['latent_stop'] +
                            p_phase['Chronic_0']['len'] +
                            p_phase['Chronic_1']['len'] -
                            dt.timedelta(seconds=1))

               # define chronic_2
                if ((res['latent_stop'] +
                     p_phase['Chronic_0']['len'] +
                     p_phase['Chronic_1']['len'] +
                     p_phase['Chronic_2']['len']) <
                    res['recording_last']):
                        res['Chronic_2_start'] = (
                            res['latent_stop'] +
                            p_phase['Chronic_0']['len']+
                            p_phase['Chronic_1']['len'])
                        res['Chronic_2_stop'] = (
                            res['latent_stop'] +
                            p_phase['Chronic_0']['len'] +
                            p_phase['Chronic_1']['len'] +
                            p_phase['Chronic_2']['len'] -
                            dt.timedelta(seconds=1))
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
