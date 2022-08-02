import numpy as np
import tools
import pandas as pd

def test_determine_start_end_of_blobs():
    x = np.array([True, False, True, True, True])
    
    # Test for different values of minlen
    ls_true = [
        np.array([
            [0, 1],
            [2, 5]]),
        np.array([
            [2, 5]]),
        np.array([])
    ]
    
    for i, min_len_i in enumerate([0, 3, 5]):
        y_true = ls_true[i]
        y_hat = tools.determine_startend_of_blobs(x, min_len=min_len_i)
        print(y_hat)
        assert np.array_equal(y_true, y_hat)


def test_define_epg_phases():
    # define criteria
    p_phase = {
       'Criteria_LatentPhase': {
           'min_len': dt.timedelta(days=3),  # minimal length
           'max_time_latent': dt.timedelta(days=55) # maximal length
       },
       'BL_0': {
           'len': dt.timedelta(days=3),
           'color': '#FDE725FF'},
       'BL_1': {
           'len': dt.timedelta(days=3),
           'color': '#C7E020FF'},
       'Latent_0': {
           'len': dt.timedelta(days=3),
           'color': '#8FD744FF'},
       'Latent_1': {
           'len': dt.timedelta(days=3),
           'color': '#75D054FF'},   
       'Latent_2': {
           'len': dt.timedelta(days=3),
           'color': '#47C16EFF'},
       'Chronic_0': {
           'len': dt.timedelta(days=3),
           'color': '#27AD81FF'},
       'Chronic_1': {
           'len': dt.timedelta(days=3),
           'color': '#1F9A8AFF'},
       'Chronic_2': {
           'len': dt.timedelta(days=3),
           'color': '#24868EFF'},
    }

    p_columnnames = {
        'date': 'Date',
        'pps_stim': 'PPS Stimulation',
        'surgery': 'Surgery'
    }
    
    # load data annotation
    t = pd.read_excel('data_annotation.ods', engine="odf", dtype=dict_format)
    
    # do test for each animal
    animals = t['ID'].unique()
    
    for animal_i in animals:

        t_i = t[t['ID']==animal_i]
        tes_i = tools.define_epg_phases(
            t_i,
            p_phase,
            p_columnnames)
        
        # assert all timepoints are equivalent to dt.datetime
        for val_i in tes_i.values():
            assert isinstance(val_i, dt.datetime)
    
    