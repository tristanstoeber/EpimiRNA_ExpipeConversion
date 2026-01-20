import pytest
import pandas as pd
import numpy as np
import datetime as dt
from tools import define_epg_phases  # Assumes the function is saved here

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
    
@pytest.fixture
def p_phase():
    """Standard phase lengths configuration."""
    return {
        'BL_0': {'len': dt.timedelta(days=2)},
        'BL_1': {'len': dt.timedelta(days=2)},
        'Latent_0': {'len': dt.timedelta(days=3)},
        'Latent_1': {'len': dt.timedelta(days=3)},
        'Latent_2': {'len': dt.timedelta(days=3)},
        'Chronic_0': {'len': dt.timedelta(days=7)},
        'Chronic_1': {'len': dt.timedelta(days=7)},
        'Chronic_2': {'len': dt.timedelta(days=7)},
    }

@pytest.fixture
def p_columnnames():
    """Column name mapping."""
    return {
        'date': 'Date',
        'pps_stim': 'PPS',
        'surgery': 'Surgery',
        'id': 'ID'
    }

@pytest.fixture
def base_df():
    """
    Creates a basic 60-day dataframe starting from 2023-01-01.
    Columns: Date, ID, PPS, Surgery, Seizure.
    """
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'ID': 'Rat_001',
        'PPS': [np.nan] * 60,
        'Surgery': [np.nan] * 60,
        'Seizure': [np.nan] * 60
    })
    # Default events
    df.loc[0, 'Surgery'] = 'x'     # Day 0: Surgery
    df.loc[5, 'PPS'] = 'x'         # Day 5: Stim Start
    df.loc[6, 'PPS'] = 'x'         # Day 6: Stim End (Stimulation Stop = 2023-01-07 23:59:59)
    return df


# -------------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------------

def test_happy_path(base_df, p_phase, p_columnnames):
    """
    Scenario: Long latent period (20 days).
    Expectation: All phases (L0, L1, L2, C0, C1, C2) generated successfully.
    """
    # Stim ends Day 6. Latent Start = Day 7.
    # Chronic Seizure on Day 30 (well past the 7-day cutoff)
    base_df.loc[30, 'Seizure'] = 'x'
    
    res = define_epg_phases(base_df, p_phase, p_columnnames)
    
    assert res is not None
    assert 'Latent_0_start' in res
    assert 'Latent_1_start' in res  # Gap is big enough (Day 7 to 30)
    assert 'Latent_2_start' in res
    assert 'Chronic_0_start' in res
    
    # Check Latent Stop accuracy (should be start of Day 30)
    expected_stop = dt.datetime.combine(base_df.loc[30, 'Date'], dt.time.min)
    assert res['latent_stop'] == expected_stop


def test_acute_seizure_only(base_df, p_phase, p_columnnames):
    """
    Scenario: Seizure occurs on Day 9 (3 days post-stim).
    Logic: This is < 7 days post-stim, so it is ACUTE.
    Expectation: 
    - Latent_0 exists.
    - Latent_Stop is NaT (no chronic seizure).
    - Latent_2, Chronic_0 should NOT exist.
    """
    # Stim ends Day 6. Cutoff is Day 6 + 7 = Day 13.
    # Seizure on Day 9 (Acute)
    base_df.loc[9, 'Seizure'] = 'x'
    
    res = define_epg_phases(base_df, p_phase, p_columnnames)
    
    assert 'Latent_0_start' in res
    assert pd.isnull(res['latent_stop'])
    assert 'Latent_2_start' not in res
    assert 'Chronic_0_start' not in res


def test_chronic_boundary_exact(base_df, p_phase, p_columnnames):
    """
    Scenario: Seizure exactly on the boundary.
    Stim Ends: Day 6 (Jan 7).
    Cutoff: Jan 7 + 7 days = Jan 14.
    Seizure on Jan 14 (Day 13) -> Should be Acute (ignored).
    Seizure on Jan 15 (Day 14) -> Should be Chronic (trigger).
    """
    # Case A: Seizure on Day 13 (Acute)
    df_acute = base_df.copy()
    df_acute.loc[13, 'Seizure'] = 'x'
    res_a = define_epg_phases(df_acute, p_phase, p_columnnames)
    assert pd.isnull(res_a['latent_stop']) # Should be ignored

    # Case B: Seizure on Day 14 (Chronic)
    df_chronic = base_df.copy()
    df_chronic.loc[14, 'Seizure'] = 'x'
    res_b = define_epg_phases(df_chronic, p_phase, p_columnnames)
    assert not pd.isnull(res_b['latent_stop']) # Should trigger
    assert res_b['latent_stop'].date() == df_chronic.loc[14, 'Date'].date()


def test_short_latency_no_L1(base_df, p_phase, p_columnnames):
    """
    Scenario: Chronic seizure happens early (Day 15).
    Latent Start: Day 7.
    Latent Stop: Day 15.
    Gap: 8 days.
    L0 (3 days) + L2 (3 days) = 6 days used.
    Remaining Gap = 2 days.
    L1 Required = 3 days.
    Expectation: L0 and L2 exist, but L1 is skipped because gap (2) < required (3).
    """
    base_df.loc[15, 'Seizure'] = 'x'
    
    res = define_epg_phases(base_df, p_phase, p_columnnames)
    
    assert 'Latent_0_start' in res
    assert 'Latent_2_start' in res
    assert 'Latent_1_start' not in res  # KEY ASSERTION


def test_overlap_L0_L2(base_df, p_phase, p_columnnames):
    """
    Scenario: Very early chronic seizure (Day 14).
    Latent Start: Day 7.
    Latent Stop: Day 13 (Assuming seizure on 13 was chronic for this test logic, 
    but based on our rules, day 14 is the first valid chronic day).
    
    Let's use Day 14 seizure.
    Latent window: Day 7 to Day 14 (7 days total).
    L0: Day 7, 8, 9.
    L2: Day 11, 12, 13.
    This fits without overlap (gap of 1 day). 
    
    Let's try creating a logical overlap by shortening the gap manually or lengthening phases.
    We will modify p_phase temporarily to force overlap.
    L0 = 4 days, L2 = 4 days. Total needed 8. Available 7.
    Expectation: Code runs without crashing, L2 might start before L0 ends mathematically.
    """
    p_phase['Latent_0']['len'] = dt.timedelta(days=4)
    p_phase['Latent_2']['len'] = dt.timedelta(days=4)
    
    # Seizure on Day 14 (First valid chronic day)
    base_df.loc[14, 'Seizure'] = 'x'
    
    res = define_epg_phases(base_df, p_phase, p_columnnames)
    
    # L0: Ends Day 7+4 = Day 11 (approx)
    # L2: Starts Day 14-4 = Day 10 (approx)
    # They overlap. The function allows this (L1 is skipped).
    assert 'Latent_0_start' in res
    assert 'Latent_2_start' in res
    assert 'Latent_1_start' not in res
    
    # Check timestamps: L2 Start should be < L0 Stop
    assert res['Latent_2_start'] < res['Latent_0_stop']


def test_missing_recording_tail(base_df, p_phase, p_columnnames):
    """
    Scenario: Animal dies/recording ends shortly after first chronic seizure.
    Seizure: Day 30.
    Recording End: Day 32.
    Chronic Phase 0 length: 7 days.
    Expectation: Chronic_0 is NOT created because it doesn't fit in the remaining 2 days.
    """
    # Cut dataframe short
    df_short = base_df.iloc[:33].copy() # Ends on Day 32
    df_short.loc[30, 'Seizure'] = 'x'
    
    res = define_epg_phases(df_short, p_phase, p_columnnames)
    
    assert not pd.isnull(res['latent_stop'])
    # Chronic 0 needs 7 days, we only have 2.
    assert 'Chronic_0_start' not in res 


def test_no_stimulation(base_df, p_phase, p_columnnames):
    """Scenario: PPS column is empty."""
    base_df['PPS'] = np.nan
    res = define_epg_phases(base_df, p_phase, p_columnnames)
    assert res is None


def test_missing_surgery(base_df, p_phase, p_columnnames):
    """Scenario: Surgery column is empty. Should allow Latent phases but skip BL_0."""
    base_df['Surgery'] = np.nan
    base_df.loc[30, 'Seizure'] = 'x' # Valid chronic seizure
    
    res = define_epg_phases(base_df, p_phase, p_columnnames)
    
    assert res is not None
    assert 'BL_0_start' not in res
    assert 'Latent_0_start' in res # Should still work


def test_multiple_chronic_seizures(base_df, p_phase, p_columnnames):
    """
    Scenario: Seizures on Day 30, 31, 35.
    Expectation: Latent Stop is determined by the FIRST one (Day 30).
    """
    base_df.loc[30, 'Seizure'] = 'x'
    base_df.loc[31, 'Seizure'] = 'x'
    base_df.loc[35, 'Seizure'] = 'x'
    
    res = define_epg_phases(base_df, p_phase, p_columnnames)
    
    expected_stop = dt.datetime.combine(base_df.loc[30, 'Date'], dt.time.min)
    assert res['latent_stop'] == expected_stop