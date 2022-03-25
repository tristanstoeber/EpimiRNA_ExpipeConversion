import numpy as np
import pandas as pd
import pdb

def test_consistency_of_annotation_table(t):
    """A collection of tests to make sure that the annotation
    table has no obvious mistakes
    
    Parameters:
    -----------
    t : pandas Dataframe
    
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
    # Shared channels between animals are not allowed to
    # overlap on a given day.
    gpb_ID = t.groupby(['Date', 'Channel', 'Machine'])
    for n_i, gp_i in gpb_ID:
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
            for key, val in attr['data'].items():
                dct['data_path_' + key] = action.data_path(key)
        if 'info' in action.modules.keys():     
            dct = {**dct, **dict(action.modules['info'])}
        ls_actions.append(dct)
    df = pd.DataFrame(ls_actions)
    return df