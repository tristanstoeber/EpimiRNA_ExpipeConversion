import numpy as np
import datetime
 
dict_format = {
    'Date' : datetime.date,
    'ID' : int,
    'group': str,
    'Machine' : str,
    'Channel' : int,
    'Notes' : str,
    'EEG manually inspected' : str,
    'PPS Stimulation' : str,
    'Artifacts reported' : str,
    'Surgery' : str,
    'Low confidence Seizure' : str,
    'Seizure' : str,
    'Missing data' : str,
    'Conflict overlap recording': str,
    'Device on' : str,
    'Device off' : str,
    'Spikes' : str,
    'Spindles' : str,
    'DBS location' : str,
    'DBS frequency' : str,
    'DBS stimulation' : str}