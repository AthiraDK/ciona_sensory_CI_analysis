#!/usr/bin/env python
# coding: utf-8

# # Preprocessing transmission files from mesmerise


# In[99]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mesmerize import Transmission
from glob import glob
from scipy import signal


# In[100]:


transmission_files = glob("/data/temp/jorgen/palp_proj_2019/trns/trns_for_panels/**/*.trn", recursive=True)


# ### Preprocessing
# We will need to process the transmission files we will work with quite a bit, so this section of the notebook will detail the required functions and operations we will apply to each transmission file.
# 
# #### Stim start and end indices.
# Some of our code downstream depends on the existence of ```_ST_START_IX``` and ```_ST_END_IX``` columns, but those do not yet exist in the given transmission files. For this we need to go to the ```stim_maps``` column. This column is for some reason filled with nested lists that at then end have a dictionary that we are interested in. We can not tell in advance how many nested lists there are before we get to the appropriate dictionary, so we need a function that can unpack these for us, that we can later apply to the whole column.

# In[101]:


def unpack_nested_list(x):
    '''recursively unpacks a list untill the length is longer than 1 or the datatype is no longer list'''
    if type(x) != list or len(x) > 1:
        return x
    else:
        x = x[0]
        return unpack_nested_list(x)


# Then we need to look into the dictionary that is in the meta column. For some reason this dictionary is stuctured like this:
# ```python
#     {"odor": None, 
#      "poke": pd.core.frame.DataFrame}
# ```
# The problem is that we need to know, per dataframe we work with, if we need to grab the odor or the poke entry. We can do this by just checking if the one of them is ```NoneType``` and if it is grab the other one.

# In[102]:


def get_stim_start(x):
    '''returns stimulus start and end indices as a numpy array'''
    if type(x["poke"]) != type(None):
        meta = x["poke"]
    else:
        meta = x["odor"]
    start = meta["start"].values[0]
    stop = meta["end"].values[0]
    return np.array([start, stop])


# The reason we return this as a numpy array is that we can retrieve the info for a whole column of stim_maps, and then just assign the new columns ```_ST_START_IX``` and ```_ST_END_IX``` by vstacking the results and using the [:,0] and [:,1].

# #### Fixing zfill on stimulus names for poking dataframes.
# 
# In dataframes with mulitple poke stimuli, there are stimulus ids that look like "poke_1_on" and "poke_11_on". The poke number is not zfilled to equal amounts of digits and this ruins plotting orders down the line. We need a function to fix this that we can apply to the appropriate columns.

# In[103]:


def fix_zfill(x):
    """zfills numeric parts of a string separated by underscores."""
    split = x.split("_")
    new_split = [x.zfill(2) if x.isnumeric() else x for x in split]
    return "_".join(new_split)

# ### Background subtraction
#
# Given the new dataset, we need to subtract the signals from the background. This is more like a baseline correction. For each of the signal recordings, we need to identify the row in which Jorgen has stored the 'measured background' for that recording. A signal and its corresponding background would have the same `SampleID`. 

# Note: the idea is to subtract the mean of the average from the signal (I guess the variations in the background are assumed to be noise and hence can be averaged)



def _subtract_bg(group):
    try:
        mean_bg = np.mean(group[group['cell_name']=='bg']['_RAW_CURVE'].values[0])
        group['_BGSUB_CURVE'] = group['_RAW_CURVE'] - mean_bg
        group['bg_missing'] = False
    except IndexError:
        group['_BGSUB_CURVE'] = group['_RAW_CURVE'] - group['_RAW_CURVE'] 
        group['bg_missing'] = True
    return group

def correct_bg(t):
    t.df = t.df.groupby('SampleID').apply(lambda g: _subtract_bg(g))
    return t

# ### DF/F, trimming and resampling
# 
# Some of these things I do because they need doing, some of these things I do because kushal did them before me.
# 
# Calculating the DF/F is the starting point. We also define a trim function because kushal added columns where curves where trimmed to n seconds after stim start. These are later used to take maxima from to have as a measurement of response size.

# In[164]:


# trim the curve
def trim(curve: np.ndarray, start_ix: int, trim_length: int):
    i = start_ix
    j = start_ix + trim_length
    return curve[i:j]

def calculate_dff(t, col = "_RAW_CURVE"):
    """calculates the df over f in transmission object t using the pre-stim mean"""
    # Use the pre-stimulus mean to set Fo
    t.df['_pre_stim_mean'] = t.df.apply(lambda row: row[col][:row._ST_START_IX].mean(), axis = 1)

    # Calculate ΔF/Fo: Curve minus Fo divided by Fo:
    t.df['_dfof'] = (t.df[col] - t.df['_pre_stim_mean'])/ t.df['_pre_stim_mean']
    
    dfof_trims = [5, 10, 30, 60, 9999]
    for n in dfof_trims:
        t.df[f'_dfof_{n}s'] = t.df.apply(lambda r: trim(r['_dfof'], r['_ST_START_IX'], int(n*r['meta']['fps'])), axis=1)   
    
    return t


# then we need a function to resample all the traces to a common sampling rate. Kushal used 10Hz:

# In[105]:


def _resample(curve: np.ndarray, fps: float, row):
    new_rate = 10.0 # set this to your favorite sampling rate, but I think 10 Hz is good enough
    Nf = fps
    Ns = curve.shape[0]
    Rn = int((Ns / Nf) * (new_rate))
    return signal.resample(curve, Rn)

def resample(t):
    #apply the resampling to 10Hz for the trimmed and df/f values
    dfof_trims = [5, 10, 30, 60, 9999]
    for n in dfof_trims:
        t.df[f'_resampled_{n}s'] = t.df.apply(lambda row: _resample(row[f'_dfof_{n}s'], row.meta['fps'], row), axis=1)

    t.df['_resampled'] = t.df.apply(lambda row: _resample(row[f'_dfof'], row.meta['fps'], row), axis=1)
    return t
    


# Lastly we can calculate the maxima of the df_f trims to have a single parameter to see response size with.

# In[106]:


def calculate_maxima(t):
    dfof_trims = [5, 10, 30, 60, 9999]
    for n in dfof_trims:
        t.df[f'maxima_{n}s'] = t.df[f'_dfof_{n}s'].apply(lambda x: x.max())
    return t


# ### Remove bg rows
def remove_bg_rows(t):
    t.df = t.df[t.df['cell_name']!='bg']
    return t

# ### Remove bg rows
def remove_if_no_bg(t):
    t.df = t.df[t.df['bg_missing']==False]
    return t

# ### One function to rule them all
# Now we combine all the above functions and create a function we call that does the preprocessing of the dataframe we load for us, and return something we can start plotting with:

# In[167]:


def preprocess_transmission(path, col = "_RAW_CURVE"):
    
    #load transmission file
    t = Transmission.from_hdf5(path)
    
    #drop incomplete entries
#     t.df = t.df.dropna(subset = [col])

    #eliminate traces where there is no good framerate listed
    t.df["fps"] = t.df["meta"].apply(lambda x: float(x["fps"]))
    t.df = t.df[t.df["fps"] > 0.1]
    
    #deal with the stim_maps column:
    t.df["stim_maps"] = t.df["stim_maps"].apply(unpack_nested_list)
    
    indices = np.vstack([get_stim_start(s) for s in t.df["stim_maps"]])
    t.df["_ST_START_IX"] = indices[:,0]
    t.df["_ST_END_IX"] = indices[:,1]
    
    t.df["_ST_START_IX"] = t.df["_ST_START_IX"].astype(int)
    t.df["_ST_END_IX"] = t.df["_ST_END_IX"].astype(int)
    
    
    #figure out if we are dealing with pokes, and if so fix the zfill issue:
    if "poke" in path.lower():
        t.df["poke"] = t.df["poke"].apply(unpack_nested_list)
        t.df["poke"] = t.df["poke"].apply(fix_zfill)
    
    # NEW: correct for the backgroud
    t = correct_bg(t)
#     t = remove_if_no_bg(t) # remove the recordings for which the background info is missing
    #do the df/f and resampling calculations
    t = calculate_dff(t, col = '_BGSUB_CURVE')
    t = remove_bg_rows(t)
    t = resample(t)
    t = calculate_maxima(t)
    
    return t
    
    






