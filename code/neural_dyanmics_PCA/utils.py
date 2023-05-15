import numpy as np
import pandas as pd


def unpack_nested_list(x):
    '''recursively unpacks a list untill the length is longer than 1 or the datatype is no longer list'''
    if type(x) != list or len(x) > 1:
        return x
    else:
        x = x[0]
        return unpack_nested_list(x)
    
def unpack_stim_maps(x):
    'unpack the stim maps and return stim_type (usually the key of the dict returned by unpack_nested_list), stim_name stim_start  and stim_end'
    x = unpack_nested_list(x)
    stim_type = list(x.keys())[0]
    try:
        stim_name = x[stim_type]['name']
        stim_start = x[stim_type]['start']
        stim_end = x[stim_type]['end']
        return {'stim_type':stim_type, 'stim_name':stim_name, 'stim_start':stim_start, 'stim_end':stim_end}
    except Exception as e:
        print(e, x)
        
        

def calculate_dff(df, col = "_RAW_CURVE"):
    
    """calculates the df over f in transmission object t using the pre-stim mean"""
    # Use the pre-stimulus mean to set Fo
    df['_pre_stim_mean'] = df.apply(lambda row: row[col][:row._ST_START_IX].mean(), axis = 1)

    # Calculate ΔF/Fo: Curve minus Fo divided by Fo:
    df['_dfof'] = (df[col] - df['_pre_stim_mean'])/ df['_pre_stim_mean']

    return df



def preprocess_transmission(df, col = "_RAW_CURVE"):
    

    #eliminate traces where there is no good framerate listed
    df["fps"] = df["meta"].apply(lambda x: float(x["fps"]))
    df = df[df["fps"] > 0.1]
    
    # Manage stimulus data
    df['stim_maps'] = df['stim_maps'].apply(lambda x: unpack_stim_maps(x))
    df['stim_type'] = df['stim_maps'].apply(lambda x: x['stim_type'])

    df['_ST_START_IX'] = df['stim_maps'].apply(lambda x: int(x['stim_end'].min()))
    df['_ST_END_IX'] = df['stim_maps'].apply(lambda x: int(x['stim_start'].max()))

    df['_ST_START_sec'] = df.apply(lambda x: int(x['stim_maps']['stim_end'].min()) * x['fps'], axis=1)
    df['_ST_END_sec'] = df.apply(lambda x: int(x['stim_maps']['stim_start'].max()) * x['fps'], axis=1)
    
    #do the df/f and resampling calculations
    df = calculate_dff(df, col = col)
    return df