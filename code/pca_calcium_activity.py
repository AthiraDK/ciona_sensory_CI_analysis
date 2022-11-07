#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from utils import *

from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from tvregdiff import TVRegDiff


def normalize_dfof(dfof):
    # Normalize dfof by the peak
    if np.nanmax(dfof) > np.abs(np.nanmin(dfof)):
        peak = np.nanmax(dfof)
    else:  # if negative trough is larger in magnitude
        peak = np.nanmin(dfof)
    norm_dfof = dfof / peak
    return norm_dfof


def get_derivative(dfof, scaling_factor):
    dev_dfof = TVRegDiff(
        dfof, 1, 2, diffkernel="sq", dx=0.1, plotflag=False)
    return dev_dfof * scaling_factor


def get_rising_points(dev_dfof):
    rise_args, _ = find_peaks(dev_dfof, height=(
        0.4*max(dev_dfof), max(dev_dfof)), distance=20)
    rise_args = rise_args - 1
    return rise_args


def get_suffix(cell_loc_list, new_loc):

    count_duplicates = len(
        [cell_loc for cell_loc in cell_loc_list if cell_loc.split('_')[1] == new_loc])
    suffix = count_duplicates
    return f'{suffix}_{new_loc}'


def process_data(df, sid):

    df_sid = df[df['SampleID'] == sid]
    df_sid.sort_values(by=['cell_loc_code'], inplace=True)

    ls_raw_curves = []
    ls_cell_locs = []
    ls_cell_types = []
    ls_curve_uuids = []
    ls_derivatives = []
    ls_rise_points = []

    for ind_, (ind_data, data_row) in enumerate(df_sid.iterrows()):

        if ind_ == 0:
            t_start = data_row['_ST_START_IX']
            t_end = data_row['_ST_END_IX']
            t_start_f = t_start - 50
            df_data_neurons = pd.DataFrame()

            tstart_ind = 50
            tend_ind = t_end - (t_start - 50)

        rc_orig = data_row['_RAW_CURVE']
        rc = rc_orig[t_start_f:t_start_f+300]
        max_rc = max(rc)

        cell_loc_suffix = get_suffix(ls_cell_locs, data_row['cell_loc'])

        dfof = data_row['_dfof'][t_start_f:t_start_f+300]
        norm_dfof = normalize_dfof(dfof)

        dev_norm = get_derivative(norm_dfof, scaling_factor=max_rc)
        if np.sum(np.isnan(dev_norm)) == 0:

            rise_args = get_rising_points(dev_norm)
            ls_raw_curves.append(rc)
            ls_cell_locs.append(cell_loc_suffix)
            ls_cell_types.append(data_row['cell_type'])
            ls_curve_uuids.append(data_row['uuid_curve'])
            ls_derivatives.append(dev_norm)
            ls_rise_points.append(rise_args)

            t_axis = np.arange(0, len(rc_orig)) - t_start
            df_data_neurons['t_axis'] = t_axis[t_start_f:t_start_f+300]
            df_data_neurons[cell_loc_suffix] = rc

    # Dataframe for plotting a heatmap
    df_data_neurons.set_index('t_axis', inplace=True)

    arr_derivatives = np.vstack(ls_derivatives)
    arr_derivatives = arr_derivatives / np.amax(arr_derivatives)

    # Perform PCA
    pca = PCA()
    result_pca = pca.fit_transform(arr_derivatives.T)

    # Max loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                            index=ls_cell_locs)
    argmax_comp = list(loadings[['PC1', 'PC2', 'PC3']].idxmax(axis=1).values)
    valmax_comp = list(loadings[['PC1', 'PC2', 'PC3']].max(axis=1).values)

    # Dataframe for metadata
    _loc_neurons = [cell_loc.split('_')[1] for cell_loc in ls_cell_locs]
    _suf_neurons = [cell_loc.split('_')[0] for cell_loc in ls_cell_locs]
    meta_dict = {'cell_loc': _loc_neurons, 'suffix_loc': _suf_neurons,
                 'cell_type': ls_cell_types, 'curve_uuid': ls_curve_uuids,
                 'max_pca_component': argmax_comp, 'max_value_pca': valmax_comp,
                 't_axis': [t_axis[t_start_f:t_start_f+300]]*len(_loc_neurons), 'raw_curve': ls_raw_curves}
    df_meta_neurons = pd.DataFrame(meta_dict)

    return {'sid': sid, 'data_heatmap': df_data_neurons, 'pc_components': result_pca,
            'start_ind': tstart_ind, 'end_ind': tend_ind, 'rise_points': ls_rise_points, 'cell_locs': ls_cell_locs,
            'df_meta': df_meta_neurons}


def plot_activity(df, dest_folder, save=True):
    fig1, ax1 = plt.subplots(1, 1)
    sns.heatmap(data=df.T, xticklabels=50,
                cmap='coolwarm', robust=True, ax=ax1)
    ax1.set_aspect(15)
    plt.yticks(rotation=0)
    if save:
        fig1.savefig(os.path.join(dest_folder, 'heatmap.svg'))
    else:
        plt.show()


def plot_pca_components(analysis_dict, dest_folder, save=True):
    result_pca = analysis_dict['pc_components']
    t_start = analysis_dict['start_ind']
    t_end = analysis_dict['end_ind']
    fig, ax = plt.subplots(1, 1)
    for i in range(4):
        ax.plot(result_pca[:, i])
    ax.axvline(t_start, c='k', ls=':')
    ax.axvline(t_end, c='k', ls=':')
    if save:
        fig.savefig(os.path.join(dest_folder,
                    'pca_components.svg'))
    else:
        plt.show()


def plot_pca_space(analysis_dict, dest_folder, dim=3, mark_peaks=True, save=True):
    result_pca = analysis_dict['pc_components']
    t_start = analysis_dict['start_ind']
    t_end = analysis_dict['end_ind']
    ls_rise_points = analysis_dict['rise_points']
    ls_cell_locs = analysis_dict['cell_locs']

    c_multi = sns.color_palette('tab10', 10)
    cell_dict1 = {'palp': c_multi[3], 'sv': c_multi[3], 'mg': c_multi[3],
                  'atena': c_multi[4], 'atenp': c_multi[6], 'rten': c_multi[-2]}
    cell_dict2 = {'palp': 'o', 'sv': '^', 'mg': 's',
                  'atena': '*', 'atenp': '*', 'rten': 's'}

    if dim == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')
        ax.plot3D(result_pca[:t_start, 0], result_pca[:t_start, 1], result_pca[:t_start, 2], label='pre',
                  c=c_multi[0], alpha=0.8)
        ax.plot3D(result_pca[t_start-1:t_end, 0], result_pca[t_start-1:t_end, 1], result_pca[t_start-1:t_end, 2],
                  label='stim', c=c_multi[1], alpha=0.8)
        ax.plot3D(result_pca[t_end-1:, 0], result_pca[t_end-1:, 1], result_pca[t_end-1:, 2], label='post',
                  c=c_multi[2], alpha=0.8)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.plot(result_pca[:t_start, 0], result_pca[:t_start,
                                                    1], label='pre', c=c_multi[0], alpha=0.5)
        ax.plot(result_pca[t_start-1:t_end, 0], result_pca[t_start -
                                                           1:t_end, 1], label='stim', c=c_multi[1], alpha=0.5)
        ax.plot(result_pca[t_end-1:, 0], result_pca[t_end -
                                                    1:, 1], label='post', c=c_multi[2], alpha=0.5)
        ax.axis('equal')
    w_rise = ''
    if mark_peaks:
        for i, rises in enumerate(ls_rise_points):
            rise_pca1_vals = [result_pca[t, 0] for t in rises]
            rise_pca2_vals = [result_pca[t, 1] for t in rises]
            rise_pca3_vals = [result_pca[t, 2] for t in rises]
            cell_loc_for_pca = ls_cell_locs[i].split('_')[1]
            rise_point_labels = [
                cell_dict1[cell_loc_for_pca] for t in rises]
            rise_point_markers = cell_dict2[cell_loc_for_pca]
            if dim == 3:
                ax.scatter(rise_pca1_vals, rise_pca2_vals, rise_pca3_vals,
                           c=rise_point_labels, label=cell_loc_for_pca, marker=rise_point_markers)
            else:
                ax.scatter(rise_pca1_vals, rise_pca2_vals,
                           c=rise_point_labels, label=cell_loc_for_pca, marker=rise_point_markers)
        w_rise = '_rises'

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    if dim == 3:
        ax.set_zlabel('PCA 3')

    # Fix legend
    hand, labl = ax.get_legend_handles_labels()
    handout = []
    lablout = []
    for h, l in zip(hand, labl):
        if l not in lablout:
            lablout.append(l)
            handout.append(h)
    fig.legend(handout, lablout)
    fname = f'{dim}d_phase_pca{w_rise}'
    fig.canvas.set_window_title(fname)

    if save:
        fig.savefig(os.path.join(dest_folder,
                    f'{fname}.svg'))
    else:
        plt.show()


def plot_pca_sorted(analysis_dict, dest_folder, save=True):
    df_meta = analysis_dict['df_meta']
    pca_sorted = pca_sorted = df_meta.sort_values(['max_pca_component', 'max_value_pca'],
                                                  ascending=[False, True])
    cpal = sns.color_palette('Set1')
    c_dict = {'PC1': cpal[0], 'PC2': cpal[1], 'PC3': cpal[2], }
    n_neurons = len(pca_sorted.index)
    fig, ax = plt.subplots(1, 1, figsize=(10, n_neurons))
    for i, (idx, row) in enumerate(pca_sorted.iterrows()):
        #         print(row['raw_curve'].shape)
        ax.plot(row['t_axis'], (row['raw_curve']/max(row['raw_curve'])
                                )+(i*1.5), c=c_dict[row['max_pca_component']])
    if save:
        fig.savefig(os.path.join(dest_folder,
                    'traces_ordered_pc123.svg'))
    else:
        plt.show()


def save_meta(analysis_dict, dest_folder, save=True):
    df_meta = analysis_dict['df_meta']
    meta_cols_to_save = ['cell_loc', 'suffix_loc', 'cell_type',
                         'curve_uuid', 'max_pca_component', 'max_value_pca']
    if save:
        df_meta[meta_cols_to_save].to_csv(os.path.join(
            dest_folder, 'meta_ordered.csv'), index_label='index')


if __name__ == "__main__":

    pd.options.mode.chained_assignment = None
    data_path = './data/root_df_20102022.trn'
    df = pd.read_hdf(data_path)

    # quickfix to order cell locations on the y-axis for heatmap
    cell_loc_code_dict = {'palp': 'a', 'rten': 'b',
                          'sv': 'e', 'mg': 'f', 'atena': 'c', 'atenp': 'd'}
    df['cell_loc_code'] = df['cell_loc'].apply(lambda x: cell_loc_code_dict[x])

    df_proc = preprocess_transmission(df)

    sample_ids = list(df_proc.SampleID.unique())

    for i in range(len(sample_ids)):
        sid = sample_ids[i]
        print(sid)

        # Set default folder to save plots through matplotlib gtk interface
        mpl.rcParams["savefig.directory"] = f'../results/{sid}/'
        mpl.rcParams["savefig.format"] = "svg"

        print('calculating...........')
        data_dict = process_data(df_proc, sid)

        # Create directory
        print('creating dir...........')
        sample_id = data_dict['sid']
        dest_folder = os.path.join(
            '/home/athira/SARS_Bergen/LD_dynamics_Jorgen/LD_dynamics_Jorgen/results/', sample_id)
        mpl.rcParams["savefig.directory"] = dest_folder
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        else:
            print(dest_folder)

        # Plotting
        print('plotting...........')
        save = False
        plot_activity(data_dict['data_heatmap'], dest_folder, save=save)
        plot_pca_components(data_dict, dest_folder, save=save)
        plot_pca_space(data_dict, dest_folder, dim=2,
                       mark_peaks=True, save=save)
        plot_pca_space(data_dict, dest_folder, dim=2,
                       mark_peaks=False, save=save)
        plot_pca_space(data_dict, dest_folder, dim=3,
                       mark_peaks=True, save=save)
        plot_pca_space(data_dict, dest_folder, dim=3,
                       mark_peaks=False, save=save)
        plot_pca_sorted(data_dict, dest_folder, save)
        save_meta(data_dict, dest_folder, save=save)
