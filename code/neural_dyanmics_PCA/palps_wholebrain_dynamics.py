import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns

import ipywidgets as widgets
from IPython.display import display

from sklearn.decomposition import PCA
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from datetime import date


def zscore_rawcurves(sample_df):
    cell_types = list(sample_df["cell_type"])
    temporal = np.vstack(sample_df["_RAW_CURVE"].values)
    zscored = TimeSeriesScalerMeanVariance().fit_transform(temporal)[..., 0]
    df_zscored = pd.DataFrame(zscored.T, columns= cell_types)
    return df_zscored

def get_PCA_results(df_zscored):
    zscored = df_zscored.values
    pca = PCA(n_components=3)
    results_pca = pca.fit_transform(zscored)
    loadings = pca.components_
    df_loadings = pd.DataFrame(loadings, columns = df_zscored.columns)
    return results_pca, df_loadings

def generate_zscore_clustermaps(df_zscored, sample_id, savepath):
    scale_cg1 = (1*len(df_zscored.columns))/2
    fig1 = sns.clustermap(data=df_zscored.T, col_cluster=False,figsize=(15,scale_cg1),
                         colors_ratio=(0.1, 0.01))
    fig1.savefig(os.path.join(savepath, f'clustermap_zscored_{sample_id}.svg'))
    return None
    

def generate_pca_loadings_clustermap(df_loadings, sample_id, savepath):
    scale_cg1 = (1*len(df_loadings.columns))/2
    fig2 = sns.clustermap(df_loadings.T, cmap="PiYG", col_cluster=False, figsize=(3, scale_cg1))
    fig2.savefig(os.path.join(savepath, f'clustermap_pca_loadings_{sample_id}.svg'))
    return None


def plot_pca_traj3d(results_pca, sample_id):
    N = results_pca.shape[0]
    T = np.ascontiguousarray(results_pca) * 10
    
    fig3 = plt.figure(figsize=(6,6))
    ax = fig3.add_subplot(projection='3d')
    
    for i in range(N):
        ax.plot(T[i-1:i+1, 0], T[i-1:i+1, 1], T[i-1:i+1, 2], 
                color=plt.cm.jet(i/N))
    
    
    ax.set_xlim(T[:,0].min()-2,T[:,0].max()+2)
    ax.set_ylim(T[:,1].min()-2,T[:,1].max()+2)
    ax.set_zlim(T[:,2].min()-2,T[:,2].max()+2)
    ax.view_init(elev=10., azim= 45.)

    x, y, z = np.zeros((3,3))

    colors = sns.color_palette("hls", 8)
    u1 = ax.quiver(x, y, z, T[:,0].max()+2,0,0, color=colors[0], arrow_length_ratio=0)
    v1 = ax.quiver(x, y, z, 0,T[:,1].max()+2,0, color=colors[2], arrow_length_ratio=0)
    w1 = ax.quiver(x, y, z, 0,0,T[:,2].max()+2, color=colors[5], arrow_length_ratio=0)
    
    ax.plot([], [], [], c=colors[0], label="PC1")
    ax.plot([], [], [], c=colors[2], label="PC2")
    ax.plot([], [], [], c=colors[5], label="PC3")

    ax.legend()
    ax.set_axis_off()
    

    fig3.canvas.manager.set_window_title(sample_id)
    plt.show()


if __name__ == "__main__":

    pd.options.mode.chained_assignment = None

    # Load the data into a pandas DataFrame object
    data_path = './root_df_20102022.trn' # TODo: change the path to match the repo dir structure
    df = pd.read_hdf(data_path)

    # Create a directory to save plots generated
    dest_folder = './results' # TODo: change the path to match the repo dir structure
    today = date.today()
    date_tag = today.strftime("%d%b%Y")
    dest_folder = f'{dest_folder}_{date_tag}'
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)


    sample_ids = list(df.SampleID.unique())
    for sid in sample_ids:
        sample_df = df[df["SampleID"] == sid]
        cell_types = list(sample_df["cell_type"])

        df_zscored = zscore_rawcurves(sample_df)
        generate_zscore_clustermaps(df_zscored, sid, dest_folder)

        results_pca, df_loadings = get_PCA_results(df_zscored)
        generate_pca_loadings_clustermap(df_loadings, sid, dest_folder)

        plot_pca_traj3d(results_pca, sid)










