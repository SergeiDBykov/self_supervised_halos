from utils.utils import rep_path, data_path, is_freya, sim_path
from utils.data.tng_query import HaloInfo, subhalos_df


import numpy as np
import h5py
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pickle

import illustris_python as il


#subhalosIDs = list(subhalos_df.sample(25).index.values)
subhalosIDs = list(subhalos_df.index.values)


for subhaloID in tqdm(subhalosIDs):
    halo_lowmass = HaloInfo(subhaloID)
    dens = halo_lowmass.make_3d_density(
                        box_half_size = -5,
                        grid_bins = 64,
                        smooth_R = 5,
                        smooth_type = 'Gaussian',
                        clip_smoothed = 1e-5)

    hist = dens['hist']
    savepath = data_path+'/freya/'

    proj_yz = dens['projections']['yz']
    proj_xz = dens['projections']['xz']
    proj_xy = dens['projections']['xy']


    np.savez(savepath+f'halo_{subhaloID}_hist.npz',
                                        hist=hist,
                                        proj_yz = proj_yz,
                                        proj_xz = proj_xz,
                                        proj_xy = proj_xy,)




#### 14 JUNE 2025: approx 1h11m for 64 bins and all 16544, disk usage du -sh freya/ = 18gb
#tarred tar czf histograms_freya_14june.tar.gz freya/ -> disk usage, approx 10-20 minm 353 mb
