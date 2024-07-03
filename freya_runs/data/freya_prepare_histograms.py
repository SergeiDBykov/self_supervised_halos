import self_supervised_halos.utils.tng as tng
HaloInfo = tng.HaloInfo
subhalos_df = tng.subhalos_df

import self_supervised_halos.utils.utils as utils
rep_path, data_path, is_freya, sim_path = utils.rep_path, utils.data_path, utils.is_freya, utils.sim_path


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
subhalosIDs = list(subhalos_df.index.values)[::-1]


for subhaloID in tqdm(subhalosIDs):
    halo_lowmass = HaloInfo(subhaloID)
    dens = halo_lowmass.make_3d_density(
                        box_half_size = -5,
                        grid_bins = 64,)

    savepath = data_path+'/freya/'

    hist = dens['hist']
    proj_yz = dens['projections']['yz']
    proj_xz = dens['projections']['xz']
    proj_xy = dens['projections']['xy']

    edges = dens['edges']
    edge_binsize = dens['edge_binsize']
    box_half_size = dens['box_half_size']
    half_mass_rad = dens['half_mass_rad']
    is_in_units_of_halfmassrad = dens['is_in_units_of_halfmassrad']


    np.savez(savepath+f'halo_{subhaloID}_hist.npz',
                                        hist=hist,
                                        proj_yz = proj_yz,
                                        proj_xz = proj_xz,
                                        proj_xy = proj_xy,
                                        edges = edges,
                                        edge_binsize = edge_binsize,
                                        box_half_size = box_half_size,
                                        half_mass_rad = half_mass_rad,
                                        is_in_units_of_halfmassrad = is_in_units_of_halfmassrad,
                                        )


#cd to data/freya and tar the files tar czf histograms_freya_14june.tar.gz freya/ 
os.chdir(data_path+'/')
print('Zipping the files...')
os.system('tar czf histograms_freya.tar.gz freya/') #10-20 min, 353 mb
print('Done')
