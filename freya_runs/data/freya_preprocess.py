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


preprocess_path = data_path+'/freya_preprocess/'
save_3d = True

if not os.path.exists(preprocess_path):
    os.makedirs(preprocess_path)
    os.makedirs(preprocess_path+'/2d')
    os.makedirs(preprocess_path+'/3d')
    os.makedirs(preprocess_path+'/mass')


for id in tqdm(subhalos_df.index): #3 min for saving without 3d data, 4 min for saving with 3d data. On freya: ~20 min saving with 3d (not srun but jupyter)
    halo = HaloInfo(id)
    halo.make_3d_density()
    data_transform = halo.data_transform(dens = None, smooth=None)

    map_2d_xz = data_transform['map_2d_xz']
    map_2d_yz = data_transform['map_2d_yz']
    map_2d_xy = data_transform['map_2d_xy']

    snap = data_transform['snapshot']
    mass_hist = data_transform['mass_hist']

    map_3d = data_transform['map_3d']
    
    
    fname_root_2d = f'{preprocess_path}/2d/halo_{id}_2d'
    fname_root_3d = f'{preprocess_path}/3d/halo_{id}_3d'
    fname_root_mass = f'{preprocess_path}/mass/halo_{id}_mass'

    np.savez(fname_root_2d, 
                map_2d_xz = map_2d_xz,
                map_2d_yz = map_2d_yz,
                map_2d_xy = map_2d_xy,
    )
    
    np.savez(fname_root_3d, 
                    map_3d = map_3d)

    np.savez(fname_root_mass,
                snap = snap,
                mass_hist = mass_hist
    )


#cd to data/freya and tar the files tar czf histograms_freya_14june.tar.gz freya/ 
os.chdir(data_path+'/')
print('Zipping the files...')
os.system('tar czf freya_preprocess.tar.gz freya_preprocess/') # 10 min, approx 400 mb
print('Done')
