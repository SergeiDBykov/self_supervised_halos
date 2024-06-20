import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


#mass bins for classification
bins = np.linspace(11, 14.7, 11)

class HaloDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, load_3d = False):
        self.root_dir = root_dir
        
        self.files = glob(root_dir+'*.npz')
        self.halos_ids = [int(x.split('_')[-1].split('.')[0]) for x in self.files]

        self.load_3d = load_3d

    def __len__(self):
        return len(self.halos_ids)
    
    def __getitem__(self, idx):
        halo_id = self.halos_ids[idx]
        fname_root = f'{self.root_dir}halo_{halo_id}'

        data = np.load(fname_root+'.npz')
        data_xy = data['map_2d_xy']
        data_xz = data['map_2d_xz']
        data_yz = data['map_2d_yz']
        snap = data['snap']
        snap_mass = data['mass']
        data_3d = data['map_3d'] 

        label_mass = subhalos_df.loc[halo_id]['logSubhaloMass']
        label_class = np.digitize(label_mass, bins)
        label = (label_mass, label_class)

        del data

        
        tuple_to_resuts = data_3d,(data_xy, data_xz, data_yz), (snap, snap_mass)

        return tuple_to_resuts, label
    
