from self_supervised_halos.utils.utils import data_preprocess_path
from self_supervised_halos.utils.dataloader import HaloDataset, subhalos_df, DataLoader, img2d_transform

root_dir = data_preprocess_path
dataset = HaloDataset(root_dir,subhalos_df, 
                      load_2d=True, load_3d=False, load_mass=False,
                        choose_two_2d = True,
                      DEBUG_LIMIT_FILES = 10)

dataloader = DataLoader(dataset, batch_size=3, shuffle=True,
                        )

batch = next(iter(dataloader))

import code; code.interact(local=locals())
