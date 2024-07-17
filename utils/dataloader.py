import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset, DataLoader

import torchio as tio #pip install torchio, for 3d data augmentation
import random #for 3d data augmentation

import self_supervised_halos.utils.tng as tng
from self_supervised_halos.utils.tng import subhalos_df



mass_bins = np.linspace(11, 14.7, 11) ## number of classes = len(mass_bins) - 1 = 10
mass_bins_nums = np.histogram(subhalos_df['logSubhaloMass'], bins=mass_bins)[0]
mass_bins_nums = np.log10(mass_bins_nums+1) #logarithm to make the difference between bins less pronounced
mass_bins_weights = np.max(mass_bins_nums)/mass_bins_nums
mass_bins_weights = mass_bins_weights / np.sum(mass_bins_weights)
mass_bins_weights = torch.tensor(mass_bins_weights, dtype=torch.float32)




class HaloDataset(torch.utils.data.Dataset):
    #mass_bins = np.linspace(11, 14.7, 11)  # Define mass_bins globally or pass as argument
    mass_bins = mass_bins
    mass_bins_weights = mass_bins_weights

    def __init__(self, root_dir, subhalos_df,
                 load_2d=True, load_3d=False, load_mass=False,
                 choose_two_2d = False,
                 DEBUG_LIMIT_FILES=None):
        self.root_dir = root_dir
        self.subhalos_df = subhalos_df
        self.files_3d = sorted(glob(root_dir +  '3d/*.npz'))
        self.files_2d = sorted(glob(root_dir + '2d/*.npz'))
        self.files_mass = sorted(glob(root_dir + 'mass/*.npz'))

        if DEBUG_LIMIT_FILES:
            self.files_3d = self.files_3d[:DEBUG_LIMIT_FILES]
            self.files_2d = self.files_2d[:DEBUG_LIMIT_FILES]
            self.files_mass = self.files_mass[:DEBUG_LIMIT_FILES]

        self.load_2d = load_2d
        self.load_3d = load_3d
        self.load_mass = load_mass
        self.choose_two_2d = choose_two_2d


        self.halos_ids = [int(file.split('_')[-2].split('.')[0]) for file in self.files_2d]
        self.loaded_data = self.preload_data()


    def preload_data(self):
        #lesson learned: loading all data at once is faster than loading it on the fly. Before that all files were loaded for each index separately and with the inference time of 0.1 sec the data loading was 30 sec
        load_2d = self.load_2d
        load_3d = self.load_3d
        load_mass = self.load_mass

        data_dict = {}

        data_dict_3d = {}
        data_dict_mass = {}

        if load_2d:
            data_dict_2d = {}
            for file in tqdm(self.files_2d, desc='Preparing 2D data'):
                halo_id = int(file.split('_')[-2].split('.')[0])
                data = np.load(file)
                data_dict_2d[halo_id] = {
                    'map_2d_xy': data['map_2d_xy'],
                    'map_2d_xz': data['map_2d_xz'],
                    'map_2d_yz': data['map_2d_yz'],
                }
            data_dict['2d'] = data_dict_2d

        if load_3d:
            for file in tqdm(self.files_3d, desc='Preparing 3D data'):
                halo_id = int(file.split('_')[-2].split('.')[0])
                data = np.load(file)
                data_dict_3d[halo_id] = {
                    'map_3d': data['map_3d'],
                }
            data_dict['3d'] = data_dict_3d

        if load_mass:
            for file in tqdm(self.files_mass, desc='Preparing mass data'):
                halo_id = int(file.split('_')[-2].split('.')[0])
                data = np.load(file)
                data_dict_mass[halo_id] = {
                    'mass_hist': data['mass_hist'],
                    'snap': data['snap'],
                }
            data_dict['mass'] = data_dict_mass

        return data_dict

    def __len__(self):
        return len(self.halos_ids)

    def select_random_projection(self, choose_two = False):
        if not choose_two:
            return np.random.choice(['xy', 'xz', 'yz'])
        else:
            return np.random.choice(['xy', 'xz', 'yz'], 2, replace=False)


    def __getitem_2d__(self, idx):
        if not self.load_2d:
            return np.zeros(1)

        halo_id = self.halos_ids[idx]
        data_2d = self.loaded_data['2d'][halo_id]

        choose_two = self.choose_two_2d

        # Select a random projection(s)
        if choose_two:
            selected_projections = self.select_random_projection(choose_two = True)
        else:
            selected_projections = [self.select_random_projection(choose_two = False)]

        selected_data = []
        for selected_projection in selected_projections:
            if selected_projection == 'xy':
                proj = np.expand_dims(data_2d['map_2d_xy'], axis=0)
            elif selected_projection == 'xz':
                proj = np.expand_dims(data_2d['map_2d_xz'], axis=0)
            elif selected_projection == 'yz':
                proj = np.expand_dims(data_2d['map_2d_yz'], axis=0)
            selected_data.append(proj)

        if choose_two:
            return (selected_data[0],selected_data[1])
        else:
            return selected_data[0]
        


    def __getitem_3d__(self, idx):
        if not self.load_3d:
            return np.zeros(1)
        halo_id = self.halos_ids[idx]
        data_3d = self.loaded_data['3d'][halo_id]
        selected_data = np.expand_dims(data_3d['map_3d'], axis=0)
        return selected_data

    def __getitem_mass__(self, idx):
        if not self.load_mass:
            return (np.zeros(1), np.zeros(1))

        halo_id = self.halos_ids[idx]
        data_mass = self.loaded_data['mass'][halo_id]
        snap = data_mass['snap']
        mass_hist = data_mass['mass_hist']
        selected_data = (snap, mass_hist)
        selected_data = np.expand_dims(selected_data, axis=0)

        return selected_data

    def __getitem_label__(self, idx):
        halo_id = self.halos_ids[idx]
        label_mass = self.subhalos_df.loc[halo_id]['logSubhaloMass']
        label_class = np.digitize(label_mass, self.mass_bins) - 1
        label = (label_mass, label_class, halo_id)
        return label


    def __getitem__(self, idx):

        data_2d = self.__getitem_2d__(idx)

        data_3d = self.__getitem_3d__(idx)

        data_mass = self.__getitem_mass__(idx)

        label = self.__getitem_label__(idx)

        result_tuple = (data_2d, data_3d, data_mass)

        return result_tuple, label


##this  was an important class then I used no minmax scaling of log(1+counts). Now it is not needed and we can fill rotated images with 0
# class FillInfWithMin:
#     def __init__(self, fill_value=-np.inf):
#         self.fill_value = fill_value

#     def __call__(self, batch):
#         # Create a mask to identify non-inf values
#         mask = batch != self.fill_value

#         # Compute the minimum value per image, ignoring -inf values
#         min_per_image = torch.where(mask, batch, torch.inf).view(batch.size(0), -1).min(dim=1)[0]

#         # Reshape min_per_image to match the dimensions of batch
#         min_per_image = min_per_image.view(batch.size(0), 1, 1, 1)

#         # Replace -inf values with the corresponding minimum values
#         filled_batch = torch.where(batch == self.fill_value, min_per_image, batch)

#         return filled_batch
# img2d_transform= transforms.Compose([
#     transforms.RandomResizedCrop(size=(64, 64), scale=(0.7, 0.99)),
#     transforms.RandomRotation(degrees=180, fill=-np.inf),
#     FillInfWithMin(fill_value=-np.inf)  # Custom transform to fill -inf with min per image
# ])



img2d_transform= transforms.Compose([
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.7, 0.99)),
    transforms.RandomRotation(degrees=180, fill=0.0),
])



# Define the 3D transformations, via chatgpt
def random_resized_crop_3d(image, output_size, scale=(0.7, 0.99)):
    # Get the input size
    input_size = image.shape
    
    # Compute the crop size
    crop_size = [int(s * random.uniform(scale[0], scale[1])) for s in input_size]
    
    # Ensure the crop size is at least the output size
    crop_size = [max(cs, os) for cs, os in zip(crop_size, output_size)]
    
    # Randomly select the crop start point
    crop_start = [random.randint(0, input_size[i] - crop_size[i]) for i in range(len(input_size))]
    
    # Perform the crop
    cropped_image = image[
        crop_start[0]:crop_start[0] + crop_size[0],
        crop_start[1]:crop_start[1] + crop_size[1],
        crop_start[2]:crop_start[2] + crop_size[2]
    ]
    
    # Resize the cropped image to the output size
    resize_transform = tio.Rescale((output_size[0] / crop_size[0], 
                                    output_size[1] / crop_size[1], 
                                    output_size[2] / crop_size[2]))
    
    resized_image = resize_transform(cropped_image)
    return resized_image

def random_rotation_3d(image, degrees=180):
    # Define the rotation angles
    angles = [random.uniform(-degrees, degrees) for _ in range(3)]
    
    # Create the rotation transform
    rotation_transform = tio.RandomAffine(scales=(1, 1), degrees=angles, translation=(0, 0, 0), isotropic=True)
    
    # Apply the rotation
    rotated_image = rotation_transform(image)
    return rotated_image

# Define a composed transform for 3D images
class Composed3DTransform:
    def __init__(self, output_size, scale=(0.7, 0.99), degrees=180):
        self.output_size = output_size
        self.scale = scale
        self.degrees = degrees
    
    def __call__(self, image):
        image = random_resized_crop_3d(image, self.output_size, self.scale)
        image = random_rotation_3d(image, self.degrees)
        return image


img3d_transform = Composed3DTransform( (64, 64, 64))
