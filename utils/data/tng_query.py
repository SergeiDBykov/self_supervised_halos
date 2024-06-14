from ..utils import rep_path, data_path, is_freya, sim_path

import requests
import numpy as np
import h5py
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import scipy.ndimage


import illustris_python as il

# if not is_freya:
#     import MAS_library as pylians_MASL
#     import smoothing_library as pylians_SL



with open(rep_path+'/tng_api_key.txt', 'r') as f:
    api_key = f.read().strip()

headers = {"api-key":api_key}

redshift = 0.0
snapshot = 99

baseUrl = 'http://www.tng-project.org/api/'
base_query = f'/TNG100-1-Dark/snapshots/{snapshot}/'

h = 0.6774 #TODO get from API??
omega_0 = 0.3089
omega_b = 0.0486
mass_dm = 0.000599968882709879

softening_dm_comoving = 1.0 #ckpc/h, https://www.tng-project.org/data/forum/topic/408/softening-lengths/

#to get physical units from comoving:
#physical = comoving * a / h, see task 6  rr *= scale_factor/little_h # ckpc/h -> physical kpc from https://www.tng-project.org/data/docs/api/

subhalos_mass_history = pickle.load(open(data_path+'subhalos_history.pkl', 'rb'))
try:
    subhalos_df = pd.read_pickle(data_path+'subhalos_df.pkl')
except:
    #no idea why pickle is not working on Freya
    subhalos_df = pd.read_csv(data_path+'subhalos_df.csv', index_col=0)



def smooth_hist(hist, filter_size_pix = 2):
    hist_filtered = scipy.ndimage.gaussian_filter(hist, filter_size_pix)
    return hist_filtered
    

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r


def get_subhalo_from_sim(subhaloID):
    assert is_freya, 'This function is only for Freya'
    base_path = sim_path+'output/'

    snap = il.snapshot.loadSubhalo(base_path, snapNum=snapshot, id = subhaloID, partType = 1)
    return snap




class HaloInfo:
    def __init__(self, haloid):
        self.haloid = haloid

        self.halo_url = f'{baseUrl}{base_query}subhalos/{haloid}/'

        self.cutout_url = f'{self.halo_url}cutout.hdf5'
        self.sublink_url = f'{self.halo_url}sublink/full.hdf5'

        self.cutout_file = data_path + f'tng/halo_{haloid}_cutout.hdf5'
        self.sublink_file = data_path + f'tng/halo_{haloid}_sublink.hdf5'

        series = subhalos_df.loc[haloid]
        self.meta = series.to_dict()
        self.mass_log_msun = np.log10(self.meta['SubhaloMass']*1e10/h)

        self.mass_history = subhalos_mass_history[haloid]


    def __repr__(self):
        return f'HaloInfo {self.haloid}; urls: {self.halo_url}'


    #make download_halo_snapshot but as a method
    def download_halo_snapshot(self, retry = True, verbose = True):
        if is_freya:
            raise Exception('Cannot download on Freya')

        subhalo_url = self.halo_url
        try:
            halo_id = subhalo_url.split('/')[-2]
            cutout_request = {'dm':'Coordinates,Potential'}
            cutout_name = f'halo_{halo_id}_cutout.hdf5'
            filepath = f'{data_path}/{cutout_name}'

            if os.path.exists(filepath):
                filesize = os.path.getsize(filepath)
                if verbose:
                    print(f'{cutout_name} already exists, {filesize/1e6:.2f} MB')
                return filepath
            
            t0 = time.time()

            download_url = subhalo_url + 'cutout.hdf5'
            if verbose:
                print(f'Downloading cutout from {download_url}')
            cutout = get(download_url, cutout_request)
            t1 = time.time()
            filesize = os.path.getsize(cutout)
            if verbose:
                print(f'\t Downloaded for {self.haloid} finished in {t1-t0:.2f} s, {filesize/1e6:.2f} MB')
            os.rename(cutout, filepath)
            return filepath

        except Exception as e:
            if verbose: print(f'Error downloading {subhalo_url}, {e}')
            if retry:
                if verbose: print(f'retrying...')
                return self.download_halo_snapshot(retry = False)
            else:
                if verbose: print(f'stopping')
                return None


    def get_snapshot(self):
        if not is_freya:
            cutout_file = self.cutout_file
            snap = {}
            with h5py.File(cutout_file, 'r') as f:
                x = f['PartType1']['Coordinates'][:,0] #units of [ckpc/h]
                y = f['PartType1']['Coordinates'][:,1]
                z = f['PartType1']['Coordinates'][:,2]
                pot = f['PartType1']['Potential'][:]

                snap['x'] = x
                snap['y'] = y
                snap['z'] = z
                snap['pot'] = pot
        
        if is_freya:
            snap = get_subhalo_from_sim(self.haloid)
            snap['x'] = snap['Coordinates'][:,0]
            snap['y'] = snap['Coordinates'][:,1]
            snap['z'] = snap['Coordinates'][:,2]
            snap['pot'] = snap['Potential']


        return snap



    def make_3d_density(self,
                        box_half_size = -5,
                        grid_bins = 64,
                        ):
        
        haloid = self.haloid

        snap = self.get_snapshot()
        x = snap['x']
        y = snap['y']
        z = snap['z']
        pot = snap['pot']

        del snap


        #coords of max potential
        max_pot = np.argmin(pot)
        x_max = x[max_pot]
        y_max = y[max_pot]
        z_max = z[max_pot]

        #centering
        x -= x_max
        y -= y_max
        z -= z_max

        pos = np.array([x,y,z]).T

        half_mass_rad = self.meta['SubhaloHalfmassRad']


        is_in_units_of_halfmassrad = False
        if box_half_size<0:
            #if box__half_size<0, then the box is is in units of halfmassrad
            assert -box_half_size < 10, 'box_half_size in units of half_mass_radius should be less than 10, usually 3-4'
            box_half_size = -box_half_size*half_mass_rad
            is_in_units_of_halfmassrad = True


        edge_binsize = 2*box_half_size/grid_bins
        box_range = [[-box_half_size, box_half_size]]*3

        hist, edges = np.histogramdd(pos, bins = grid_bins, range = box_range)

        #TODO NOTE THAT float32 is used, so the values are not very precise
        hist = hist.astype(np.float32)
        #hist = hist.astype(np.int32)



        projections_2d = {}
        proj_name = ['yz', 'xz', 'xy']

        for axis_i in range(3):
            proj = proj_name[axis_i]
            map_2d = hist.sum(axis = axis_i).T

            projections_2d[proj] = map_2d

        result = {
                'hist':hist, 
                'edges':edges, 
                'edge_binsize':edge_binsize,
                'box_half_size': box_half_size,
                'half_mass_rad':half_mass_rad,
                'is_in_units_of_halfmassrad':is_in_units_of_halfmassrad, 
                'projections':projections_2d,
                }

        return result





    def plot_2d_density(self, dens_res,
                        proj = 'xy',
                        ax = None, 
                        levels = [-5,-4,-3,-2,-1],
                        smooth_size = 1):

        map_2d = dens_res['projections'][proj]

        if smooth_size:
            map_2d = smooth_hist(map_2d, filter_size_pix=smooth_size)


        map_2d = np.log10(map_2d/np.nanmax(map_2d))

        if ax is None:
            f = plt.figure(figsize=(8, 8))
            ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
            axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])

        else:
            #f = ax.figure
            #ax = ax
            axcolor = None


        im = ax.imshow(map_2d, cmap='bone')
        CS = ax.contour(map_2d, levels=[-5,-4,-3,-2,-1], cmap='Reds', linewidths=1.5)
        ax.clabel(CS, CS.levels, inline=True, fmt='%1.0f', fontsize=10)

        if axcolor:
            #add colorbar similar to 
            f.colorbar(im, cax=axcolor, orientation='vertical', ticks = levels, label = 'log10 [density/max density]')
        else:
            None
                
        
        edge_binsize = dens_res['edge_binsize']

        #make ticks in physical units instead of bins number, zero is in the middle
        ticks = np.linspace(-dens_res['box_half_size'], dens_res['box_half_size'], 5)
        #ticks = np.round(ticks/edge_binsize, 1)
        ticks = np.round(ticks, 1)

        ax.set_xticks(np.linspace(0, map_2d.shape[0], 5))
        ax.set_xticklabels(ticks)
        ax.set_yticks(np.linspace(0, map_2d.shape[1], 5))
        ax.set_yticklabels(ticks)

        ax.set_xlabel(f'{proj[0]} [ckpc/h]')
        ax.set_ylabel(f'{proj[1]} [ckpc/h]')
    
    

        label = f"ID {self.haloid}; logM {self.mass_log_msun:.2f} Msun/h; \n {proj} plane"

        #set title inside, top left
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


    def plot_mass_history(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        mass_history = self.mass_history
        snap = mass_history['snap']
        mass = mass_history['mass']
        ax.plot(snap, mass)
        ax.set_title(f'Mass history of halo {self.haloid}; logM {self.mass_log_msun:.2f} Msun/h')
        ax.set_xlabel('snapshot')
        ax.set_ylabel('mass [Msun/h]')
        ax.set_yscale('log')


    def plot_all(self, dens):
        fig,  axs =  plt.subplots(2,2, figsize = (12,12))
        ax1, ax2, ax3, ax4 = axs.flatten()

        self.plot_2d_density(dens, ax = ax1)
        self.plot_2d_density(dens, ax = ax2, proj='xz')
        self.plot_2d_density(dens, ax = ax3, proj='yz')
        self.plot_mass_history(ax = ax4)

        self.plot_2d_density(dens)



