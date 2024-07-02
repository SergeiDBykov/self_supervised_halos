import importlib
from self_supervised_halos import utils
rep_path, data_path, is_freya, sim_path = utils.rep_path, utils.data_path, utils.is_freya, utils.sim_path


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

subhalos_df['logSubhaloMass'] = np.log10(subhalos_df['SubhaloMass']*1e10/h)


def smooth_hist(hist, filter_size_pix = 2):
    hist_filtered = scipy.ndimage.gaussian_filter(hist, filter_size_pix)
    return hist_filtered
    

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


        self.hist_filepath = data_path + f'freya/halo_{haloid}_hist.npz'
        try:
            hist_file = np.load(self.hist_filepath)
            self.hist_file = hist_file
        except:
            self.hist_file = None


        series = subhalos_df.loc[haloid]
        self.meta = series.to_dict()
        self.mass_log_msun = np.log10(self.meta['SubhaloMass']*1e10/h)

        self.mass_history = subhalos_mass_history[haloid]


    def __repr__(self):
        return f'HaloInfo {self.haloid}; urls: {self.halo_url}'


    def get_snapshot(self):
        assert is_freya, 'This function is only for Freya'
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
        
        if not self.hist_file:
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

            #TODO note that float32 is used, so the values are not very precise
            hist = hist.astype(np.float32)

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
            
            self.dens = result

        else:
            #print(f'Using precomputed histogram: {self.haloid}')
            hist_file = self.hist_file

            hist = hist_file['hist']
            proj_yz = hist_file['proj_yz']
            proj_xz = hist_file['proj_xz']
            proj_xy = hist_file['proj_xy']

            projections_2d = {
                'yz':proj_yz,
                'xz':proj_xz,
                'xy':proj_xy,
            }

            edges = hist_file['edges']
            edge_binsize = hist_file['edge_binsize']
            box_half_size = hist_file['box_half_size']
            half_mass_rad = hist_file['half_mass_rad']
            is_in_units_of_halfmassrad = hist_file['is_in_units_of_halfmassrad']

            result = {
                    'hist':hist, 
                    'edges':edges, 
                    'edge_binsize':edge_binsize,
                    'box_half_size': box_half_size,
                    'half_mass_rad':half_mass_rad,
                    'is_in_units_of_halfmassrad':is_in_units_of_halfmassrad, 
                    'projections':projections_2d,
            }
        
        self.dens = result


        return result



    def data_transform(self, dens = None, smooth = None):
        def count_normalization(array):
            array = array + 1 
            array = array / np.max(array)
            array = np.log10(array)
            return array
        

        result = {}

        if dens is None:
            dens = self.make_3d_density()
        
        map_3d = dens['hist']
        map_2d_xy = dens['projections']['xy']
        map_2d_xz = dens['projections']['xz']
        map_2d_yz = dens['projections']['yz']

        if smooth:
            map_2d_xy = smooth_hist(map_2d_xy, filter_size_pix=smooth)
            map_2d_xz = smooth_hist(map_2d_xz, filter_size_pix=smooth)
            map_2d_yz = smooth_hist(map_2d_yz, filter_size_pix=smooth)
            map_3d = smooth_hist(map_3d, filter_size_pix=smooth)
        
        map_3d = count_normalization(map_3d)
        map_2d_xy = count_normalization(map_2d_xy)
        map_2d_xz = count_normalization(map_2d_xz)
        map_2d_yz = count_normalization(map_2d_yz)


        #mass history transform:
        #need to always be of the same shape, starting with snapshot 0 and ending with 99

        mass_history = self.mass_history
        snapshot = np.array(mass_history['snap'])
        mass = np.array(mass_history['mass'])

        #snapshot = np.array(snapshot)/99
        mass = mass / np.max(mass)
        mass = np.log10(mass)

        snapshot = snapshot[::-1]
        mass = mass[::-1]


        #the following is to make the mass history always have the same shape as needed for pytorch

        new_snapshot = np.arange(100)
        new_mass = np.full(100, np.nan)

        mask = np.in1d(new_snapshot, snapshot)
        new_mass[mask] = mass

        new_snapshot = new_snapshot/99


        result['map_3d'] = map_3d
        result['map_2d_xy'] = map_2d_xy
        result['map_2d_xz'] = map_2d_xz
        result['map_2d_yz'] = map_2d_yz
        result['snapshot'] = new_snapshot
        result['mass_hist'] = new_mass

        return result


    def plot_2d_density(self, dens_res,
                        proj = 'xy',
                        ax = None, 
                        levels = [-5,-4,-3,-2,-1],
                        smooth_size = 1,
                        cmap = 'afmhot'):
        if dens_res is None:
            dens_res = self.make_3d_density()


        map_2d = dens_res['projections'][proj]

        if smooth_size:
            map_2d = smooth_hist(map_2d, filter_size_pix=smooth_size)

        map_2d = map_2d + 1.0
        map_2d = map_2d / np.max(map_2d)

        map_2d = np.log10(map_2d)

        if ax is None:
            f = plt.figure(figsize=(8, 8))
            ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
            axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])

        else:
            #f = ax.figure
            #ax = ax
            axcolor = None


        im = ax.imshow(map_2d, cmap=cmap)
        CS = ax.contour(map_2d, levels=levels, cmap='Reds', linewidths=1.5)
        ax.clabel(CS, CS.levels, inline=True, fmt='%1.0f', fontsize=10)

        if axcolor:
            #add colorbar similar to 
            f.colorbar(im, cax=axcolor, orientation='vertical', ticks = levels, label = 'log10 [(1 + counts) / max]')
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
        
    
        return ax


    def plot_mass_history(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize = (8,4))
        
        mass_history = self.mass_history
        snap = mass_history['snap']
        mass = mass_history['mass']
        ax.plot(snap, mass, '.-')
        ax.set_title(f'Mass history of halo {self.haloid}; logM {self.mass_log_msun:.2f} Msun/h')
        ax.set_xlabel('snapshot')
        ax.set_ylabel('mass [Msun/h]')
        ax.set_yscale('log')


    def plot_all(self, dens, compact = False, smooth_size = 1):
        if dens is None:
            dens = self.make_3d_density()

        if not compact:
            fig,  axs =  plt.subplots(2,2, figsize = (12,12))
            ax1, ax2, ax3, ax4 = axs.flatten()

            self.plot_2d_density(dens, ax = ax1, smooth_size = smooth_size)
            self.plot_2d_density(dens, ax = ax2, proj='xz', smooth_size = smooth_size)
            self.plot_2d_density(dens, ax = ax3, proj='yz', smooth_size = smooth_size)
            self.plot_mass_history(ax = ax4)

            self.plot_2d_density(dens)

        else:
            fig,  axs =  plt.subplots(1,2, figsize = (12,6))
            ax1, ax2 = axs.flatten()
            self.plot_2d_density(dens, ax = ax1, proj = 'xy' ,smooth_size = smooth_size)
            self.plot_mass_history(ax = ax2)
        
        return fig, axs



