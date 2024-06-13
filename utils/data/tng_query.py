from ..utils import rep_path, data_path
import requests
import numpy as np
import h5py
import os
import time
from pylab import figure, cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

import MAS_library as pylians_MASL
import smoothing_library as pylians_SL

data_path = data_path + '/tng'
baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"9af02c30810f12be44f17ad2bd4b6510"}

sim_name = 'TNG100-1-Dark'
redshift = 0.0
base_query = f'/{sim_name}/snapshots/z={redshift}/'

h = 0.6774 #TODO get from API??
omega_0 = 0.3089
omega_b = 0.0486
mass_dm = 0.000599968882709879

softening_dm_comoving = 1.0 #ckpc/h, https://www.tng-project.org/data/forum/topic/408/softening-lengths/

#to get physical units from comoving:
#physical = comoving * a / h, see task 6  rr *= scale_factor/little_h # ckpc/h -> physical kpc from https://www.tng-project.org/data/docs/api/


#TODO make get for the case when I am on the server and use simulation files directly

is_freya = True if 'freya' in os.uname().nodename else False

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



class HaloInfo:
    def __init__(self, haloid):
        self.haloid = haloid

        self.halo_url = f'{baseUrl}{base_query}subhalos/{haloid}/'

        self.cutout_url = f'{self.halo_url}cutout.hdf5'
        self.sublink_url = f'{self.halo_url}sublink/full.hdf5'

        self.cutout_file = data_path + f'/halo_{haloid}_cutout.hdf5'
        self.sublink_file = data_path + f'/halo_{haloid}_sublink.hdf5'

        #get attrubutes via url like this:
        #meta = get(halo.halo_url)
        #meta['mass] etc... but get takes time


    def __repr__(self):
        return f'HaloInfo {self.haloid}; urls: {self.halo_url}'


    #make download_halo_snapshot but as a method
    def download_halo_snapshot(self, retry = True, verbose = True):
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


    def download_sublink(self, retry = True):
        subhalo_url = self.halo_url

        try:
            halo_id = subhalo_url.split('/')[-2]
            mass_hist_name = f'halo_{halo_id}_sublink.hdf5'
            filepath = f'{data_path}/{mass_hist_name}'

            if os.path.exists(filepath):
                filesize = os.path.getsize(filepath)
                print(f'{mass_hist_name} already exists, {filesize/1e6:.2f} MB')
                return filepath
            
            t0 = time.time()

            download_url = subhalo_url + 'sublink/full.hdf5'
            print(f'Downloading cutout from {download_url}')
            sublink = get(download_url)
            t1 = time.time()
            filesize = os.path.getsize(sublink)
            print(f'Downloaded {mass_hist_name} in {t1-t0:.2f} s, {filesize/1e6:.2f} MB')
            os.rename(sublink, filepath)
            return filepath

        except Exception as e:
            print(f'Error downloading sublink {subhalo_url}, {e}')
            if retry:
                print(f'retrying...')
                return self.download_sublink(retry = False)
            else:
                print(f'stopping')
                return None


    def make_3d_density(self,
                        box_size = 500,
                        grid_bins = 128,
                        smooth_R = 5,
                        smooth_type = 'Gaussian',
                        clip_smoothed = 1e-5
                        ):
        
        haloid = self.haloid

        cutout_file = self.cutout_file

        with h5py.File(cutout_file, 'r') as f:
            x = f['PartType1']['Coordinates'][:,0] #units of [ckpc/h]
            y = f['PartType1']['Coordinates'][:,1]
            z = f['PartType1']['Coordinates'][:,2]
            pot = f['PartType1']['Potential'][:]

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

            hist, edges = np.histogramdd(pos, bins = grid_bins, range = [[-box_size/2, box_size/2]]*3)

            hist_32 = hist.astype(np.float32)

            # #pylians3 https://pylians3.readthedocs.io/en/master/construction.html

            W_k = pylians_SL.FT_filter(box_size, smooth_R, grid_bins, smooth_type, 28)

            hist_smoothed_32 = pylians_SL.field_smoothing(hist_32, W_k, 28)

            #hist_smoothed_32[hist_smoothed_32 < 0] = 0
            if clip_smoothed is not None:
                hist_smoothed_32[hist_smoothed_32 < clip_smoothed] = 0

        mass_log_msun = np.log10(pos.shape[0] * mass_dm * 1e10 / h)

        result = {'hist':hist, 'hist_smoothed':hist_smoothed_32, 'edges':edges, 'pos':pos, 'pot':pot,  'haloid':haloid, 'mass_log_msun':mass_log_msun}

        return result



    def make_2d_density(self,
                        plane= 'xy',
                        box_size = 500,
                        grid_bins = 128,
                        smooth_R = 5,
                        smooth_type = 'Gaussian',
                        clip_smoothed = 1e-5
                        ):
        
        axis_to_proj = {'xy':2, 'xz':1, 'yz':0}
        axis_to_proj = axis_to_proj[plane]
        
        haloid = self.haloid

        cutout_file = self.cutout_file

        with h5py.File(cutout_file, 'r') as f:
            x = f['PartType1']['Coordinates'][:,0] #units of [ckpc/h]
            y = f['PartType1']['Coordinates'][:,1]
            z = f['PartType1']['Coordinates'][:,2]
            pot = f['PartType1']['Potential'][:]

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

            hist, edges = np.histogramdd(pos, bins = grid_bins, range = [[-box_size/2, box_size/2]]*3)

            hist = np.sum(hist, axis = axis_to_proj)

            hist_32 = hist.astype(np.float32)

            # #pylians3 https://pylians3.readthedocs.io/en/master/construction.html

            W_k = pylians_SL.FT_filter_2D(box_size, smooth_R, grid_bins, smooth_type, 28)

            hist_smoothed_32 = pylians_SL.field_smoothing_2D(hist_32, W_k, 28)

            if clip_smoothed is not None:
                hist_smoothed_32[hist_smoothed_32 < clip_smoothed] = 0


        mass_log_msun = np.log10(pos.shape[0] * mass_dm * 1e10 / h)

        result = {'hist':hist, 'hist_smoothed':hist_smoothed_32, 'edges':edges, 'pos':pos, 'pot':pot,  'haloid':haloid, 'mass_log_msun':mass_log_msun}

        return result


    @staticmethod
    def _plot_3d_density(dens_res):
        hist = dens_res['hist']
        hist_smoothed = dens_res['hist_smoothed']
        edges = dens_res['edges']


        #make 3x2 plots which are sum of 3d density along each axis, and unsmoothed/smoothed

        fig, axs = plt.subplots(3,2, figsize = (10,15))

        for axis_i in range(3):
            for smooth_i in range(2):
                if smooth_i == 0:
                    map_2d = hist.sum(axis = axis_i).T
                else:
                    map_2d = hist_smoothed.sum(axis = axis_i).T
                
                map_2d = np.log10(map_2d/np.nanmax(map_2d))

                ax = axs[axis_i, smooth_i]

                #im = ax.imshow(map_2d, cmap='bone', norm=LogNorm(vmin=0.01, vmax=5))
                im = ax.imshow(map_2d, cmap='bone')
                ax.contour(map_2d, levels=[-5,-4,-3,-2,-1], cmap='Reds', linewidths=1.5)

                title_x = ['yz', 'xz', 'xy'][axis_i]
                title_smooth = ['unsmoothed', 'smoothed'][smooth_i]
                ax.set_title(f'{title_x} {title_smooth}')

        #set title
        fig.suptitle(f'3d density, halo {dens_res["haloid"]}')
        plt.show()


        #make a plot with colorbar and levels of xy plane

        f = figure(figsize=(12, 12))
        ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
        axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])

        map_2d = hist_smoothed.sum(axis = 2).T
        map_2d = np.log10(map_2d/np.nanmax(map_2d))

        #im = ax.imshow(map_2d, cmap='bone', norm=LogNorm(vmin=0.01, vmax=5))
        im = ax.imshow(map_2d, cmap='bone')
        levels = [-5,-4,-3,-2,-1]
        CS = ax.contour(map_2d, levels=levels, cmap='Reds', linewidths=1.5)

        ax.clabel(CS, CS.levels, inline=True, fmt='%1.0f', fontsize=10)

        #f.colorbar(im, cax=axcolor, orientation='vertical')
        f.colorbar(im, cax=axcolor, orientation='vertical', ticks = levels, label = 'log10 [density/max density]')
        ax.set_title(f'xy smoothed density, halo {dens_res["haloid"]}; logM {dens_res["mass_log_msun"]:.2f} Msun/h')
        plt.show()


    @staticmethod
    def _plot_2d_density(dens_res):
        hist_smoothed = dens_res['hist_smoothed']


        f = figure(figsize=(12, 12))
        ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
        axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])

        map_2d = hist_smoothed.T
        map_2d = np.log10(map_2d/np.nanmax(map_2d))

        #im = ax.imshow(map_2d, cmap='bone', norm=LogNorm(vmin=0.01, vmax=5))
        im = ax.imshow(map_2d, cmap='bone')
        levels = [-5,-4,-3,-2,-1]
        CS = ax.contour(map_2d, levels=levels, cmap='Reds', linewidths=1.5)

        ax.clabel(CS, CS.levels, inline=True, fmt='%1.0f', fontsize=10)

        #f.colorbar(im, cax=axcolor, orientation='vertical')
        f.colorbar(im, cax=axcolor, orientation='vertical', ticks = levels, label = 'log10 [density/max density]')
        ax.set_title(f'projected smoothed density, halo {dens_res["haloid"]}; logM {dens_res["mass_log_msun"]:.2f} Msun/h')
        plt.show()


    @staticmethod
    def _plot_2d_density_surface(dens_res, clip = -3):
        hist_smoothed = dens_res['hist_smoothed']


        map_2d = hist_smoothed.T
        map_2d = np.log10(map_2d/np.nanmax(map_2d))

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(0, map_2d.shape[0])
        y = np.arange(0, map_2d.shape[1])
        X, Y = np.meshgrid(x, y)
        Z = map_2d
        #make -inf to 0
        Z[Z==-np.inf] = np.nanmin(Z[Z!=-np.inf])

        if clip is not None:
            Z[Z < clip] = clip

        ax.plot_surface(X, Y, Z, cmap='bone')


        ax.set_title(f'projected smoothed density, halo {dens_res["haloid"]}; logM {dens_res["mass_log_msun"]:.2f} Msun/h')
        plt.show()



    def make_mass_history(self):
        sublink_file = self.sublink_file
        with h5py.File(sublink_file, 'r') as f:
            snap = f['SnapNum'][:]
            mass = f['SubhaloMass'][:]
            id = f['SubhaloID'][:]

            id_0 = id[0]

            #only id==id_0
            mask = id == id_0
            snap = snap[mask]
            mass = mass[mask]

            mass_dict = dict(zip(snap, mass))

        return mass_dict

