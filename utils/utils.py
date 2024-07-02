import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import os 
import sys

import torch
import time



is_freya = True if 'freya' in os.uname().nodename else False

if not is_freya:
    rep_path = '/Users/sdbykov/work/self_supervised_halos/'
    sim_path = None

    print('Running on local machine')
else:
    print('Running on Freya')
    rep_path = '/freya/u/sdbykov/self_supervised_halos/'
    sim_path = '/virgotng/universe/IllustrisTNG/TNG100-1-Dark/'


data_path = f'{rep_path}/data/'
data_postprocess_path = f'{data_path}/freya_postprocess/'

res_path = f'{rep_path}/results/'

#add to path
sys.path.append(rep_path)




def set_mpl(palette = 'pastel', desat = 0.8):

    # matplotlib.use('MacOSX') 
    rc = {
        "figure.figsize": [8, 8],
        "figure.dpi": 100,
        "savefig.dpi": 300,
        # fonts and text sizes
        #'font.family': 'sans-serif',
        #'font.family': 'Calibri',
        #'font.sans-serif': 'Lucida Grande',
        'font.style': 'normal',
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 18,

        # lines
        "axes.linewidth": 1.25,
        "lines.linewidth": 1.75,
        "patch.linewidth": 1,

        # grid
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linestyle": "--",
        "grid.linewidth": 0.75,
        "grid.alpha": 0.75,

        # ticks
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,

        'lines.markeredgewidth': 0.5, #1.5,
        "lines.markersize": 5,
        "lines.markeredgecolor": "k",
        'axes.titlelocation': 'left',
        "axes.formatter.limits": [-3, 3],
        "axes.formatter.use_mathtext": True,
        "axes.formatter.min_exponent": 2,
        'axes.formatter.useoffset': False,
        "figure.autolayout": False,
        "hist.bins": "auto",
        "scatter.edgecolors": "k",
    }




    sns.set_context('notebook', font_scale=1.25)
    matplotlib.rcParams.update(rc)
    if palette == 'shap':
        #colors from shap package: https://github.com/slundberg/shap
        cp = sns.color_palette( ["#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"])
        sns.set_palette(cp, color_codes = True, desat = desat)
    elif palette == 'shap_paired':
        #colors from shap package: https://github.com/slundberg/shap, + my own pairing of colors
        cp = sns.color_palette( ["#1E88E5", "#1e25e5", "#ff0d57", "#ff5a8c",  "#13B755", "#2de979","#7C52FF", "#b69fff", "#FFC000", "#ffd34d","#00AEEF", '#3dcaff'])
        sns.set_palette(cp, color_codes = True, desat = desat)
    else:
        sns.set_palette(palette, color_codes = True, desat = desat)
    print('matplotlib settings set')
set_mpl()




def check_cuda():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_freya:
        print('working of Freya:', os.uname().nodename)

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"CUDA is available. Number of GPUs: {num_gpus}")
        for idx, name in enumerate(gpu_names):
            print(f"GPU {idx}: {name}")
    else:
        print("CUDA is not available.")

    print(f"Device: {device}")

    return device
