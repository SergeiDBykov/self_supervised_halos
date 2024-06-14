import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sys


is_freya = True if 'freya' in os.uname().nodename else False

if not is_freya:
    rep_path = '/Users/sdbykov/work/self-supervised-halos/'
    sim_path = None

    print('Running on local machine')
else:
    print('Running on Freya')
    rep_path = '/freya/u/sdbykov/self-supervised-halos/'
    sim_path = '/virgotng/universe/IllustrisTNG/TNG100-1-Dark/'


data_path = f'{rep_path}/data/'

#add to path
sys.path.append(rep_path)



