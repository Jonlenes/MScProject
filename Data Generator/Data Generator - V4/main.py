import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import random
import cv2

from img_util import MatplotlibUtil, set_img_label
from functions import Functions
from io_util import show, save
from config import panel_side_base
from timer import Timer
from tqdm import tqdm
from utils import *


def init():    
    # Object of functions
    return Functions()


def normalize(x, lower=-1, upper=1):
    return ((x - x.min()) / (x.max() - x.min()) * (upper - lower) + lower)


def plot_on_nparray(array, x, y):
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)	
    valid_idxs = np.where(np.logical_and((y >= 0), (y < panel_side_base)))[0]
    reflexivity = random.uniform(-1,1)
    array[ y[valid_idxs], x[valid_idxs] ] = reflexivity
        

def generate_curves(funcs, func_name, verbose=True):
    # Config matplotlib to plot
    MatplotlibUtil.full_frame(panel_side_base, panel_side_base)
    plt.ylim(0, panel_side_base)
    plt.xlim(0, panel_side_base)

    # Incluination of actual data
    funcs.random_params()
    panel = np.zeros((panel_side_base, panel_side_base))

    y_pos = -int(0.3*panel_side_base)
    while y_pos < int(1.3 * panel_side_base):
        x, y = funcs( func_name )
        y = y_pos + y
        plot_on_nparray(panel, x, y)
        y_pos += random.randint(10, 20)

    return panel


if __name__ == '__main__':  

    timer = Timer()
    timer.start()

    # Needed configs before data generation 
    funcs = init()
    
    debug = True
    print_f_names = True
    posfix_name = '_data.png'
    number_of_funs = len(funcs.f_names) 

    for i in tqdm(range(1)):
        func_name = 'f1' # funcs.f_names[ random.randint(0, number_of_funs - 1 ) ]
        panel = generate_curves( funcs, func_name, False )
    
        if debug: 
            show(panel, vmin=-1, vmax=1, color='seismic')
        else:
            save(panel, str(i) + '_structure.png', 'seismic', vmin=-1, vmax=1)

        # Params to ricker wavelet
        taper_size = 4
        wavelet_window_size = 50
        dt = 0.004
        frequency = 20 #random.randint(5, 25)

        # Convolve the panel with ricker wavelet
        panel = convolve_wavelet(panel=panel, frequency=frequency, dt=dt, taper_size=taper_size, window_size=wavelet_window_size)

        #Normaliza o dado de [-1,1]
        panel = range_normalize_std( panel )
        # panel = normalize( panel )
        
        if debug: 
            show(panel, vmin=-1, vmax=1, color='seismic')
        else:
            save(panel, str(i) + posfix_name, 'seismic', vmin=-1, vmax=1)

        if not debug and print_f_names:
            im = cv2.imread(str(i) + posfix_name)
            set_img_label(im, funcs.get_str_eq( func_name ), (25, 50))
            cv2.imwrite(str(i) + posfix_name, im)
            
    print( timer.diff() )
