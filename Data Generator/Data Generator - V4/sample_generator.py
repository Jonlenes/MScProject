import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import random
import cv2
import math
import numpy as np

from glob import glob

from img_util import MatplotlibUtil, set_img_label
from functions import Functions, _bring_to_zero, scale
from io_util import show, save
from config import panel_side_base
from timer import Timer
from tqdm import tqdm
from scipy.signal import convolve2d

from cepetro_util import add_faulting, calculate_min_offset, range_normalize_std, convolve_wavelet

from config import output_folder

from util import get_valid_indexs


def init():    
    # Object of functions
    return Functions()

def load_angles():
    #Importa lista com todos os angulos possiveis 
    radianos_file = open('radianos.txt','r')
    radianos = []

    for line in radianos_file:
        radianos.extend([float(i) for i in line.split()])
    
    return radianos


dt = 0.004
frequency = 20 #random.randint(5, 25)
radians = load_angles()

# All configs need before data generation 
funcs = init()
        

def normalize(x, lower=-1, upper=1):
    return ((x - x.min()) / (x.max() - x.min()) * (upper - lower) + lower)


def plot_on_nparray(array, x, y, color=None, panel_size=None):
    # Check sizes
    if len(x) == 0: return
    if panel_size is None: panel_size = panel_side_base
    
    # Convert to int
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    
    # Valid bounds
    valid_idxs = get_valid_indexs(x, y, panel_size)
    if len(x[valid_idxs]) == 0: return
    
    # Make a plot
    if color is None:
        color = random.uniform(-1,1)
    array[ y[valid_idxs], x[valid_idxs] ] = color
        

def generate_curves(funcs, func_name, verbose=True):
    # Incluination of actual data
    funcs.random_params()
    panel = np.zeros((panel_side_base + 10, panel_side_base + 10))

    if '_e_' in func_name:
        list_x, list_y = funcs( func_name )
        # print(len(list_x), len(list_y))
        for x, y in zip(list_x, list_y):
            plot_on_nparray(panel, x, y)
    else:    
        y_pos = -int(0.3*panel_side_base)
        while y_pos < int(1.3 * panel_side_base):
            x, y = funcs( func_name )
            y = y_pos + y
            plot_on_nparray(panel, x, y)
            y_pos += random.randint(10, 20)

    # Add faulting ***********************************
    # Calcula qual o afastamento minimo na quebra
    min_offset = calculate_min_offset(frequency, dt=dt)
    max_offset = 20
    
    # Sorteia um o afastamento e angulo da quebra
    offset = random.randint(min_offset, max_offset)
    angle_rad = random.choice(radians)
    
    # Gera o painel com a quebra escolhida
    panel = add_faulting(angle_rad=angle_rad, panel_a=panel, offset=offset, output_shape=panel.shape)
    # Add faulting ***********************************

    # Two panels
    panel_ricker = panel
    panel_psf = panel.copy()

    # Add noise, artifact and faul
    count = np.random.randint(1000, 2000)
    
    for _ in range( count ):
        size = np.random.randint(3, 7)
        
        x = np.random.randint(size, panel_side_base-size)
        y = np.random.randint(size, panel_side_base-size)

        v_straight = np.arange(0, size)
        sign = random.choice([True, False])
        angle = (np.random.randint(0, 360) * np.pi) / 180.0
        
        if sign: 
            o_straight = np.tan(angle) * v_straight 
            o_straight = _bring_to_zero( o_straight )
            plot_on_nparray(panel_psf, v_straight + x, o_straight + y)
        else:
            if not math.isclose(np.tan(angle), 0):
                o_straight = v_straight / np.tan(angle) 
                o_straight = _bring_to_zero( o_straight )
                plot_on_nparray(panel_psf, o_straight + x, v_straight + y)

    # Gausian noise
    count = np.random.randint(2000, 4000)
    data = np.random.randint(0, panel_side_base, (count, 2))
    plot_on_nparray(panel_psf, data[:, 0], data[:, 1])

    return panel_ricker, panel_psf


def get_psf():
    # This rang is fixed for now
    index = random.randint(0, 4)
    # A ultima PSF não pode ser inserida pois está com a polaridade invertida e a Ricker não.
    return np.load('PSFs/psf_' + str(index) + '.npy')
    

def get_sample(debug=False, save_data=False, posfix_name=None, print_f_names=False):
    # Choose one function
    func_name = funcs.f_names[ random.randint(0, len(funcs.f_names) - 1 ) ] # 'f_e_ellipse'
    # Generate the two panels
    panel_ricker, panel_psf = generate_curves( funcs, func_name, False )

    if debug: 
        show(panel_ricker, title="Panel Ricker", vmin=-1, vmax=1, color='seismic')
        show(panel_psf, title="Panel PSF", vmin=-1, vmax=1, color='seismic')
    elif save_data:
        save(panel_ricker, str(i) + '_structure_ricker.png', 'seismic', vmin=-1, vmax=1)
        save(panel_psf, str(i) + '_structure_psf.png', 'seismic', vmin=-1, vmax=1)

    # PSF ******************************************************************************
    # Choose a PSF
    psf = get_psf()
    # Convolve with its panel
    panel_psf = convolve2d(panel_psf, psf, mode='same')

    # Add Gaussian noise
    # filter_sigma = 2.0
    # signal_to_noise = 0.1
    # panel_psf = add_noise(panel_psf, filter_sigma=filter_sigma, signal_to_noise=signal_to_noise)
    
    # Normalize
    panel_psf = range_normalize_std( panel_psf )
    # PSF ******************************************************************************


    # Ricker ***************************************************************************
    # Params to ricker wavelet
    taper_size = 4
    wavelet_window_size = 50

    # Convolve the panel with ricker wavelet
    panel_ricker = convolve_wavelet(panel=panel_ricker, frequency=frequency, dt=dt, taper_size=taper_size, window_size=wavelet_window_size)
    
    # Normalize
    panel_ricker = range_normalize_std( panel_ricker )
    # PSF *************'    *****************************************************************
    
    # If debug, don't save imagens, just show.
    if debug: 
        show(panel_ricker, title="Ricker", vmin=-1, vmax=1, color='seismic')
        show(panel_psf, title="PSF", vmin=-1, vmax=1, color='seismic')
    elif save_data:
        save(panel_ricker, str(i) + posfix_name + 'ricker.png', 'seismic', vmin=-1, vmax=1)
        save(panel_psf, str(i) + posfix_name + 'psf.png', 'seismic', vmin=-1, vmax=1)


    # Check if is to put the function name on image
    if save_data and print_f_names:
        # Read image
        im = cv2.imread(output_folder + str(i) + posfix_name + 'ricker.png')
        # Add the label (Function name)
        set_img_label(im, funcs.get_str_eq( func_name ), (25, 50))
        # Replace image on disk
        cv2.imwrite(output_folder + str(i) + posfix_name + 'ricker.png', im)

    return panel_ricker, panel_psf, funcs.get_str_eq( func_name )


if __name__ == '__main__':  

    # Starting time
    timer = Timer()
    timer.start()

    debug = False
    print_f_names = True
    posfix_name = '_data_'
    
    for i in tqdm(range(1)):
        get_sample(debug=debug, save_data=not debug, posfix_name=posfix_name, print_f_names=print_f_names)
            
    print( timer.diff() )
