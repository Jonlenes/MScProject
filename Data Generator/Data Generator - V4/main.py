import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import random
import cv2
import math

from img_util import MatplotlibUtil, set_img_label
from functions import Functions, _bring_to_zero
from io_util import show, save
from config import panel_side_base
from timer import Timer
from tqdm import tqdm
from utils import *
from scipy.signal import convolve2d


def init():    
    # Object of functions
    return Functions()


def normalize(x, lower=-1, upper=1):
    return ((x - x.min()) / (x.max() - x.min()) * (upper - lower) + lower)


def plot_on_nparray(array, x, y, panel_size=None):
    if len(x) == 0: return
    if panel_size is None: panel_size = panel_side_base
    
    x = np.round(x).astype(int)
    y = np.round(y).astype(int) #+ np.random.randint(-1, 1, (len(y)))
    
    valid_idxs1 = np.logical_and((x >= 0), (x < panel_side_base))
    valid_idxs2 = np.logical_and((y >= 0), (y < panel_side_base))

    valid_idxs = np.where( np.logical_and(valid_idxs1, valid_idxs2) )
    
    if len(x[valid_idxs]) == 0: return
    
    reflexivity = random.uniform(-1,1)
    array[ y[valid_idxs], x[valid_idxs] ] = reflexivity
        

def generate_curves(funcs, func_name, verbose=True, artifacts=True):
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

    # Add noise, artifact and faul
    if artifacts:
        # V1
        # sizes = [3, 20, 40, 60]
        # qtds = [1000, 1000, 300, 100]

        # V2
        sizes = [3, 15, 25, 50]
        qtds = [1000, 500, 150, 50]
        
        
        for i in range(len(sizes)):
            size = sizes[i]
            # Little trace 
            for j in range( qtds[i] ):
                x = np.random.randint(size, panel_side_base-size)
                y = np.random.randint(size, panel_side_base-size)

                v_straight = np.arange(0, size)
                sign = random.choice([True, False])
                angle = (np.random.randint(0, 360) * np.pi) / 180.0
                
                if sign: 
                    o_straight = np.tan(angle) * v_straight 
                    o_straight = _bring_to_zero( o_straight )
                    plot_on_nparray(panel, v_straight + x, o_straight + y)
                else:
                    if not math.isclose(np.tan(angle), 0):
                        o_straight = v_straight / np.tan(angle) 
                        o_straight = _bring_to_zero( o_straight )
                        plot_on_nparray(panel, o_straight + x, v_straight + y)

    return panel


def get_psf():
    # Seleciona um PSF para convolver com o dado de estruturas
    # A ultima PSF não pode ser inserida pois está com a polaridade invertida e a Ricker não.
    index = random.randint(0, 4)

    return np.load('PSFs/psf_' + str(index) + '.npy')
    


if __name__ == '__main__':  

    timer = Timer()
    timer.start()

    # Needed configs before data generation 
    funcs = init()
    
    debug = False
    psf = False
    print_f_names = True
    posfix_name = '_data.png'
    number_of_funs = len(funcs.f_names) 

    for i in tqdm(range(10)):
        func_name = funcs.f_names[ random.randint(0, number_of_funs - 1 ) ]
        panel = generate_curves( funcs, func_name, False )

        if debug: 
            print("Here 5")
            show(panel, vmin=-1, vmax=1, color='seismic')
        else:
            save(panel, str(i) + '_structure.png', 'seismic', vmin=-1, vmax=1)

        if psf:
            
            psf = get_psf()
            print( "PSF Infos:", psf.min(), psf.max() )
            panel = convolve2d(panel, psf, mode='same')

        else:
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
