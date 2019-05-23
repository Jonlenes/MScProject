import numpy as np
import random

from io_util import show, save
from sample_generator import get_psf
from scipy.signal import convolve2d
from cepetro_util import convolve_wavelet
from dataset_generator import process_sample
from config import output_folder


if __name__ == '__main__':
    data_psf = np.zeros((256, 256))
    data_ricker = np.zeros((256, 256))
    
    """for i in range(32, 256, 32):
        for j in range(32, 256, 32):
            panel[i, j] = random.uniform(-1,1)
    """
    for i in range(0, 256-40, 41):
        for j in range(0, 256-40, 41):
            sub_panel = np.zeros((41, 41))
            sub_panel[20, 20] = random.uniform(-1,1)
            psf = get_psf()
            data_ricker[i:i+41, j:j+41] = sub_panel
            sub_data = convolve2d(sub_panel, psf, mode='same')
            data_psf[i:i+41, j:j+41] = sub_data

    data_ricker = convolve_wavelet(panel=data_ricker, frequency=20, dt=0.004, taper_size=4, window_size=50)
    process_sample(i, output_folder, [data_ricker], [data_psf], "psf_vs_ricker:")

    # show( data_psf, vmin=-1, vmax=1, color="seismic")
    # show( data_ricker, vmin=-1, vmax=1, color="seismic")



