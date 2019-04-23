import numpy as np
import matplotlib.pyplot as plt
from time import time
from numpy.random import randint as ri
from main import plot_on_nparray
from io_util import show, save
from img_util import MatplotlibUtil
import cv2
from utils import *


t = time()
size = 1024
c_start = 3
b = ri(20, 30)
factor = 1
y_step = -(b + 10)
coeficient = 0

MatplotlibUtil.full_frame(size, size)
plt.ylim(0, size)
plt.xlim(0, size)


def f(x, a):
	return (x**2) / 3 #(2 + a/10) # return 1./x # 3 * (x ** 2) + x - 14  

def f_domo(x, a, b, c, puts):
	return a - b*np.exp(-(x - f(c, puts))**2)

def get_cof(cofs, count):
	if count >= len(cofs):
		cofs.append( cofs[ len(cofs) - 1 ] + 1 )
	return cofs[ count ]

if __name__ == '__main__': 
	
	list_cof = [1]
	for i in range(1): 	
		factor = ri(1, 5)
		c_step = ri(1, 7)
		c_start_step = ri(2, 3)
		panel = np.zeros((size, size))
		
		for a in range(size, 0, y_step):
			x_start = 0
			c = c_start
			coeficient += 1
			count = 0
			fim = False
		
			while True:
				x_end = f(c, get_cof(list_cof, count)) + 5
				if x_end > size: 
					x_end = size - 1
					fim = True
				
				x = np.arange(x_start - 1, x_end)
				y = f_domo(x, a, b, c, get_cof(list_cof, count))
				plot_on_nparray(panel, x, y, size)
				plt.plot(x, y, color='black')
				x_start = x_end
				c += c_step
				count += 1

				if fim: break
				# break;
			
			c_start = c_start + c_start_step

		print ("Time: ", time() - t)
		plt.savefig("c_temp.png")

		panel = cv2.imread('c_temp.png', 0)
		# Params to ricker wavelet
		taper_size = 4
		wavelet_window_size = 50
		dt = 0.004
		frequency = 20 #random.randint(5, 25)

        # Convolve the panel with ricker wavelet
		panel = convolve_wavelet(panel=panel, frequency=frequency, dt=dt, taper_size=taper_size, window_size=wavelet_window_size)
		panel = range_normalize_std( panel )
		show(panel, vmin=-1, vmax=1, color='seismic')
		# plt.clf()
