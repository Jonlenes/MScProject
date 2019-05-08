import matplotlib
import matplotlib.pyplot as plt
import random
import cv2
import math
import numpy as np

from img_util import MatplotlibUtil, set_img_label
from functions import Functions, _bring_to_zero, scale
from io_util import show, save
from config import panel_side_base
from timer import Timer
from tqdm import tqdm
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from sample_generator import plot_on_nparray


def test():
	def rotate(x, y, a):
		return (x*np.cos(a) - y*np.sin(a), x * np.sin(a) + y * np.cos(a));

	def scale_vector(x, y, scale):
		return Vetor(x * scale, y * scale);

	def d(x, y):
		return np.sqrt(x**2 + y**2)
	
	a = 200
	b = 100

	angle = (60 * np.pi) / 180.0

	x_ok = a * np.cos( angle )
	y_ok = b * np.sin( angle )

	print("(X, Y) = ", x_ok, y_ok)
	print("Len =", d(x_ok, y_ok))

	h = np.sqrt(x_ok ** 2 + y_ok ** 2)
	print("h = ", h)
	x, y = rotate(h, 0, angle)
	print("(X, Y) = ", x, y)
	print("Len =", d(x, y))
            
        


if __name__ == '__main__': 
	f = Functions()
	xs, ys = f.f_e_ellipse()
	# x, y = scale(np.array(x), np.array(y), panel_side_base, (0, panel_side_base))
	panel = np.zeros((panel_side_base + 10, panel_side_base + 10))
	"""
	from scipy.interpolate import interp1d
	new_x, new_y = [], []
	len_ = len(xs[-1])
	for i in range(len_ - 2): #, ys[-1]):
		x = xs[-1][i:(i+2)%len_]
		y = ys[-1][i:(i+2)%len_]

		print(x.shape, y.shape)

		f = interp1d(x, y)
	 	x = np.linspace(x.min(), x.max())
		new_x.append( x )
		new_y.append( f(x) )

	plt.plot(new_x, new_y, 'o', linewidth=1)
	"""


	for a, b in zip(xs, ys):
		plot_on_nparray(panel, a, b)
		# from scipy.interpolate import interp1d
		# f = interp1d(a, b)
		# x = np.linspace(a.min(), a.max(), num=11, endpoint=True)
		# print( len(a), len(b) )
		# print( a )
		# plt.plot(a, b)
		# break
	# plt.show()
	show(panel, 'Sts', 'seismic')

"""
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
"""

# For each curve, set 'refletivity' as [-1, +1] value
# back = panel == 0
# panel[ back ] = panel.mean()
# panel = normalize( panel )
# panel[ back ] = 0


# Fill the panel with the curves
"""y_pos = -(0.3*panel_side_base)
while y_pos < int(1.3 * panel_side_base): 
    x, y = funcs( func_name )
    # x, y = scale(x, y, 25, (0, panel_side_base))
    
    plt.plot(x, y + y_pos, linewidth=1, color=str(np.random.uniform(.5, 1))) # 
    y_pos += random.randint(15, 15)

# plt.show()

# Export the data and convert to Gray
panel = MatplotlibUtil.fig2data( plt.gcf() )
panel = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)

plt.clf()
if verbose:
    print( panel.shape, panel.min(), panel.max() )
    
# Return negative panel normalized between 0 and 1
return (255 - panel) / 255."""
    