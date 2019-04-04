import numpy as np
import matplotlib.pyplot as plt
from time import time
from numpy.random import randint as ri

t = time()
size = 1024
c_start = 3
b = ri(5, 30)
factor = 1

def f(x): return (x**2)/factor
 	
factor = ri(1, 5)
c_step = ri(1, 7)
c_start_step = ri(1, 3)
for a in range(size, 0, -(b + 10)):
	x_start = 0
	c = c_start		
	while f(c) < size:
		x_end = f(c) + 5 if f(c + 5) < size else size
		x = np.arange(x_start, x_end)
		y = a - b*np.exp(-(x - f(c))**2)

		plt.plot(x, y, color='black')
		x_start = x_end
		c += c_step
	c_start = c_start + c_start_step

print "Time: ", time() - t
plt.show()
