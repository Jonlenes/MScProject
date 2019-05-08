import random
import numpy as np
import math

from config import panel_side_base
from img_util import MatplotlibUtil


def _bring_to_zero(array):
    return array - array.min()


def _bring_to_zero2(array):
    v_min = 0
    for a in array: v_min = min(v_min, min(a))
    for i in range(len(array)): array[i] = array[i] - v_min 
    return array, v_min


def scale(x, y, y_size, x_bound):
    # Bring the y to zero
    x = _bring_to_zero(x)
    y = _bring_to_zero(y)
    
    # Rescaling
    # Limite de tela em x divido pelo max da função em x
    # Limite de tela em y divido pelo max da função em y
    factor_x, factor_y = (x_bound[1] - x_bound[0]) / x.max(), y_size / y.max()
    
    return (x * factor_x) + x_bound[0] , (y * factor_y)


def var():
    return random.randint(-15, 15)


# a = random.randint(-2, 2)/10.0 # Add a inclination
# c = .05 # Frequency 
# d = random.randint(7, 13) # Amplitude

class Functions():
    def __init__(self):
        self.a = 0
        self.d = 1
        self._buid_functions_names()


    def random_params(self):
        self.a = random.randint(-3, 3) / 10.0 # Add a inclination
        self.d = random.randint(0, 20)


    def _buid_functions_names(self):
        attrs = dir(self)
        self.f_names = []

        for attr in attrs:
            if attr[0] == 'f':
                self.f_names.append( attr )


    def get_str_eq(self, func):
        if func not in self.f_names: return ""
        return getattr(self, func)(func_to_str=True)


    def __call__(self, func):
        if func not in self.f_names:
            raise Exception('Function ' + func + ' isn\'t avaliable. Check the docs or codes.')
        if '_e_' in func:
            return getattr(self, func)() 
        # Generate x for all panel by .1
        x = np.arange(0, panel_side_base)
        # Compute Y
        y = getattr(self, func)(x)

        return x, y


    def f1(self, x=None, func_to_str=False, c=None, d=None): #func_sin():
        if func_to_str: return 'f1: f(x)=ax+d*sin(c*x)'
        if c is None: c = .05
        if d is None: d = self.d + random.randint(5, 10) # Amplitude
        return self.a * x + d * np.sin(c * x)


    def f2(self, x=None, func_to_str=False): #func_sin_2():
        if func_to_str: return 'f2: f(x)=ax+d*sin((cx)**2)'
        c = .01
        d = self.d + random.randint(0, 5) # Amplitude
        # return self.a * x + d * np.sin(c * (x**2))
        return self.a * x + d*np.sin((c*x)**2)
        

    def f3(self, x=None, func_to_str=False):
        if func_to_str: return 'f3: f(x) = a(x^2) + 2x + 3'
        y = self.a * (x**2) + 2*x + 3
        x, y = scale(x, y, panel_side_base*.4, (0, panel_side_base))
        return y
        

    def f12(self, x=None, func_to_str=False):
        if func_to_str: return 'f12: f1( f2(x) )'
        return self.f1( self.f2(x) )
        

    def f21(self, x=None, func_to_str=False):
        if func_to_str: return 'f21: f2( f1(x) )'
        return self.f2( self.f1(x) )


    def f31(self, x=None, func_to_str=False):
        if func_to_str: return 'f31: f3( f1(x) )'
        return self.f3( self.f1(x, c=.1, d=random.randint(5, 8)) )


    def f_wave(self, x=None, func_to_str=False):
        if func_to_str: return 'wave: f(x)=ax + d*cos( cx )'
        c = .05 # Frequency 
        d = random.randint(11, 20) # Amplitude

        return self.a*x + d*np.cos(c * x)
    

    def out_f_senoide(self, x=None, func_to_str=False):
        if func_to_str: return 'senoide: '
        c = 0 # Frequency
        b = 1024
        d = 1
        # d = random.randint(11, 20) # Amplitude

        return b*(np.exp(-(x-c)**2) + d * np.exp(-(2*x+c)**2))


    def interpolate_points(self, xs, ys):
        def f(a, b, x): return a*x + b
        # from scipy.interpolate import interp1d
        xs = np.append( xs, xs[0] )
        ys = np.append( ys, ys[0] )

        i_xs, i_ys = [], []
        len_ = len(xs)
        for i in range(len_ - 1):
            x = xs[ i:(i+2) ]
            y = ys[ i:(i+2) ]
            # print( x.shape, y.shape )
            # f = interp1d(x, y)
            a = (y[1] - y[0]) / (x[1] - x[0])
            b = y[0] - a * x[0]
            # if (i == 0): print(x, x.min(), x.max())
            x = np.arange(x.min(), x.max())
            # if (i == 0): print(x, x.min(), x.max())
            i_xs.extend( x )
            i_ys.extend( f(a, b, x) )
        
        # print(i_xs)
        # print(i_ys)
        return i_xs, i_ys


    def f_e_ellipse(self, x=None, func_to_str=False):
        if func_to_str: return 'ellipse: \'ellipse\''
        
        def scale_vector(x, y, scale):
            return (x * scale, y * scale);

        def dist(x, y):
            return np.sqrt(x**2 + y**2)
    
        def sign(a, var):
            if math.isclose(a, 0): 
                return -sign(var, 1)
            return a / abs(a)
        
        def rotate(x, y, ang):
            return x*np.cos(ang)-y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)
        
        def rotate_polygon(xs, ys, ang):
            for i in range(len(xs)):
                x, y = rotate( xs[i], ys[i], ang )
                xs[i] = x 
                ys[i] = y
            return xs, ys

        def get_scale(x, y, a):
            if math.isclose(self.var, self.len):
                if self.count < 3:
                    self.count += 1
                else:
                    self.count = 0
                    self.len = -sign(self.len, self.var) * abs(np.random.randint(-10, 10))
               
            if a == 360: self.var = 0

            d = dist(x, y)
            s = (d + self.var) / d
            if self.count == 0:
                self.var += sign( self.len, self.var )
            return s

        l_x, l_y = [], []

        a = 200
        b = 100
        step = np.random.randint(10, 20)
        dmax = panel_side_base
        count = 0

        xs, ys = np.zeros((360)), np.zeros((360))
        self.var = 0
        self.len = np.random.randint(-10, 10)
        self.count = 0
        for i in range(0, 360):
            angle = (i * np.pi) / 180.0

            x = a * np.cos( angle )
            y = b * np.sin( angle )

            x, y = scale_vector(x, y, get_scale(x, y, i))
            xs[i] = x
            ys[i] = y

        scale = 1
        i = 0
        c = [0, 0]
        while (i + 1) * step < dmax:
            x_t, y_t = scale_vector(xs, ys, scale)
            if i == 0:
                num_ptos = len(xs)
                c[0] = xs.sum() / num_ptos
                c[1] = ys.sum() / num_ptos

            # if abs(x_t[0] - x_t[1]) > 2: 
            x_t, y_t = self.interpolate_points( x_t, y_t )
            l_x.append( x_t )
            l_y.append( y_t )
            scale = (i + 1) * step / b
            i += 1
            
        # Move figure to (x, y) >= 0    
        l_x, x_min = _bring_to_zero2( np.array(l_x) )
        l_y, y_min = _bring_to_zero2( np.array(l_y) )

        # Move centroid
        c[0] -= x_min
        c[1] -= y_min

        # Compute difference
        diff_x = c[0] - panel_side_base/2
        diff_y = c[1] - panel_side_base/2

        return l_x - diff_x, l_y - diff_y