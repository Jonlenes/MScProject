import random
import numpy as np
from config import panel_side_base


def _bring_to_zero(array):
    return array - array.min()


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
        d = self.d + random.randint(5, 10) # Amplitude
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
    

    def f_senoide(self, x=None, func_to_str=False):
        if func_to_str: return 'aaaa: f(x)=ax + d*cos( cx )'
        c = 0 # Frequency
        b = 1024
        d = 1
        # d = random.randint(11, 20) # Amplitude

        return b*(np.exp(-(x-c)**2) + d * np.exp(-(2*x+c)**2))


    def f_domo(self, x=None, func_to_str=False):
        if func_to_str: return 'bbbb: f(x)=ax + d*cos( cx )'
        b = 1
        c = 0

        return -b*np.exp(-(x - c)**2)


    """case 'senoide'
    R = a + b*(exp(-sig*(x-c).^2)+exp(-sig*(2*x+c).^2));

        case 'domo'
    R = a -b*exp(-sig*(x-c).^2);"""