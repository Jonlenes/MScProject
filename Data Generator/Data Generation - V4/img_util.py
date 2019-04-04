import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
import operator as op



class MatplotlibUtil:    
    
    @staticmethod
    def full_frame(width=None, height=None):
        mpl.rcParams['savefig.pad_inches'] = 0
        figsize = None if width is None else (width/100, height/100)
        plt.figure(figsize=figsize, dpi=100)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)


    @staticmethod
    def fig2data( fig ):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()
    
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
        buf.shape = ( w, h, 3 )
    
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = numpy.roll ( buf, 3, axis = 2 )
        return buf

def set_img_label(im, label, pt_ori):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2

    def _sum_tuple(t1, t2):
        return tuple(map(op.add, t1, t2))
    
    
    size, baseline = cv2.getTextSize(label, fontface, scale, thickness)
    p1 = _sum_tuple(pt_ori, (0, baseline))
    p2 = _sum_tuple(pt_ori, (size[0], -size[1]))
    p1 = _sum_tuple(p1, (-10, +5))
    p2 = _sum_tuple(p2, (+10, -10))
    
    cv2.rectangle(im, p1, p2, (255, 255, 255), -1)
    cv2.putText(im, label, pt_ori, fontface, scale, (0, 0, 0), thickness, 8)
