from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np


def _show(img, scala, color, vmin, vmax, show_axis):
    fig, ax = plt.subplots()
    dpi = float(fig.get_dpi())
    img_width = int(img.shape[0] / dpi)
    img_height = int(img.shape[1] / dpi)
    fig.set_size_inches(scala * img_width, scala * img_height)
    if not show_axis: ax.axis("off")
    ax.imshow(img, cmap=color, vmin=vmin, vmax=vmax, interpolation='none')
    fig.show()
    plt.show(block=True)
    
def show(img, scala=1, color='gray', vmin=None, vmax=None, show_axis=False):
    process = Process(target=_show, args=(img, scala, color, vmin, vmax, show_axis))
    process.start()

    return process
        
        
def save(img, path, color=None, vmin=None, vmax=None):
    if vmin is None and color == 'seismic':
        vmax = abs(img).max()
        vmin = -vmax
    plt.imsave(path, img, cmap=color, vmin=vmin, vmax=vmax)
