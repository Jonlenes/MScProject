from multiprocessing import Process
import matplotlib.pyplot as plt
import numpy as np

from config import output_folder


def _show(img, scala, color, vmin, vmax, show_axis, title):
    fig, ax = plt.subplots()
    dpi = float(fig.get_dpi())
    img_width = int(img.shape[0] / dpi)
    img_height = int(img.shape[1] / dpi)
    print( img_width, img_height )
    fig.set_size_inches(scala * img_width, scala * img_height)
    if not show_axis: ax.axis("off")
    ax.imshow(img, cmap=color, vmin=vmin, vmax=vmax, interpolation='none')
    ax.set_title(title)
    fig.show()
    plt.show(block=True)
    
def show(img, title='title', scala=1, color='gray', vmin=None, vmax=None, show_axis=False):
    process = Process(target=_show, args=(img, scala, color, vmin, vmax, show_axis, title))
    process.start()

    return process
        
        
def save(img, path, color=None, vmin=None, vmax=None, use_out_folder=True):
    if vmin is None and color == 'seismic':
        vmax = abs(img).max()
        vmin = -vmax
    if use_out_folder:
        path = output_folder + path
    plt.imsave(path, img, cmap=color, vmin=vmin, vmax=vmax)
