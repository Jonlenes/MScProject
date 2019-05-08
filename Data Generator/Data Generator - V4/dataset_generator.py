import sample_generator as sg

from io_util import save
from config import output_folder, samples_by_data

from timer import Timer 

from time import time
from joblib import Parallel, delayed
from tqdm import tqdm 
from glob import glob

import os
import random
import multiprocessing as mp
import numpy as np
import shutil


def split_data(synthetic_ricker2D, synthetic_psf):
    psfs, rickers = [], []
    for i in range(256, synthetic_psf.shape[0], 256):
        for j in range(256, synthetic_psf.shape[1], 256):
            psfs.append(synthetic_psf[i-256:i, j-256:j])
            rickers.append(synthetic_ricker2D[i-256:i, j-256:j])
    
    for i in range(8):
        x = random.randint(1, 768)
        y = random.randint(1, 768)
        psfs.append(synthetic_psf[y:y+256, x:x+256])
        rickers.append(synthetic_ricker2D[y:y+256, x:x+256])
    return psfs, rickers


def process_sample(index_block, path):
    data_ricker, data_psf, data_name = sg.get_sample()
    xs, ys = split_data(data_ricker, data_psf)
    for i, (x, y) in enumerate(zip(xs, ys)):
        img = np.append(x, y, axis=1)
        name = path + "/" + str(index_block + i) + "_" + data_name.split(":")[0] + ".png"
        save(img, name, color='gray', use_out_folder=False)


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False


def create_dataset(n_samples, clear_folder=False):
    timer = Timer()
    timer.start()
    path = output_folder

    folders = [ ("train", n_samples), 
                ("val", int(n_samples * 0.2)), 
                ("test", int(n_samples * 0.2)) ]
    
    total_datas = int(n_samples * 1.4) + 1
    
    nc = mp.cpu_count()
    print("Cpu count: ", nc)

    if clear_folder and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)    
    
    start = 0
    if not create_folder(path):
        files = glob(path + "/*.png")
        if len(files) > 0: 
            start = len(files)

    # Generate data parallel        
    Parallel(n_jobs=nc)(delayed(process_sample) (i, path) for i in tqdm(range(start, total_datas, samples_by_data)))
    
    data_names = glob(path + "/*.png")
    np.random.shuffle(data_names)
    start = 0

    for folder in folders:
        # Path
        path_temp = path + folder[0]
        # Create folder
        create_folder( path_temp )
        # File names 
        sub_names = data_names[ start:start+folder[1] ]
        # Move files
        for name in sub_names:
            # print(name, name.split("/")[-1], path_temp + "/" + name.split("/")[-1])
            shutil.move(name, path_temp + "/" + name.replace("\\", "/").split("/")[-1])

        # Update start
        start += folder[1] 
        
        print("Total images of {0}: {1}".format(folder[0], len(glob(path_temp + "/*.png"))))
        
    print( timer.diff() )


if __name__ == '__main__':
    create_dataset(10000)