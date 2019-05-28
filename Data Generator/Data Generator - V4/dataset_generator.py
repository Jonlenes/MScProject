import sample_generator as sg

from io_util import save
from config import output_folder, panel_side_base, output_panel_side

from timer import Timer 

from joblib import Parallel, delayed
from tqdm import tqdm 
from glob import glob

import time
import os
import random
import multiprocessing as mp
import numpy as np
import shutil

extra_random_samples = 2
samples_by_data = int(panel_side_base / output_panel_side)*2 + extra_random_samples
aligned = False


def split_data(synthetic_ricker2D, synthetic_psf):
    psfs, rickers = [], []
    step = output_panel_side
    for i in range(step, synthetic_psf.shape[0], step):
        for j in range(step, synthetic_psf.shape[1], step):
            psfs.append(synthetic_psf[i-step:i, j-step:j])
            rickers.append(synthetic_ricker2D[i-step:i, j-step:j])
    
    for i in range(extra_random_samples):
        x = random.randint(1, synthetic_psf.shape[0] - step)
        y = random.randint(1, synthetic_psf.shape[0] - step)
        psfs.append(synthetic_psf[y:y+step, x:x+step])
        rickers.append(synthetic_ricker2D[y:y+step, x:x+step])
    return psfs, rickers


def process_sample(index_block, path, xs=None, ys=None, data_name=None):
    if xs is None:
        data_ricker, data_psf, data_name = sg.get_sample()
        xs, ys = split_data(data_ricker, data_psf)
    
    img_path = path + "/{0}{1}_" + data_name.split(":")[0] + ".png"
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        if aligned:
            img = np.append(x, y, axis=1)
            save(img, img_path.format("", str(index_block + i)), color='gray', use_out_folder=False)
        else:
            save(x, img_path.format("A/", str(index_block + i)), color='gray', use_out_folder=False)
            save(y, img_path.format("B/", str(index_block + i)), color='gray', use_out_folder=False)


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False


def create_dataset(n_samples):
    timer = Timer()
    timer.start()
    path = output_folder

    folders = [ ("train", n_samples), 
                ("val", int(n_samples * 0.2)), 
                ("test", int(n_samples * 0.2)) ]
    
    total_datas = int(n_samples * 1.4) + 1
    print("Total of data:", total_datas)
    
    nc = mp.cpu_count()
    print("Cpu count: ", nc)

    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        time.sleep(2)
        os.mkdir(path)    
    
    create_folder(path)
    if not aligned:
        create_folder(path + "/A")
        create_folder(path + "/B")
     
    # Generate data parallel        
    Parallel(n_jobs=nc)(delayed(process_sample) (i, path) for i in tqdm(range(0, total_datas, samples_by_data)))
    
    # Load data paths
    if aligned: data_paths = glob(path + "/*.png")
    else: data_paths = glob(path + "/A/*.png")
        
    # Shuffle the data 
    np.random.shuffle(data_paths)
    start = 0

    for folder in folders:
        # Paths
        folder_path = path + folder[0]
        if aligned:
            create_folder( folder_path )
        else:    
            create_folder( folder_path + "_A")
            create_folder( folder_path + "_B")
            
        # File names 
        data_paths_folder = data_paths[ start:start+folder[1] ]
        
        # Move files
        for data_path in data_paths_folder:
            if aligned:
                shutil.move(data_path, folder_path + "/" + data_path.replace("\\", "/").split("/")[-1])
            else:
                # Dado A
                shutil.move(data_path, folder_path + "_A/" + data_path.replace("\\", "/").split("/")[-1])
                # Dado B
                data_path = data_path.replace("A", "B")
                shutil.move(data_path, folder_path + "_B/" + data_path.replace("\\", "/").split("/")[-1])

        # Update start
        start += folder[1] 
        
        if not aligned: folder_path += "_A"
        print("Total images of {0}: {1}".format(folder[0], len(glob(folder_path + "/*.png"))))
        
    print( timer.diff() )


if __name__ == '__main__':
    create_dataset(10000)
