# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:04:36 2019

Significant changes made by Yvan Satyawan.

@author: Ruibzhan
@author: Yvan Satyawan
"""
import math
from pathlib import Path
from typing import Iterable, List, Sized

from multiprocessing import Pool, cpu_count
import pandas as pd
import signal_to_image.refined.para_hill as para_hill
import pickle
import numpy as np
from itertools import product
from tqdm import tqdm


def split_into_chunks(data_list: Sized, num_chunks: int) -> list:
    """Does a best try split of a data list into chunks.

    Last chunk may have fewer values in it.
    """
    data_amount = len(data_list)
    heap_size = math.ceil(data_amount / num_chunks)
    chunks = []
    for proc_idx in range(num_chunks):
        try:
            chunk = data_list[heap_size * proc_idx:heap_size * (proc_idx + 1)]
        except IndexError:
            chunk = data_list[heap_size * proc_idx:]
        chunks.append(chunk)

    return chunks


def generate_centroids(init_coord: tuple, image_dim: int) -> List[tuple]:
    """Generates centroids from initial coordinates."""
    xxx = [init_coord[0] + i * 3
           for i in range(int(image_dim / 3) + 1)
           if (init_coord[0] + i * 3) < image_dim]
    yyy = [init_coord[1] + i * 3
           for i in range(int(image_dim / 3) + 1)
           if (init_coord[1] + i * 3) < image_dim]
    centroid_list = list(product(xxx, yyy))

    return centroid_list


def hill_climb_master(output_dir: Path, mds_fp: Path, antenna_number: int,
                      num_iters: int = 5):
    """Performs a hill climb algorithm with multiprocessing."""
    output_fp = output_dir / f"mapping_{antenna_number}.pkl"

    if output_fp.exists():
        print(f"Hill climb calculation exists. Skipping...")
        return

    print(f"Performing hill climb...")
    print(f"Processors found: {cpu_count()}")

    with open(mds_fp, "rb") as f:
        dist_matr, map_in_int = pickle.load(f)

    num_feats = dist_matr.shape[0]  # Nn
    image_dim = math.ceil(math.sqrt(num_feats))
    corr_evol = []

    corr = para_hill.universial_corr(dist_matr, map_in_int)
    prog_bar = tqdm(desc=f"Correlation: {float(corr):.3e}",
                    total=num_iters * 9)

    for i in range(num_iters):
        init_coords = [x for x in product([0, 1, 2], repeat=2)]

        # Generate the centroids
        for init_coord in init_coords:
            centroid_list = generate_centroids(init_coord, image_dim)

            with Pool(cpu_count()) as p:
                chunks = split_into_chunks(centroid_list, cpu_count())
                chunks = [(chunk, num_iters, map_in_int, dist_matr)
                          for chunk in chunks]
                swap_dicts = p.starmap(hill_climb_slave, chunks)

            swap_dict = dict()
            for d in swap_dicts:
                swap_dict.update(d)

            map_in_int = para_hill.execute_dict_swap(swap_dict, map_in_int)

            corr = para_hill.universial_corr(dist_matr, map_in_int)
            prog_bar.set_description(f"Correlation: {float(corr):.3e}")
            corr_evol.append(corr)
            prog_bar.update()

    coords = np.array(
        [[item[0] for item in np.where(map_in_int == ii)]
         for ii in range(num_feats)]
    )
    with open(output_fp, 'wb') as file:
        pickle.dump([coords, map_in_int], file)

    corr_evol = pd.DataFrame(corr_evol, columns=['Correlation'])
    corr_evol.to_csv(output_dir / f"corr_evol_{antenna_number}.csv",
                     index=False)


def hill_climb_slave(centroid_list: Iterable, num_iters: int,
                     map_in_int: list, dist_matr: list):
    swap_dicts = {}
    for i in range(num_iters):
        for _ in range(9):
            swap_dicts.update(para_hill.evaluate_centroids_in_list(
                centroid_list, dist_matr, map_in_int
            ))

    return swap_dicts


if __name__ == '__main__':
    for split in ("train", "test"):
        hill_climb_master(output_dir=Path("../../../data/refined_images",),
                          num_iters=5)
