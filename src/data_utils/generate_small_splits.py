"""Generate Small Splits.

Creates a h5 file for each dataset split in the small set.
"""
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2

from tqdm import tqdm
import pickle


def parse_args():
    p = ArgumentParser()
    p.add_argument("DATA_FP", type=Path,
                   help="Path to the Widar3.0 directory.")
    return p.parse_args()


def copy_files(widar_dir: Path):
    split_names = ("train",
                   "validation",
                   "test_room",
                   "test_location")
    for split_name in split_names:
        with open(widar_dir / f"{split_name}_index_small.pkl", "rb") as f:
            samples = pickle.load(f)

        split_dir = widar_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Figure out num_reps
        keys = list(samples[0].keys())
        csi_path_keys = []
        bvp_path_keys = []
        for key in keys:
            if key.startswith("csi_path"):
                csi_path_keys.append(key)
            elif key.startswith("bvp_path"):
                bvp_path_keys.append(key)

        for sample in tqdm(samples):
            # Abandoning this for now because the size of each receiver array
            # is not consistent between receivers.
            # csi_arrays = []
            # for key in csi_path_keys:
            #     csi_receivers = []
            #     for i in range(6):
            #         csidata = csiread.Intel(str(sample[key][i]))
            #         csidata.read()
            #         csi_receivers.append(
            #             csidata.get_scaled_csi_sm()[:, :, :, :1]
            #         )
            #     csi_arrays.append(csi_receivers)
            # breakpoint()
            for csi_path_key in csi_path_keys:
                for csi_file in sample[csi_path_key]:
                    copy2(src=csi_file,
                          dst=split_dir / csi_file.name)

            for bvp_path_key in bvp_path_keys:
                copy2(src=sample[bvp_path_key],
                      dst=split_dir / sample[bvp_path_key].name)


if __name__ == '__main__':
    args = parse_args()
    copy_files(args.DATA_FP)
