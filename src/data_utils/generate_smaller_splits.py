"""Generate Small Splits.

Creates a dir for each dataset split in the small set.
"""
import argparse
import pickle
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2

from tqdm import tqdm


def parse_args():
    p = ArgumentParser()
    p.add_argument("DATA_FP", type=Path,
                   help="Path to the Widar3.0 directory.")
    p.add_argument("SPLIT_TYPE", type=str,
                   help="Type of split to process. Options are `small` or "
                        "`single_domain`")

    # Verify that SPLIT_TYPE is one of the possible options
    args = p.parse_args()
    if args.SPLIT_TYPE not in ("small", "single_domain"):
        raise ValueError(f"SPLIT_TYPE must be one of `small` or "
                         f"`single_domain`. Got {args.SPLIT_TYPE} instead.")

    return args


def copy_files(widar_dir: Path, split_type: str):
    split_names = ("train",
                   "validation",
                   "test_room",
                   "test_location",
                   "test")

    for split_name in split_names:
        pkl_fp = widar_dir / f"{split_name}_index_{split_type}.pkl"
        if not pkl_fp.exists():
            continue
        with open(pkl_fp, "rb") as f:
            samples = pickle.load(f)['samples']

        split_dir = widar_dir / f"widar_{split_type}" / split_name
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

        for sample in tqdm(samples, desc=f"Copying files for {split_name}"):
            for csi_path_key in csi_path_keys:
                for csi_file in sample[csi_path_key]:
                    copy2(src=csi_file,
                          dst=split_dir / csi_file.name)

            for bvp_path_key in bvp_path_keys:
                copy2(src=sample[bvp_path_key],
                      dst=split_dir / sample[bvp_path_key].name)


if __name__ == '__main__':
    parsed_args = parse_args()
    copy_files(parsed_args.DATA_FP, parsed_args.SPLIT_TYPE)
