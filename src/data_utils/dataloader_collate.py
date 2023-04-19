"""Dataloader Collate.

The collate function of the dataloader has to be modified, since we return
4 objects instead of just 1 in the WidarDataset.

Author:
    Yvan Satyawan
"""
import torch


def widar_collate_fn(samples):
    """Collate function for the Widar dataset.

    Since we return a tuple in the dataset, we need a custom collate_fn to
    produce the tensors we need for training. There is also the possibility that
    the WidarDataset returns Nones. In this case, we return None for the entire
    item batch.
    """
    if samples[0][0] is None:
        amps = None
    else:
        amps = torch.stack([sample[0] for sample in samples])
    if samples[0][1] is None:
        phases = None
    else:
        phases = torch.stack([sample[1] for sample in samples])
    if samples[0][2] is None:
        bvps = None
    else:
        bvps = torch.stack([sample[2] for sample in samples])
    infos = [sample[3] for sample in samples]

    return amps, bvps, phases, infos


if __name__ == '__main__':
    from pathlib import Path

    from torch.utils.data import DataLoader

    from data_utils.widar_dataset import WidarDataset

    dataset = WidarDataset(Path("../../data"), "train", True)
    dataloader = DataLoader(dataset, collate_fn=widar_collate_fn, batch_size=10)

    for i in dataloader:
        print(i)
        break
