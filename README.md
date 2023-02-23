# DARLInG

Domain Auto-labeling through Reinforcement Learning for the Inference of Gestures

This is the repository for Yvan Satyawan's Master Thesis.

## Widar3.0

These are mostly notes to myself to understand what the data looks like.

The files are stored in a _lot_ of files, split into multiple folders.
Folder naming scheme doesn't have much meaning, other than to split the dataset into when the data was captured.

### CSI file

The CSI files are `.dat` files, which are simply CSI dumps from the tool used by the team to gather CSI data.
The file naming convention is as follows:

`id-a-b-c-d-Rx.dat`

| `id`    | `a`           | `b`            | `c`              | `d`               | `Rx`              |
|---------|---------------|----------------|------------------|-------------------|-------------------|
| User ID | Gesture Class | Torso Location | Face Orientation | Repetition Number | Wi-Fi Receiver ID |

Each recorded CSI sequence can be understood as an tensor with the shape (i, j, k, 1).
- i is the packet number
- j is the subcarrier number
- k is the receiver antenna number

In the case of Widar3.0, the value of k is always 3 (3 antennas per receiver).
Widar3.0 uses 1 transmitter and 6 receivers placed around the sensing area.

We use the package [csiread](https://github.com/citysu/csiread) to read the file.
The `.dat` files can be read using `csiread.Intel`.

### BVP file

The BVP files are `.mat` files (MATLAB) that have been preprocessed by the authors.
The file naming convention is as follows:

`id-a-b-c-d-suffix.dat`

| `id`    | `a`           | `b`            | `c`              | `d`               |
|---------|---------------|----------------|------------------|-------------------|
| User ID | Gesture Class | Torso Location | Face Orientation | Repetition Number |

`suffix` has no explanation. As far as I understand, I think it's just the configuration used to produce the BVP.
There is no receiver ID since all 6 receivers were combined to produce the BVP.

Each file is a 20x20xT tensor.

| Dimension | Meaning                                 |
|-----------|-----------------------------------------|
| 0         | Velocity along x-axis from [-2, +2] m/s |
| 1         | Velocity along y-axis from [-2, +2] m/s |
| 2         | Timestamp with 10 Hz sampling rate      |

We use Scipy to read the `.mat` files with `scipy.io.loadmat()`.
