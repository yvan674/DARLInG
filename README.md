# DARLInG

![DARLInG Banner showing the model logo and the name](media/github-repo-banner.png)
Domain Auto-labeling through Reinforcement Learning for the Inference of Gestures

This is the repository for Yvan Satyawan's Master Thesis.

## Widar3.0

These are mostly notes to myself to understand what the data looks like.

The files are stored in a _lot_ of files, split into multiple folders.
Folder naming scheme doesn't have much meaning, other than to split the dataset into when the data was captured.

### Gestures

Not all gestures are performed by all users.
As such, we will only use gestures 1-6 in this work.

| User  | Gesture -> |     |     |     |     |     |     |     |     |    |       |
|-------|------------|-----|-----|-----|-----|-----|-----|-----|-----|----|-------|
|       | 1          | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10 | Total |
| 1     | 130        | 130 | 130 | 130 | 130 | 130 | 65  | 65  | 65  | 40 | 1015  |
| 2     | 200        | 175 | 175 | 175 | 150 | 125 | 25  | 25  | 25  | 25 | 1100  |
| 3     | 150        | 150 | 150 | 125 | 125 | 125 |     |     |     |    | 825   |
| 4     | 25         | 25  | 25  | 25  | 25  | 25  |     |     |     |    | 150   |
| 5     | 50         | 50  | 50  | 50  | 50  | 50  | 25  | 25  | 25  |    | 375   |
| 6     | 50         | 50  | 50  | 50  | 50  | 50  |     |     |     |    | 300   |
| 7     | 25         | 25  | 25  | 25  | 25  | 25  |     |     |     |    | 150   |
| 8     | 25         | 25  | 25  | 25  | 25  | 25  |     |     |     |    | 150   |
| 9     | 25         | 25  | 25  | 25  | 25  | 25  |     |     |     |    | 150   |
| 10    | 25         | 25  | 25  | 25  | 25  | 25  | 25  | 25  | 25  |    | 225   |
| 11    | 25         | 25  | 25  | 25  | 25  | 25  | 25  | 25  | 25  |    | 225   |
| 12    | 25         | 25  | 25  | 25  | 25  | 25  | 25  | 25  | 25  |    | 225   |
| 13    | 25         | 25  | 25  | 25  | 25  | 25  | 25  | 25  | 25  |    | 225   |
| 14    | 25         | 25  | 25  | 25  | 25  | 25  | 25  | 25  | 25  |    | 225   |
| 15    | 25         | 25  | 25  | 25  | 25  | 25  | 25  | 25  | 25  |    | 225   |
| 16    | 25         | 25  | 25  | 25  | 25  | 25  | 25  | 25  | 25  |    | 225   |
| 17    | 25         | 25  | 25  | 25  | 25  | 25  | 25  | 25  | 25  |    | 225   |
| Total | 880        | 855 | 855 | 830 | 805 | 780 | 315 | 315 | 315 | 65 | 6015  |

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

BVP lengths are not consistent. 
We pad them to the all have the same length of 28.

#### Known Issues

The directory `20181130-VS` contains no `6-link` subdirectory.

### In vs Out of Domain

As we are testing in vs out of domain performance, we will use the following 
data split for train, validation, and test.

| Set                 | Room IDs | User IDs   | Torso Location |
|---------------------|----------|------------|----------------|
| Training            | 1, 2     | 1, 2, 4, 5 | 1-5            |
| Validation          | 1        | 10-17      | 1-5            |
| Test Room           | 3        | 3, 7, 8, 9 | 1-5            |
| Test Torso Location | 1        | 1          | 6-8            |

We split it this way to make sure that the test set is truly unseen while the validation set is an unseen room-user combination instead of truly unseen.

We only use gestures 1-6, since these are the gestures which have samples from all participants.

For single domain, we use only User 2 in Room 1 with torso location 1 and face orientation 1.
This was chosen as it has the largest number of samples.
Test, validation, and training splits are randomly generated.


### Small Dataset

We create a small version of the  dataset containing 2 repetitions of each action and only 10% of all available data.

Torso locations in each room and in general are balance for torso locations 1-5.
Torso location 6-8 only exists in room 1 with user 1.

Face orientation is balanced as well.
All face orientations are done by every user in every room they participate in.

Gestures are balanced for each user, i.e., user 1 does every gesture the same number of times, but gestures are not balanced overall.
We only use gestures 1-6

Our approach is:
1. Select those room ids and user ids, and torso locations as listed above.
2. Choose a 10% stratified random set, stratified based on room id, user id, and gesture
3. Choose 2 random repetitions for each sample

## Experimental Steps

1. Generate the small dataset, if desired.
   1. Generate an index for the small dataset using `src/data_utils/generate_dataset_index.py`.
   2. Generate the smaller datasets using `src/data_utils/generate_smaller_splits.py`.
2. Otherwise, generate the dataset index using `src/data_utils/generate_dataset_index.py`.
3. Calculate the mean and standard deviation of amplitude and phase using `src/data_utils/calculate_mean_std.py`

## Config File

Some run configuration files are stored in the `run_configs` directory as YAML files.
These files have a top level key of at least one of the following: `[train_config, test_config]`.
The `train_runner` script uses the `train_config` dictionary in the config file.
The contents of the dictionary are passed directly as kwargs to the `run_training()` function.

## Todo

- [x] Go through the dataset documentation to figure out what the dataset looks like.
- [x] Use `csiread` to figure out what the actual data looks like.
- [x] Try parsing the files naïvely, but this failed.
- [x] Go through the entire dataset again, analyzing each combination of room id, user id, etc. to figure out how to split the dataset.
- [x] Consider how to split the dataset in a way that makes sense semantically.
- [x] Consider how to split the dataset into a small version.
- [x] Small version of the dataset which actually works.
- [x] Check the distribution of CSI array lengths.
- [x] Consider how to deal with uneven CSI array lengths.
- [x] Implement a working dataset class, which stacks antennas and receivers together.
- [x] Implement a signal processor base class.
- [x] Implement a lowpass filter.
- [x] Implement a phase unwrap filter.
- [x] Implement a phase median and uniform filter.
- [x] Implement a phase linear fit filter.
- [x] Implement a phase derivative filter.
- [x] Implement a pipeline class for all SignalProcessors
- [x] Implement DeepInsight for our dataset.
  - DeepInsight can be simply imported
- [x] Explore Phase unwrap shapes
- [x] Calculate mean and std of the dataset.
- [x] Implement standard scalar.
- [x] Implement Recurrent Plots for our dataset.
- [x] Implement GAF for our dataset.
- [x] Implement MTF for our dataset.
- [x] Implement SignalPreprocessing module
- [ ] Implement BVP dummy SignalPreprocessing module
- [x] Implement null agent
- [ ] Implement known-domain agent
- [ ] Implement experimentation framework for signal processing ablation study.
- [ ] Make sure lists of layers are actually stored as nn.ModuleList
