# DARLInG

Domain Auto-labeling through Reinforcement Learning for the Inference of Gestures

This is the repository for Yvan Satyawan's Master Thesis.

## Widar3.0

These are mostly notes to myself to understand what the data looks like.

The files are stored in a _lot_ of files, split into multiple folders.
Folder naming scheme doesn't have much meaning, other than to split the dataset into when the data was captured.

## Gestures

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

#### Known Issues

The directory `20181130-VS` contains no `6-link` subdirectory.

## In vs Out of Domain

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


## Small Dataset

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
   2. Generate the small dataset using `src/data_utils/generate_small_splits.py`.
2. Calculate the mean and std-dev per channel using `src/data_utils/`??

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
- [ ] Explore Phase unwrap shapes
- [ ] Implement standard scalar.
- [ ] Implement REFINED for our dataset.
- [ ] Implement GAF for our dataset.
- [ ] Implement MTF for our dataset.
- [ ] Implement FGAF for our dataset.
- [ ] Implement FMTF for our dataset.
- [ ] Implement SignalPreprocessing module
- [ ] Implement BVP dummy SignalPreprocessing module
- [ ] Implement null agent
- [ ] Implement known-domain agent
- [ ] Implement experimentation framework for signal processing ablation study.
