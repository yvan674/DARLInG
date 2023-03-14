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

Our approach is:
1. Select those room ids and user ids, and torso locations as listed above.
2. Choose a 10% stratified random set, stratified based on room id, user id, and gesture
3. Choose 2 random repetitions for each sample

## Todo

- [ ] Make a single-file version of the data sources to make file access faster

## Work Log

- Went through the dataset documentation to figure out what the dataset looks like
- Used `csiread` to figure out what the actual data looks like
- Tried parsing the files naïvely, but this failed
- Went through the entire dataset again, analyzing each combination of room id, user id, etc. to figure out how to split the dataset
- Considered how to split the dataset in a way that makes sense semantically
- Considered how to split the dataset into a small version.
- 