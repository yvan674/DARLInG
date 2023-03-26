"""IGTD Transform.

Taken from the IGTD repository on GitHub with _significant_ changes.

References:
    Zhu, Y., Brettin, T., Xia, F. et al. Converting tabular data into images
        for deep learning with convolutional neural networks. Sci Rep 11,
        11325 (2021). https://doi.org/10.1038/s41598-021-90923-y
    IGTD repository <https://github.com/zhuyitan/IGTD/>

Authors:
    Yitan Zhu <https://github.com/zhuyitan>
"""
import pickle
import time
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from tqdm import tqdm, trange

from preprocessing.cnn_loc_transform import pcc_calc
from data_utils.pickle_viewer import show_images


def min_max_transform(data: np.ndarray) -> np.ndarray:
    """Min-max transformation on data.

    This function does a linear transformation of each feature, so that the
    minimum and maximum values of a feature are 0 and 1, respectively.

    Args:
        data: an input data array with a size of [n_sample, n_feature]

    Returns:
        norm_data: the data array after transformation
    """
    norm_data = np.empty(data.shape)
    norm_data.fill(np.nan)
    for i in range(data.shape[1]):
        v = data[:, i].copy()
        if np.max(v) == np.min(v):
            norm_data[:, i] = 0
        else:
            v = (v - np.min(v)) / (np.max(v) - np.min(v))
            norm_data[:, i] = v
    return norm_data


def calculate_rankings(arr: np.ndarray, num: int) -> np.ndarray:
    """Calculates ranking of each element in the given array."""
    print("Generating tril indices")
    tril_id = np.tril_indices(num, k=-1)
    print("Doing heavy ranking computation step...")
    arg_sort = np.argsort(arr[tril_id])
    rank = np.empty_like(arg_sort)
    rank[arg_sort] = np.arange(len(arg_sort))
    # rank = rankdata(cord_dist[tril_id])

    # Applies the ranking to an image
    print("Generating ranking image")
    ranking = np.zeros((num, num))
    ranking[tril_id] = rank

    # Produces the symmetric image
    ranking = ranking + np.transpose(ranking)
    return ranking


def generate_feature_distance_ranking(
        data: np.ndarray,
        method: str = 'pearson') -> Tuple[np.ndarray, np.ndarray]:
    """Generates a ranking of based on a distance score.

    This function generates ranking of distances/dissimilarities between
    features for tabular data. Used for calculating pair-wise distances between
    features

    Args:
        data: input data, n_sample by n_feature
        method: 'pearson' uses Pearson correlation coefficient to evaluate
            similarity between features;
            'spearman' uses Spearman correlation coefficient to evaluate
            similarity between features;
            'euclidean' uses euclidean distance between features;
            'set' uses Jaccard index to evaluate
            similarity between features that are binary variables.
    Returns:
        The symmetric ranking matrix based on dissimilarity and the matrix of
            distances between features.
    """
    num = data.shape[1]
    if method == 'pearson':
        corr = np.corrcoef(np.transpose(data))
    elif method == 'spearman':
        corr = spearmanr(data).correlation
    elif method == 'euclidean':
        corr = squareform(pdist(np.transpose(data), metric='euclidean'))
        corr = np.max(corr) - corr
        corr = corr / np.max(corr)
    elif method == 'set':
        # This is the new set operation to calculate similarity.
        # It does not tolerate all-zero features.
        corr1 = np.dot(np.transpose(data), data)
        corr2 = data.shape[0] - np.dot(np.transpose(1 - data), 1 - data)
        corr = corr1 / corr2
    else:
        raise ValueError(f"Given method {method} is not one of the possible "
                         f"options. Possible options are `Pearson`, "
                         f"`spearman`, `euclidean`, or `set`")

    corr = 1 - corr
    corr = np.around(a=corr, decimals=10)

    ranking = calculate_rankings(corr, num)

    # tril_id = np.tril_indices(num, k=-1)
    # rank = rankdata(corr[tril_id])
    # ranking = np.zeros((num, num))
    # ranking[tril_id] = rank
    # ranking = ranking + np.transpose(ranking)

    return corr, ranking


def generate_matrix_distance_ranking(
        num_r: int,
        num_c: int,
        method='euclidean') -> Tuple[np.ndarray, np.ndarray]:
    """Generates a ranking based on distance in a matrix.

    This function calculates the ranking of distances between all pairs of
    entries in a matrix of size num_r by num_c. Used for calculating pair-wise
    distances between pixels.

    Args:
        num_r: number of rows in the matrix
        num_c: number of columns in the matrix
        method: method used to calculate distance. Can be 'euclidean' or
            'manhattan'.
    Returns:
        The coordinate matrix with size num_r * num_c by 2 matrix giving the
            coordinates of elements in the matrix and a num_r * num_c by
            num_r * num_c matrix giving the ranking of pair-wise distance.
    """
    # Here we generate tuples of each grid coordinate position as an array of
    # shape (num_r * num_c, 2) so we can do pairwise operations on each pair of
    # coordinates
    num = num_c * num_r

    # First step is to create the meshgrid
    coordinates = np.meshgrid(np.arange(num_r),
                              np.arange(num_c),
                              indexing="ij")

    # Then we apply a depth-stack to basically create a num_r x num_c x 2 tensor
    # Which contains coordinate information. This is then reshaped into a
    # num x 2 matrix containing all the coordinates.
    coordinates = np.dstack(coordinates).reshape((num_r * num_c, 2))

    # calculate the closeness of the elements
    cord_dist = np.zeros((num, num))
    if method == "euclidean":
        for i in trange(num):
            a_squared = np.square(coordinates[i, 0]
                                  * np.ones(num)
                                  - coordinates[:, 0])
            b_squared = np.square(coordinates[i, 1]
                                  * np.ones(num)
                                  - coordinates[:, 1])
            cord_dist[i, :] = np.sqrt(a_squared + b_squared)
    elif method == 'manhattan':
        for i in trange(num):
            a_dist = np.abs(coordinates[i, 0]
                            * np.ones(num)
                            - coordinates[:, 0])
            b_dist = np.abs(coordinates[i, 1]
                            * np.ones(num)
                            - coordinates[:, 1])
            cord_dist[i, :] = a_dist + b_dist

    # generate the ranking based on distance
    ranking = calculate_rankings(cord_dist, num)

    coordinates = coordinates.astype(np.int64)

    return coordinates, ranking


def igtd(source: np.ndarray,
         target: np.ndarray,
         op: callable = np.abs,
         max_step: int = 1000,
         switch_t: float = 0,
         val_step: int = 50,
         min_gain: float = 0.00001,
         random_state: int = 1) -> Tuple[np.ndarray, List[float], List[float]]:
    """Calculates error between source and target with a given penalty.

    This function switches the order of rows (columns) in the source ranking
    matrix to make it similar to the target ranking matrix. In each step, the
    algorithm randomly picks a row that has not been switched with others for
    the longest time and checks all possible switch of this row, and selects
    the switch that reduces the dissimilarity most. Dissimilarity (i.e. the
    error) is the summation of absolute difference of lower triangular elements
    between the rearranged source ranking matrix and the target ranking matrix.

    Args:
        source: a symmetric ranking matrix with zero diagonal elements.
        target: a symmetric ranking matrix with zero diagonal elements.
            'source' and 'target' should have the same size.
        op: The operand to use. Should be either `np.square` or `np.abs`
        max_step: the maximum steps that the algorithm should run if never
            converges.
        switch_t: the threshold to determine whether switch should happen.
        val_step: number of steps for checking gain on the objective function
            to determine convergence
        min_gain: if the objective function is not improved more than 'min_gain'
            in 'val_step' steps, the algorithm terminates.
        random_state: for setting random seed.


    Returns:
        The indices to rearrange the rows(columns) in source obtained during the
        optimization process;
        the errors obtained during the optimization process;
        the time taken by each step in the optimization process.
    """
    np.random.seed(random_state)

    source = source.copy()
    num = source.shape[0]
    tril_id = np.tril_indices(num, k=-1)
    index = np.array(range(num))
    index_record = np.empty((max_step + 1, num))
    index_record.fill(np.nan)
    index_record[0, :] = index.copy()

    # calculate the error associated with each row
    # First make a full array of np.nan
    err_v = np.full(num, np.nan)
    for i in range(num):
        err_v[i] = np.sum(op(source[i, 0:i] - target[i, 0:i])) + \
                   np.sum(op(source[(i + 1):, i] - target[(i + 1):, i]))

    step_record = -np.ones(num)
    err_record = [np.sum(op(source[tril_id] - target[tril_id]))]
    pre_err = err_record[0]
    t1 = time.time()
    run_time = [0]

    def error_row(src: np.ndarray, tgt: np.ndarray, x: int, y: int) -> float:
        """Calculates the error value for the row coordinate."""
        # Variable names correspond to components of the error
        err_a = np.sum(op(src[y, :x] - tgt[x, :x]))
        err_b = np.sum(op(src[(x + 1):y, y] - tgt[(x + 1):y, x]))
        err_ca = op(src[(y + 1):, y] - tgt[(y + 1):, x])
        err_cb = op(src[x, y] - tgt[y, x])
        err_c = np.sum(err_ca + err_cb)

        return err_a + err_b + err_c

    def error_col(src: np.ndarray, tgt: np.ndarray, x: int, y: int) -> float:
        """Calculates the error value for the column coordinate."""
        # Variable names correspond to components of the error
        err_a = np.sum(op(src[x, :x] - tgt[y, :x]))
        err_b = np.sum(op(src[x, (x + 1):y] - tgt[y, (x + 1):y]))
        err_ca = op(src[(y + 1):, x] - tgt[(y + 1):, y])
        err_cb = op(src[x, y] - tgt[y, x])
        err_c = np.sum(err_ca + err_cb)

        return err_a + err_b + err_c

    def switch_rows_i_and_j(src: np.ndarray, d: np.ndarray, pre_e: np.ndarray,
                            idx: np.ndarray, s_record: np.ndarray,
                            x: int, y: int, step: int) -> tuple:
        """Does what the function says it does.

        Returns:
            src, err, index, step_record
        """
        x_v = src[x, :].copy()
        y_v = src[y, :].copy()
        src[x, :] = y_v
        src[y, :] = x_v
        x_v = src[:, x].copy()
        y_v = src[:, y].copy()
        src[:, x] = y_v
        src[:, y] = x_v
        e = d[y] + pre_e

        # update rearrange index
        t = idx[x]
        idx[x] = idx[y]
        idx[y] = t

        # update step record
        s_record[x] = step
        s_record[y] = step

        return src, e, idx, s_record

    prog_bar = tqdm("step/s", "IGTD error: 0.000e+00", total=max_step)
    for s in range(max_step):
        delta = np.ones(num) * np.inf

        # randomly pick a row that has not been considered for the longest time
        idr = np.where(step_record == np.min(step_record))[0]
        ii = idr[np.random.permutation(len(idr))[0]]

        for jj in range(num):
            if jj == ii:
                continue

            if ii < jj:
                i = ii
                j = jj
            else:
                i = jj
                j = ii

            err_ori = err_v[i] + err_v[j] - op(source[j, i] - target[j, i])


            err_i = error_row(source, target, i, j)
            err_j = error_col(source, target, i, j)
            err_test = err_i + err_j - op(source[i, j] - target[j, i])

            delta[jj] = err_test - err_ori

        delta_norm = delta / pre_err
        ident = np.where(delta_norm <= switch_t)[0]
        if len(ident) > 0:
            jj = int(np.argmin(delta))

            # Update the error associated with each row
            if ii < jj:
                i = ii
                j = jj
            else:
                i = jj
                j = ii
            for k in range(num):
                if k < i:
                    err_v[k] -= op(source[i, k] - target[i, k])
                    err_v[k] -= op(source[j, k] - target[j, k])
                    err_v[k] += op(source[j, k] - target[i, k])
                    err_v[k] += op(source[i, k] - target[j, k])
                elif k == i:
                    err_v[k] = error_row(source, target, i, j)
                elif k < j:
                    err_v[k] -= op(source[k, i] - target[k, i])
                    err_v[k] -= op(source[j, k] - target[j, k])
                    err_v[k] += op(source[k, j] - target[k, i])
                    err_v[k] += op(source[i, k] - target[j, k])
                elif k == j:
                    err_v[k] = error_col(source, target, i, j)
                else:
                    err_v[k] -= op(source[k, i] - target[k, i])
                    err_v[k] -= op(source[k, j] - target[k, j])
                    err_v[k] += op(source[k, j] - target[k, i])
                    err_v[k] += op(source[k, i] - target[k, j])

            source, err, index, step_record = switch_rows_i_and_j(
                source, delta, pre_err, index, step_record, ii, jj, s
            )
        else:
            # error is not changed due to no switch
            err = pre_err

            # update step record
            step_record[ii] = s

        err_record.append(err)
        prog_bar.update(1)
        prog_bar.set_description(f"IGTD error: {err:.3e}")
        index_record[s + 1, :] = index.copy()
        run_time.append(time.time() - t1)

        if s > val_step:
            if np.sum((err_record[-val_step - 1] - np.array(
                    err_record[(-val_step):])) / err_record[
                          -val_step - 1] >= min_gain) == 0:
                prog_bar.set_description()
                break

        pre_err = err
    prog_bar.close()
    if s < max_step - 1:
        print(f"IGTD has converged at step {s}.")
    else:
        print("IGTD has not converged.")
    index_record = index_record[:len(err_record), :].astype(int)

    return index_record, err_record, run_time


def save_igtd_results(index_record: np.ndarray,
                      err_record: List[float],
                      run_time: List[float],
                      save_folder: Path,
                      file_name: str = ""):
    """Saves results of the IGTD run into a result and a performance CSV.

    In case we want to store and reuse results later as well as saving the
    running performance statistics.

    Args:
        index_record: The indices to rearrange the rows(columns) in source
            obtained during the optimization process.
        err_record: The errors obtained during the optimization process.
        run_time: The time taken by each step in the optimization process.
        save_folder: a path to save the picture of source ranking matrix in
            the optimization process.
        file_name: a string as part of the file names for saving results
    """
    # Save the index records to csv
    pd.DataFrame(index_record).to_csv(
        save_folder / f"{file_name}_index.csv",
        header=False,
        index=False,
    )

    # Transform the error records into a table, including the step of each
    # error and the timestamp and save to csv
    error_record_array = np.vstack(
        (np.arange(len(run_time)), run_time, err_record)
    ).T
    pd.DataFrame(error_record_array, columns=['step', 'run_time', 'error']) \
        .to_csv(save_folder / f"{file_name}_error_and_step.csv",
                header=True,
                index=False)


def generate_image_data(data: pd.DataFrame | np.ndarray,
                        index: np.ndarray,
                        num_row: int,
                        num_column: int,
                        coord: np.ndarray
                        ) -> Tuple[List[np.ndarray], List[str]]:
    """Does the image generation using indices calculated from the IGTD error.

    This function generates the data in image format according to rearrangement
    indices.

    Args:
         data: original tabular data, 2D array or data frame with shape
            [n_samples, n_features].
        index: indices of features obtained through optimization, according to
            which the features can be arranged into a num_r by num_c image.
        num_row: number of rows in image.
        num_column: number of columns in image.
        coord: coordinates of features in the image/matrix

    Returns:
        The generated data as a list of 2d images. The range of values is
        [0, 255]. Small values actually indicate high values in the original
        data; The list of indices of each sample.
    """
    if isinstance(data, pd.DataFrame):
        samples = data.index.map(np.str)
        data = data.values
    else:
        samples = [str(i) for i in range(data.shape[0])]

    data_2 = data.copy()
    data_2 = data_2[:, index]

    # Normalization and invert so that black means high value
    max_v = np.max(data_2)
    min_v = np.min(data_2)
    data_2 = 1 - (data_2 - min_v) / (max_v - min_v)

    # Transforming into uint8 range
    data_2 = (data_2 * 255).astype(np.uint8)

    image_data = []

    for i in trange(data_2.shape[0]):
        data_i = np.full((num_row, num_column), np.nan)
        data_i[coord] = data_2[i, :]
        image_data.append(data_i)

    return image_data, samples


def save_image_results(image_data: List[np.ndarray],
                       sample_idxs: List[str],
                       save_folder: Path,
                       file_name: str = "",
                       use_pickle: bool = True):
    """Saves all images as separate png files or as a single pickle file."""
    if use_pickle:
        if file_name == "":
            raise ValueError("Saving as pickle requires a filename.")

    imgs = [Image.fromarray(image_datum, mode="L")
            for image_datum in image_data]

    if use_pickle:
        with open(save_folder / f"{file_name}.pkl", "wb") as f:
            pickle.dump(imgs, f)
    else:
        for i, img in enumerate(imgs):
            img.save(save_folder / f"{sample_idxs[i]}.png")


def igtd_transform(data: dict,
                   output_dir: Path,
                   fea_dist_method: str = "euclidean",
                   image_dist_method: str = "euclidean",
                   max_step: int = 30000,
                   val_step: int = 500,
                   error: str = "abs",
                   switch_t: float = 0.,
                   min_gain: float = 1e-5):
    """Performs the transformation introduced in the IGTD paper.

    This function converts tabular data into images using the IGTD algorithm.

    This function does not return any variable, but saves multiple result files,
    which are the following:

    1.  Feature ranking plots for each data split.
    2.  Pixel-distance plots for each data split.
    3.  Rankings from the IGTD algorithm, including performance metrics for
        each split.
    4.  Resulting images from the IGTD algorithm in a pickle file for each

    Args:
        data: The data from the data ingest pipeline.
        output_dir: The directory to save result files.
        fea_dist_method: a string indicating the method used for calculating the
            pairwise distances between features. Possible options are
                'pearson' uses the Pearson correlation coefficient;
                'spearman' uses the Spearman correlation coefficient;
                'euclidean' uses euclidean distance;
                'set' uses the Jaccard index to evaluate the similarity between
                    features that are binary variables;
        image_dist_method: a string indicating the method used for calculating
            the distances between pixels in image. Possible options are
            'Euclidean' or 'Manhattan'.
        max_step: the maximum number of iterations that the IGTD algorithm will
            run if never converges.
        val_step: the number of iterations for determining algorithm
            convergence. If the error reduction rate is smaller than min_gain
            for val_step iterations, the algorithm converges.
        error: a string indicating the function to evaluate the difference
            between feature distance ranking and pixel distance ranking.
                'abs' indicates the absolute function.
                'squared' indicates the square function.
        switch_t: in each iteration, if the smallest error change rate resulted
            from all possible feature swapping is not larger than switch_t, the
            feature swapping that results in the smallest error change rate will
            be performed. Error change rate is the difference between the errors
            after and before feature swapping divided by the error before
            feature swapping. If switch_t <= 0, the IGTD algorithm monotonically
            reduces the error during optimization.
        min_gain: if the error reduction rate is not larger than min_gain for
            val_step iterations, the algorithm converges.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    arrs = data["trn_rss"], data["tst_rss"]

    if error == "abs":
        op = np.abs
    elif error == "square":
        op = np.square
    else:
        raise ValueError(f"Given error measure {error} is not one of the "
                         f"possible options.")

    print(f"Running IGTD algorithm with penalty type {error}")

    def save_plot(features: np.ndarray, fp: Path):
        figure = plt.figure(figsize=(6, 6))
        plt.imshow(np.max(features) - features, cmap='gray',
                   interpolation='nearest')
        plt.savefig(fname=fp,
                    bbox_inches='tight', pad_inches=0)
        plt.close(figure)

    #################################################################
    # Removing features with the lowest pearson correlation coefficient
    # To get a square number of features
    loc_df = data["trn_df"][["x", "y", "f"]]
    pcc_vec = np.abs(pcc_calc(arrs[0], loc_df))
    # Calculate how many features are necessary to keep
    num_features = int(np.sqrt(len(pcc_vec))) ** 2
    # Get the indices of the n features with the highest correlation
    # coefficient
    idxs = np.argsort(pcc_vec)[::-1][:num_features]
    # Remove the features with the lowest correlation coefficient
    arrs = [arr[:, idxs] for arr in arrs]

    for arr, split in zip(arrs, ("train", "test")):
        #################################################################
        # Generating feature distance ranking
        out_fp = (output_dir / f"{split}_feat_dist_ranking.pkl")
        if out_fp.exists():
            print(f"Found precalculated feature distance ranking for "
                  f"{split} set...")
            with open(out_fp, "rb") as f:
                corr, ranking_feature = pickle.load(f)
        else:
            print(f"Generating feature distance ranking for {split} set...")
            corr, ranking_feature = generate_feature_distance_ranking(
                data=arr,
                method=fea_dist_method
            )

            # Save original feature ranking plot
            save_plot(ranking_feature, output_dir /
                      f"{split}_feature_ranking.png")
            with open(out_fp, "wb") as f:
                pickle.dump((corr, ranking_feature), f)

        #################################################################
        # Generating pixel distance ranking
        out_fp = output_dir / f"{split}_matrix_dist_ranking.pkl"
        if out_fp.exists():
            print(f"Found precalculated pixel distance ranking for "
                  f"{split} set...")
            with open(out_fp, "rb") as f:
                coordinate, ranking_image = pickle.load(f)
        else:
            print(f"Generating pixel distance ranking for {split} set...")
            coordinate, ranking_image = generate_matrix_distance_ranking(
                num_r=np.sqrt(ranking_feature.shape[0]).astype(int),
                num_c=np.sqrt(ranking_feature.shape[0]).astype(int),
                method=image_dist_method)

            # Save image ranking plot
            save_plot(ranking_image, output_dir / f"{split}_image_ranking.png")
            with open(out_fp, "wb") as f:
                pickle.dump((coordinate, ranking_image), f)

        #################################################################
        # Doing IGTD
        print(f"Doing actual IGTD algorithm for {split} set...")
        out_fp = output_dir / f"{split}_index.csv"

        if not out_fp.exists():
            idx_record, err_record, run_time = igtd(
                source=ranking_feature,
                target=ranking_image,
                op=op,
                max_step=max_step,
                switch_t=switch_t,
                val_step=val_step,
                min_gain=min_gain,
                random_state=1)

            print(f"Saving results...")
            save_igtd_results(idx_record, err_record, run_time,
                              output_dir, split)

            # Plot error vs runtime and error vs iter
            fig = plt.figure(figsize=(6, 6))
            plt.plot(run_time, err_record)
            plt.savefig(fname=output_dir / f'{split}_error_and_runtime.png',
                        bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            fig = plt.figure()
            plt.plot(range(len(err_record)), err_record)
            plt.savefig(fname=output_dir / f'{split}_error_and_iteration.png',
                        bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Plot optimized feature rankings
            min_id = np.argmin(err_record)
            rank_feature_random = ranking_feature[idx_record[min_id, :], :]
            rank_feature_random = rank_feature_random[:, idx_record[min_id, :]]

            save_plot(rank_feature_random,
                      output_dir / f"{split}_optimized_feature_ranking.png")

            print("Generating images and saving...")
            data, samples = generate_image_data(
                data=arr,
                index=idx_record[min_id, :],
                num_row=ranking_feature.shape[0],
                num_column=ranking_feature.shape[0],
                coord=coordinate)

            save_image_results(data, samples, output_dir, split, True)


if __name__ == '__main__':
    from preprocessing.data_ingest import full_ingest_pipeline

    data_path = Path("../../data/UJI_LIB_DB_v2.2/01")
    out_path = Path("../../data/igtd_images")
    igtd_transform(full_ingest_pipeline(data_path), out_path)

    show_images(out_path / "train.pkl", 5)
