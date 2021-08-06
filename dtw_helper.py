import numpy as np
import multiprocessing as mp

# Dependencies for the DTW matrix calculation
from fastdtw import fastdtw
from scipy.spatial.distance import squareform
from functools import partial
from haversine import haversine


def pad_trajectory(trajectory_list, max_length):
    """
    Use this to ensure all trajectories have the same length.
    Otherwise, get_dtw_matrix() will not be able to calculate on arrays of varying lengths
    :param trajectory_list: use this on each list of lat long in the pandas col
    :param max_length: must be an even integer
    :return:
    """
    original_length = max_length

    if max_length % 2 != 0:
            raise Exception('max_length should be even')

    if len(trajectory_list) % 2 != 0:
        max_length += 1

    padding_per_side = int(((max_length - len(trajectory_list))/2))

    t_lol = [list(ele) for ele in trajectory_list]
    padded_list = [t_lol[0]] * padding_per_side + t_lol + [t_lol[-1]] * padding_per_side
    padded_list = [tuple(ele) for ele in padded_list]

    if max_length > original_length:
        padded_list.pop(0) # remove first element of list to prevent final list exceeding max length

    return padded_list


def get_dtw_matrix(pandas_col, num_processes=mp.cpu_count()):
    """
    Brute force approach for computing DTW distance among all pairs
    in the dataset. This should be used on datasets with up to a few thousand trajectories.
    :param pandas_col: E.g. df["trajectory_lat_long"]
    :param num_processes: Defaults to max processes
    :return: NxN numpy matrix in the shape of (len(pandas_col), len(pandas_col))
    """
    data = np.array(pandas_col.tolist())

    N, _, _ = data.shape
    upper_triangle = [(i, j) for i in range(N) for j in range(i + 1, N)]

    with mp.Pool(processes=num_processes) as pool:
        result = pool.starmap(partial(fastdtw, dist=haversine), [(data[i], data[j]) for (i, j) in upper_triangle])

    dist_mat = squareform([item[0] for item in result])
    print("finished computing DTW matrix")
    print(dist_mat.shape)

    return dist_mat


def get_most_similar_path_idx(dist_mat, k=1):
    """
    Returns the column index of the most similar trajectory (i.e. trajectory with lowest non-zero
    DTW value for each row). This can be used as the index to extract most similar trajectory.
    TODO: expand this for k>1 most similar trajectories.

    :param dist_mat: NxN numpy DTW distance matrix
    :param k: k most similar paths (lowest DTW distance scores among all trajectory pairs)
    :return: column index representing the most similar trip
    """
    mx = np.ma.masked_array(dist_mat, mask=dist_mat == 0)
    col_idx = np.argmin(mx, 1)

    return col_idx
