import numpy as np
from scipy.spatial import distance_matrix

from magnitude import magnitude_cg_warm_start, magnitude_dimension


# estimate intrinsic dimension of data points using magnitude dimension
def intrinsic_dimension(data_points, scales):
    # distance matrix
    D = distance_matrix(data_points, data_points)
    # compute magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)
    # compute magnitude dimensions
    dimensions = magnitude_dimension(scales, magnitudes)
    return np.max(dimensions)