import numpy as np
from sklearn.neighbors import NearestNeighbors

from data.toy_data import *
from intrinsic_dimension import intrinsic_dimension



# return k data points clossest to point
def local_intrinsic_dimension(data_points, point, k, scales):
    nbh = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nbh.fit(data_points)
    _, indices = nbh.kneighbors([point])
    # k data points clossest to point
    closest_data_points = np.array([data_points[i] for i in indices[0]])
    return intrinsic_dimension(closest_data_points, scales)