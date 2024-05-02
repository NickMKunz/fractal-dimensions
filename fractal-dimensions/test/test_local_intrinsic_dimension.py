import numpy as np
import sys
sys.path.append("../fractal-dimensions")

from data.openml_data import openml_data
from data.toy_data import *
from local_intrinsic_dimension import local_intrinsic_dimension



# get data set from openml
data_set = openml_data(43007)

# create array of data points
data_points = data_set.to_numpy()

# define scales at which to estimate magnitude and magnitude dimension
scales = np.logspace(-3, 3, 100)

# estimate intrinsic dimension with magnitude
dim = local_intrinsic_dimension(data_points, data_points[0], 1000, scales)

print(f"Local Intrinsic Dimension Estimate: {dim:.03f}")

"""
# get data points from toy data set
data_points = toy_data_01()

# define scales at which to estimate magnitude and magnitude dimension
scales = np.logspace(-3, 3, 100)

# estimate intrinsic dimension with magnitude
dim = intrinsic_dimension(data_points, scales)

print(f"Intrinsic Dimension Estimate: {dim:.03f}")

# get data points from toy data set
data_points = toy_data_02(2, 3, 1000)

# define scales at which to estimate magnitude and magnitude dimension
scales = np.logspace(-3, 3, 100)

# estimate intrinsic dimension with magnitude
dim = intrinsic_dimension(data_points, scales)

print(f"Intrinsic Dimension Estimate: {dim:.03f}")
"""