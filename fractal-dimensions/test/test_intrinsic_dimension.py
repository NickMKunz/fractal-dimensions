import numpy as np
import sys
sys.path.append("../fractal-dimensions")

from data.openml_data import openml_data
from data.toy_data import *
from intrinsic_dimension import intrinsic_dimension



# get data set from openml
data_set = openml_data(43007)

# sample some points from the data set  
sample = data_set.sample(1000)

# create array of data points
data_points = sample.to_numpy()

# define scales at which to estimate magnitude and magnitude dimension
scales = np.logspace(-3, 3, 100)

# estimate intrinsic dimension with magnitude
dim = intrinsic_dimension(data_points, scales)

print(f"Intrinsic Dimension Estimate: {dim:.03f}")

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