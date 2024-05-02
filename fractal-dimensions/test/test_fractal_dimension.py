import numpy as np
import sys
sys.path.append("../fractal-dimensions")

from data.picture_data import picture_data
from intrinsic_dimension import intrinsic_dimension



# get data points from picture
data_points = picture_data('fractal_01.png')

# define scales at which to estimate magnitude and magnitude dimension
scales = np.logspace(-3, 3, 100)

# estimate intrinsic dimension with magnitude
dim = intrinsic_dimension(data_points, scales)

print(f"Intrinsic Dimension Estimate: {dim:.03f}")