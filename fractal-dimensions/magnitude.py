import numpy as np
from scipy.sparse.linalg import cg



# approximate magnitude at different scales using conjugate gradient
def magnitude_cg_warm_start(D, scales):
    N = len(D)
    S = len(scales)
    magnitudes = np.zeros(S)
    vec_one = np.ones(N)
    # initial guess of weight
    weight = np.zeros(N)
    for i in range(S):
        # associated similarity matrix
        Z = np.exp(-scales[i]*D)
        weight, *_ = cg(Z, vec_one, weight)
        magnitudes[i] = sum(weight)
    return magnitudes


# approximate magnitude dimension at different scales as slope of log-log graph
def magnitude_dimension(scales, magnitude):
    return np.diff(np.log(magnitude))/np.diff(np.log(scales))