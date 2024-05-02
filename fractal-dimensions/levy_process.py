# simulate Levy stable process
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import random
import matplotlib.colors as colors

# Set the parameters of the Levy process
#alpha = 1  # Stability parameter
#beta = 0  # Skewness parameter
dim = 2 #128

# [100, 200, 500, 750, 1000, 1500]
# Set the number of steps in the process
n_steps = 1000

def sample_levy(alpha, seed, n_steps = 500, dim=3):
  # Set the parameters of the Levy process
  alpha = alpha  # Stability parameter
  beta = 0  # Skewness parameter
  seed = 0
  np.random.seed(seed)
  levy_process = np.zeros((n_steps, dim))
  # Generate the Levy process
  for d in range(dim):
      levy_process[:, d] = np.cumsum(st.levy_stable.rvs(alpha, beta, size=n_steps)*np.sqrt(1/n_steps))
  return levy_process

## sample a levy process with ID=alpha=1
levy_process = sample_levy(1, seed=18, n_steps=n_steps, dim=dim)

x= levy_process[:,0]
y= levy_process[:,1]
# create a colormap using the Vidris color map
cmap = plt.get_cmap('viridis')

# create a normalizer to scale the values to the range 0-1
norm = colors.Normalize(vmin=0, vmax=n_steps)

# create a line plot with a continuously changing color
fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
for i in range(len(x)-1):
    ax.plot3D([x[i], x[i+1]], [i, i+1], [y[i], y[i+1]], color=cmap(norm(i)), linewidth=2)
#ax.set_xlabel('Levy 1')
ax.set_ylabel('Iterations')
#ax.set_zlabel('Levy 2')

# add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm,fraction=0.02, pad=0.15, ax=ax).ax.set_title('Iterations', size=10)
ax.view_init(25, 310)
plt.show()