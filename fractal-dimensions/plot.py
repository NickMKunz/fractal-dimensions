import matplotlib.pyplot as plt



# standard plot 
def plot(x, y, title=""):
    plt.plot(x, y)
    plt.title(title)
    plt.show()
    return 0

# log plot 
def log_plot(x, y, title=""):
    plt.plot(x, y)
    plt.xscale('log')
    plt.title(title)
    plt.show()
    return 0

# log-log plot 
def loglog_plot(x, y, title=""):
    plt.plot(x, y)
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.show()
    return 0

# plot data points in 2D
def plot_2d(data_points, title=""):
    plt.scatter(data_points[:, 0], data_points[:, 1], marker='.')
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.show()
    return 0


# plot data points in 3D
def plot_3d(data_points, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], marker='.')
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()
    return 0