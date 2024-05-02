import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tabulate import tabulate
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix, distance
import sys
sys.path.append("../fractal-dimensions")

from magnitude import magnitude_cg_warm_start, magnitude_dimension
from persistent_homology import plot_vietoris_rips, frac_dim, persistence, sum_lifetime, plot_barcode, asymptotic_slope
from data.openml_data import openml_data
from data.toy_data import toy_data_01, toy_data_swiss_roll, toy_data_affine, toy_data_sphere, sierpinski_triangle, toy_annulus, cantor_set, cantor_dust
from intrinsic_dimension import intrinsic_dimension
from plot import log_plot, loglog_plot, plot_3d



###--- MAGNITUDE ON CLASSICAL 3-POINT EXAMPLE ---###
def fun_01():
    # classicla 3-point example
    data_points = toy_data_01()

    scales = np.logspace(-3, 3, 100)

    # distance matrix
    D = distance_matrix(data_points, data_points)

    # magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)

    loglog_plot(scales, magnitudes, title="Magnitude Function of the classic 3-Point Space")
    return 0


###--- SWISS ROLL ---###
def fun_02():
    # points samples uniformly on swiss roll
    data_points = toy_data_swiss_roll(5_000)

    # scales at which to compute magnitude
    scales = np.logspace(-3, 2, 100)

    # distance matrix
    D = distance_matrix(data_points, data_points)

    # magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)

    # plot magnitude function
    loglog_plot(scales, magnitudes, title="Magnitude Function of the Swiss Roll (5000 points)")

    # magnitude dimension
    dimensions = magnitude_dimension(scales, magnitudes)

    # intrinsic dimension
    intrinsic_dim = np.max(dimensions)

    # plot magnitude dimentions
    x = (scales[:-1] + scales[1:])/2
    log_plot(x, dimensions, title="Magnitude Dimensions of the Swiss Roll (5000 points) (ID = "+str(round(intrinsic_dim, 3))+")")
    return 0


###--- AFFINE SUBSPACE ---###
def fun_03():
    # points samples uniformly d-plane in R^n
    d = 4
    n = 4
    N = 7000
    data_points = toy_data_affine(d, n, N)

    # scales at which to compute magnitude
    scales = np.logspace(-2, 3, 100)

    # distance matrix
    D = distance_matrix(data_points, data_points)

    # magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)

    # plot magnitude function
    loglog_plot(scales, magnitudes, title="Magnitude Function of "+str(d)+"-dim in R^"+str(n)+" ("+str(N)+" points)")

    # magnitude dimension
    dimensions = magnitude_dimension(scales, magnitudes)

    # intrinsic dimension
    intrinsic_dim = np.max(dimensions)

    # plot magnitude dimentions
    x = (scales[:-1] + scales[1:])/2
    log_plot(x, dimensions, title="Magnitude Dimensions of "+str(d)+"-dim in R^"+str(n)+" (ID = "+str(round(intrinsic_dim, 3))+") ("+str(N)+" points)")
    return 0


###--- Sphere ---###
def fun_04():
    # points samples uniformly on d-sphere in R^n
    d = 4
    n = 5
    N = 5000
    data_points = toy_data_sphere(d, n, N)

    # scales at which to compute magnitude
    scales = np.logspace(-2, 3, 100)

    # distance matrix
    D = distance_matrix(data_points, data_points)

    # magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)

    # plot magnitude function
    loglog_plot(scales, magnitudes, title="Magnitude Function of "+str(d)+"-sphere in R^"+str(n)+" ("+str(N)+" points)")

    # magnitude dimension
    dimensions = magnitude_dimension(scales, magnitudes)

    # intrinsic dimension
    intrinsic_dim = np.max(dimensions)

    # plot magnitude dimentions
    x = (scales[:-1] + scales[1:])/2
    log_plot(x, dimensions, title="Magnitude Dimensions of "+str(d)+"-sphere in R^"+str(n)+" (ID = "+str(round(intrinsic_dim, 3))+") ("+str(N)+" points)")
    return 0


###--- REAL DATA SATELLITE IMAGE (INTRINSIC DIMENSION = 2, EMBEDDED DIMENSION = 36) ---###
def fun_05():
    # 238095 points from openml (satellite_image: ID=294)
    data_sample = openml_data(294)
    data_sample = data_sample.to_numpy()
    data_points = data_sample[:1500]

    scales = np.logspace(-4, 0, 100)

    # distance matrix
    D = distance_matrix(data_points, data_points)

    # magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)
    loglog_plot(scales, magnitudes, title="Magnitude Function (Satellite Image)")

    # magnitude dimension
    dimensions = magnitude_dimension(scales, magnitudes)
    log_plot(scales[:-1], dimensions, title="Magnitude Dimensions (Satellite Image)")

    # intrinsic dimension
    intrinsic_dim = intrinsic_dimension(data_points, scales)
    print(f"Intrinsic Dimension Estimate (Satellite Image): {intrinsic_dim:.03f}")

    # local intrinsic dimension (1500 closest points)
    nbh = NearestNeighbors(n_neighbors=1500, algorithm='auto')
    nbh.fit(data_sample)
    _, indices = nbh.kneighbors([data_sample[0]])
    nbh_points = np.array([data_sample[i] for i in indices[0]])
    local_intrinsic_dim = intrinsic_dimension(nbh_points, scales)
    print(f"Local Intrinsic Dimension Estimate (Satellite Image): {local_intrinsic_dim:.03f}")
    return 0


###--- MORE REAL DATA ---###
def fun_06():
    scales = np.logspace(-3, 3, 100)

    # get (samples of) data sets from openml
    ids = [23383, 1479, 155, 60, 375, 43007]
    correct_dims = [1, 1, 3, 7, 6, 10]
    data_sets = [openml_data(id) for id in ids]
    data_sets = [data_set.sample(min(1500, len(data_set.index))).to_numpy() for data_set in data_sets]

    # intrinsic dimension
    for i in range(len(ids)):
        intrinsic_dim = intrinsic_dimension(data_sets[i], scales)
        print(f"Intrinsic Dimension Estimate (ID={ids[i]}): {intrinsic_dim:.03f}   vs Scikit-Dimension (ID={ids[i]}): {correct_dims[i]}")
    return 0


###--- TABLE ID ESTIMATES FOR DIFFERENT NUMBER OF DATA POINTS ---###
def fun_09(data_set, name="table"):
    # number of data points
    l = 5
    Ns = [int(N) for N in np.logspace(3.8, 4.3, l)]

    # array to hold ID estimates
    intrinsic_dims = np.zeros(l)

    # scales at which to compute magnitude
    scales = np.logspace(-2, 3, 100)

    for i in range(l):
        data_points = data_set[:Ns[i]]

        # distance matrix
        D = distance_matrix(data_points, data_points)

        # magnitude
        magnitudes = magnitude_cg_warm_start(D, scales)

        # magnitude dimension
        dimensions = magnitude_dimension(scales, magnitudes)

        # intrinsic dimension
        intrinsic_dims[i] = round(np.max(dimensions), 3)
    
    # create table
    table = tabulate([[Ns[i], intrinsic_dims[i]] for i in range(l)], headers=['Number of Data Points', 'ID Estimate'])

    # save table
    with open("experimental/results/"+name, "w") as file:
        file.write(table)
    
    # display table
    print(table)
    return 0


###--- ESTIMATE INTRINSIC DIMENSION WITH MAGNITUDE ---###
def fun_10(data_points):
    # scales at which to compute magnitude
    scales = np.logspace(-2, 3, 100)
    
    # distance matrix
    D = distance_matrix(data_points, data_points)

    # magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)

    # magnitude dimension
    dimensions = magnitude_dimension(scales, magnitudes)

    # intrinsic dimension
    print("Number of Data Points: "+str(len(data_points))+"    "+"ID: "+str(round(np.max(dimensions), 3)))
    return 0


###--- SAMPLE AND PLOT POINTS FROM SIERPINSKI TRIANGLE ---###
def fun_11():
    N = 10000
    points = sierpinski_triangle(N)
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=1, color='blue')
    plt.title(f'Sierpinski Triangle with {N} points')
    plt.axis('equal')
    plt.show()
    return 0


###--- PLOT VIETORIS-RIPS COMPLEX OF POINTS ON ANNULUS ---###
def fun_12():
    points = toy_annulus(20, r1=0.9, r2=1.1)
    thresholds = [0, 0.3, 0.6]
    plot_vietoris_rips(points, thresholds)
    return 0


###--- ESTIMATE FRACTAL DIMENSION OF SIERPINSKI TRIANGLE WITH PH ---###
def fun_13():
    sizes = np.logspace(3, 4, 20).astype(int)
    points = sierpinski_triangle(N=sizes[-1])

    d = frac_dim(points, sizes)

    print("Estimate of Fractal Dimension: "+str(d))
    return 0


###--- ANNULUS EXAMPLE ---###
def fun_14():
    # sample points from annulus
    np.random.seed(0)
    points = toy_annulus(20, r1=0.8, r2=1.2)

    # plot sampled points
    plt.scatter(points[:, 0], points[:, 1], color='black')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.savefig('example_annulus/annulus.png')
    plt.show()

    # plot vietoris-rips complex points
    thresholds = np.linspace(0, 1.75, 8)
    plot_vietoris_rips(points, thresholds, ppl=4)

    # compute lifetime intervals
    pers = persistence(points, diameter=2.5, max_dim=2, min_pers=0.001)

    pers0 = [interval[1] for interval in pers if interval[0] == 0]
    pers1 = [interval[1] for interval in pers if interval[0] == 1]

    print(pers0)
    print(pers1)

    plot_barcode(pers0)
    plot_barcode(pers1)

    """
    # plot barcode at dim 0
    gd.plot_persistence_barcode(pers0)
    plt.savefig('example_annulus/barcode0.png')
    plt.show()
    # plot barcode at dim 1
    gd.plot_persistence_barcode(pers1, legend=False)
    plt.savefig('example_annulus/barcode1.png')
    plt.show()
    """

    # plot persistence diagram at dim 0
    gd.plot_persistence_diagram(pers0)
    plt.gca().set_aspect('equal')
    plt.savefig('example_annulus/persistence_diagram0.png')
    plt.show()# plot persistence diagram at dim 1
    gd.plot_persistence_diagram(pers1, legend=False, greyblock=False)
    plt.gca().set_aspect('equal')
    plt.savefig('example_annulus/persistence_diagram1.png')
    plt.show()
    return 0


###--- CANTOR SET ---###
def fun_15():
    N = 10000
    points = cantor_set(N)
    plt.scatter(points, np.zeros(N), s=1, color='blue')
    plt.title(f'Cantor Set with {N} points')
    plt.show()
    return 0


###--- CANTOR DUST ---###
def fun_16():
    N = 30000
    points = cantor_dust(N)
    plt.scatter(points[:, 0], points[:, 1], s=1, color='blue')
    plt.title(f'Cantor Dust with {N} points')
    plt.show()
    return 0


###--- PLOT CONSTRUCTION OF CANTOR SET ---###
def fun_17():
    plt.plot([0, 1], [0, 0], color='blue')
    plt.plot([0, 1/3], [-1, -1], [2/3, 1], [-1, -1], color='blue')
    plt.plot([0, 1/9], [-2, -2], [2/9, 1/3], [-2, -2], [2/3, 7/9], [-2, -2], [8/9, 1], [-2, -2], color='blue')
    plt.show()
    return 0


###--- PH-DIMENSION OF CANTOR DUST ---###
def fun_18():
    sizes = np.logspace(3, 4, 5).astype(int)
    points = cantor_dust(N=sizes[-1])

    d = frac_dim(points, sizes)

    print("Estimate of Fractal Dimension: "+str(d))
    return 0


###--- PH-DIM OF SIERPINSKI TRIANGLE ---###
def fun_19():
    sizes = np.logspace(0, 4, 50).astype(int)
    points = sierpinski_triangle(N=sizes[-1])

    l = len(sizes)
    Ls = np.zeros(l)

    for i in range(l):
        pers = persistence(points[:sizes[i]], max_dim=0)
        Ls[i] = sum_lifetime(pers)

    plt.plot(sizes, Ls, 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    alpha = asymptotic_slope(np.log(sizes), np.log(Ls))
    d = 1/(1 - alpha)


    print("Slope: "+str(alpha))

    print("Estimate of Fractal Dimension: "+str(d))
    return 0


def fun_20():
    # 0.001, 0.1, 1
    scale = 1
    points = np.array([np.array([0, 0]), scale*np.array([20, 1]), scale*np.array([20, -1])])
    D = distance_matrix(points, points)
    Z = np.exp(-D)
    weight = np.linalg.solve(Z, np.ones(3))
    magnitude = sum(weight)
    print(magnitude)
    plt.scatter(points[:, 0], points[:, 1], marker='o', color='black', s=300)
    plt.axis('off')
    plt.xlim(points[:, 0].min() - 10, points[:, 0].max() + 10)
    plt.ylim(points[:, 1].min() - 10, points[:, 1].max() + 10)
    plt.show()
    return 0

def fun_21():
    length = 200_000
    width = 2    
    
    # compute magnitude
    def magnitude_rectangle(l, w):
        return (1+l/2)*(1+w/2)
    scales = np.logspace(-8, 4, 100)
    magnitudes = [magnitude_rectangle(t*length, t*width) for t in scales]

    # magnitude function
    plt.plot(scales, magnitudes, color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    #compute magnitude dimension
    middle_scales = np.sqrt(scales[:-1]*scales[1:])
    dimensions = magnitude_dimension(scales, magnitudes)

    # magnitude dimension
    plt.plot(middle_scales, dimensions, color='black')
    plt.xscale('log')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5).set_dashes((2, 2))
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5).set_dashes((2, 2))
    plt.axhline(y=2, color='black', linestyle='--', alpha=0.5).set_dashes((2, 2))
    plt.show()
    return 0


def fun_22():
    scales=np.logspace(-3, 5, 40)
    # sierpinski triangle
    n = 10_000
    points = sierpinski_triangle(n)

    # plot points
    plt.scatter(points[:, 0], points[:, 1], color='black', marker='.')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.show()

    # distance matrix
    D = distance_matrix(points, points)
    # compute magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)
    # compute magnitude dimensions
    dimensions = magnitude_dimension(scales, magnitudes)

    # magnitude function
    plt.plot(scales, magnitudes, color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    middle_scales = np.sqrt(scales[:-1]*scales[1:])

    # magnitude dimension
    plt.plot(middle_scales, dimensions, color='black')
    plt.xscale('log')
    plt.axhline(y=np.log(3)/np.log(2), color='black', linestyle='--', alpha=0.5, label="log(3)/log(2)").set_dashes((2, 2))
    plt.ylim(0, 2)
    plt.legend()
    plt.show()
    print(max(dimensions))
    return 0


def fun_23():
    scales=np.logspace(-3, 5, 30)
    n = 15_000
    points = np.random.normal(n)
    # distance matrix
    D = distance_matrix(points, points)
    # compute magnitude
    magnitudes = magnitude_cg_warm_start(D, scales)
    # compute magnitude dimensions
    dimensions = magnitude_dimension(scales, magnitudes)

    middle_scales = np.sqrt(scales[:-1]*scales[1:])

    # magnitude dimension
    plt.plot(middle_scales, dimensions, color='black')
    plt.xscale('log')
    plt.axhline(y=np.log(3)/np.log(2), color='black', linestyle='--', alpha=0.5, label="log(3)/log(2)").set_dashes((2, 2))
    plt.ylim(0, 2)
    plt.legend()
    plt.show()
    return 0


###--- SCALING LINE, SQUARE, CUBE ---###
def fun_24():
    # Create figure and subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # First row: Line segments
    axs[0, 0].plot([0, 1], [0, 0], 'b')  # Line segment of length 1
    axs[0, 0].set_title('Line Segment (Length 1)')
    axs[0, 1].plot([0, 2], [0, 0], 'r')  # Line segment of length 2
    axs[0, 1].set_title('Line Segment (Length 2)')

    # Second row: Squares
    axs[1, 0].add_patch(plt.Rectangle((0, 0), 1, 1, color='b', alpha=0.3))  # Square with side length 1
    axs[1, 0].set_aspect('equal', 'box')
    axs[1, 0].set_title('Square (Side Length 1)')
    axs[1, 1].add_patch(plt.Rectangle((0, 0), 2, 2, color='r', alpha=0.3))  # Square with side length 2
    axs[1, 1].set_aspect('equal', 'box')
    axs[1, 1].set_title('Square (Side Length 2)')

    # Third row: Cubes
    axs[2, 0] = fig.add_subplot(3, 2, 5, projection='3d')
    axs[2, 0].add_collection3d(plt.Poly3DCollection([[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]]))
    axs[2, 0].set_title('Cube (Side Length 1)')
    axs[2, 1] = fig.add_subplot(3, 2, 6, projection='3d')
    axs[2, 1].add_collection3d(plt.Poly3DCollection([[(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], [(0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)], [(0, 0, 0), (0, 2, 0), (0, 2, 2), (0, 0, 2)], [(2, 0, 0), (2, 2, 0), (2, 2, 2), (2, 0, 2)], [(0, 0, 0), (2, 0, 0), (2, 0, 2), (0, 0, 2)], [(0, 2, 0), (2, 2, 0), (2, 2, 2), (0, 2, 2)]]))
    axs[2, 1].set_title('Cube (Side Length 2)')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
    return 0


###--- ANNULUS EXAMPLE 2 ---###
def fun_25():
    # sample points from annulus
    np.random.seed(0)
    points = toy_annulus(20, r1=0.8, r2=1.2)

    """
    # plot sampled points
    plt.scatter(points[:, 0], points[:, 1], color='black')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.savefig('example_annulus/annulus.png')
    plt.show()

    # plot vietoris-rips complex points
    thresholds = np.linspace(0, 1.75, 8)
    plot_vietoris_rips(points, thresholds, ppl=4)
    """

    # compute lifetime intervals
    pers = persistence(points, diameter=2.5, max_dim=2, min_pers=0.001)

    pers0 = [interval[1] for interval in pers if interval[0] == 0]
    pers1 = [interval[1] for interval in pers if interval[0] == 1]

    def plot_barcode(intervals, limx, limy):
        x_max = max([x for _, x in intervals if x != np.inf])
        h = 0.5
        for [birth, death] in intervals:
            if death == np.inf:
                plt.plot([birth, x_max+10], [h, h], color='black')
            else:
                plt.plot([birth, death], [h, h], color='black')
            h += 0.5
        plt.xlim(limx)
        plt.ylim(limy)
        plt.box(False)
        plt.yticks([])
        plt.gca().axes.get_xaxis().set_ticklabels([])
        plt.xlabel("Scale Parameter")
        plt.tight_layout()
        plt.show()
        return 0

    plot_barcode(pers0, [-0.1, 10.1], [-0.1, 10.1])
    plot_barcode(pers1, [-0.1, 10.1], [-0.1, 10.1])

    # plot persistence diagram at dim 0
    gd.plot_persistence_diagram(pers0, legend=False, greyblock=False, colormap=[[0, 0, 0]])
    plt.gca().set_aspect('equal')
    plt.savefig('example_annulus/persistence_diagram0.png')
    plt.show()# plot persistence diagram at dim 1
    gd.plot_persistence_diagram(pers1, legend=False, greyblock=False, colormap=[[0, 0, 0]])
    plt.gca().set_aspect('equal')
    plt.savefig('example_annulus/persistence_diagram1.png')
    plt.show()
    return 0


def fun_26():
    dist = 0.1
    Nx = 10
    Ny = 2_500

    x_coords = np.linspace(0, dist*(Nx-1), Nx)
    y_coords = np.linspace(0, dist*(Ny-1), Ny)

    x, y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((x.ravel(), y.ravel()))
    
    scales = np.logspace(-3, 1, 30)
    D = distance.cdist(points, points, metric='cityblock')
    magnitudes = magnitude_cg_warm_start(D, scales)

    # magnitude function
    plt.plot(scales, magnitudes, color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    #compute magnitude dimension
    middle_scales = np.sqrt(scales[:-1]*scales[1:])
    dimensions = magnitude_dimension(scales, magnitudes)

    # magnitude dimension
    plt.plot(middle_scales, dimensions, color='black')
    plt.xscale('log')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5).set_dashes((2, 2))
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5).set_dashes((2, 2))
    plt.axhline(y=2, color='black', linestyle='--', alpha=0.5).set_dashes((2, 2))
    plt.show()
    return 0


def fun_27():
    np.random.seed(0)
    points = cantor_dust(10000)

    """
    plt.scatter(points[:, 0], points[:, 1], color='black', marker='.')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.show()
    # MAGNITUDE
    scales = np.logspace(-3, 5.5, 50)
    D = distance_matrix(points, points)
    magnitudes = magnitude_cg_warm_start(D, scales)

    # magnitude function
    plt.plot(scales, magnitudes, color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    #compute magnitude dimension
    middle_scales = np.sqrt(scales[:-1]*scales[1:])
    dimensions = magnitude_dimension(scales, magnitudes)

    # magnitude dimension
    plt.plot(middle_scales, dimensions, color='black')
    plt.xscale('log')
    plt.axhline(y=np.log(4)/np.log(3), color='black', linestyle='--', alpha=0.5, label="log(4)/log(3)").set_dashes((2, 2))
    plt.legend()
    plt.show()

    print("magnitude"+str(max(dimensions)))
    """
    
    # PH
    sizes = np.logspace(1, 4, 20).astype(int)

    l = len(sizes)
    Ls = np.zeros(l)

    for i in range(l):
        pers = persistence(points[:sizes[i]], max_dim=0)
        Ls[i] = sum_lifetime(pers)

    plt.plot(sizes, Ls, 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    alpha = asymptotic_slope(np.log(sizes), np.log(Ls))
    d = 1/(1 - alpha)

    print("Slope: "+str(alpha))

    print("Estimate of Fractal Dimension: "+str(d))
    return 0


def fun_28():
    def koch_segment(start, end):
        p1 = start + (end - start) / 3.0
        p3 = end - (end - start) / 3.0
        angle = np.pi / 3.0
        dx = p3[0] - p1[0]
        dy = p3[1] - p1[1]
        p2 = p1 + np.array([dx * np.cos(angle) - dy * np.sin(angle), 
                            dx * np.sin(angle) + dy * np.cos(angle)])
        return [start, p1, p2, p3, end]

    # Function to generate the Koch curve
    def generate_koch_curve(iterations):
        segment = np.array([[0.0, 0.0], [1.0, 0.0]])
        for _ in range(iterations):
            new_segments = []
            for i in range(len(segment) - 1):
                new_segments.extend(koch_segment(segment[i], segment[i + 1]))
            segment = np.array(new_segments)
        return segment

    # Generate the 5th iteration of the Koch curve
    koch_curve = generate_koch_curve(5)

    # Calculate the cumulative length of the curve
    segment_lengths = np.linalg.norm(koch_curve[1:] - koch_curve[:-1], axis=1)
    cumulative_lengths = np.cumsum(segment_lengths)
    total_length = cumulative_lengths[-1]

    # Sample points uniformly from the Koch curve
    num_points = 10000
    random_lengths = np.linspace(0, total_length, num_points + 1)[1:]
    sampled_points = []

    for length in random_lengths:
        idx = np.searchsorted(cumulative_lengths, length)
        if idx == 0:
            t = length / cumulative_lengths[0]
        else:
            t = (length - cumulative_lengths[idx - 1]) / (cumulative_lengths[idx] - cumulative_lengths[idx - 1])
        point = koch_curve[idx] + t * (koch_curve[idx + 1] - koch_curve[idx])
        sampled_points.append(point)

    points = np.array(sampled_points)

    
    plt.scatter(points[:, 0], points[:, 1], color='black', marker='.')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.show()
    """
    # MAGNITUDE
    scales = np.logspace(-3, 5.5, 50)
    D = distance_matrix(points, points)
    magnitudes = magnitude_cg_warm_start(D, scales)

    # magnitude function
    plt.plot(scales, magnitudes, color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    #compute magnitude dimension
    middle_scales = np.sqrt(scales[:-1]*scales[1:])
    dimensions = magnitude_dimension(scales, magnitudes)

    # magnitude dimension
    plt.plot(middle_scales, dimensions, color='black')
    plt.xscale('log')
    plt.axhline(y=np.log(4)/np.log(3), color='black', linestyle='--', alpha=0.5, label="log(4)/log(3)").set_dashes((2, 2))
    plt.legend()
    plt.show()

    print("magnitude"+str(max(dimensions)))
    """
    
    np.random.shuffle(points)
    # PH
    sizes = np.logspace(1, 4, 20).astype(int)
    l = len(sizes)
    Ls = np.zeros(l)

    for i in range(l):
        pers = persistence(points[:sizes[i]], max_dim=0, min_pers=0.00001)
        Ls[i] = sum_lifetime(pers)

    plt.plot(sizes, Ls, 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    alpha = asymptotic_slope(np.log(sizes), np.log(Ls))
    d = 1/(1 - alpha)

    print("Slope: "+str(alpha))

    print("Estimate of Fractal Dimension: "+str(d))



    return 0


def spread(t, D):
        Z = np.exp(-t*D)
        return np.sum(1 / np.sum(Z, axis=1))
    
    
def spread_dim(t, D):
    Z = np.exp(-t*D)
    return t/spread(t, D) * np.sum((np.sum(D*Z, axis=1))/(np.sum(Z, axis=1))**2)


def fun_29():
    scales=np.logspace(-3, 5, 40)

    # sierpinski triangle
    n = 10_000
    points = sierpinski_triangle(n)

    # distance matrix
    D = distance_matrix(points, points)

    spreads = [spread(t, D) for t in scales]

    plt.plot(scales, spreads, color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_aspect('equal')
    plt.show()

    dimensions = [spread_dim(t, D) for t in scales]

    # magnitude dimension
    plt.plot(scales, dimensions, color='black')
    plt.xscale('log')
    plt.axhline(y=np.log(3)/np.log(2), color='black', linestyle='--', alpha=0.5, label="log(3)/log(2)").set_dashes((2, 2))
    plt.ylim(0, 2)
    plt.legend()
    plt.show()
    print(max(dimensions))
    return 0


fun_29()

def fun_30():
    l = 1
    s = l/20
    t = 3
    x_dif = l/3
    y_dif = l/5

    lim_y = [-1.2, 1.5]

    # Line 1

    plt.plot([-l/2, l/2], [0, 0], 'black', linewidth=t)
    plt.plot([-l/2, -l/2], [-s, s], 'black', linewidth=t)
    plt.plot([l/2, l/2], [-s, s], 'black', linewidth=t)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim(lim_y)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Line 2

    plt.plot([-l, l], [0, 0], 'black', linewidth=t)
    plt.plot([-l, -l], [-s, s], 'black', linewidth=t)
    plt.plot([l, l], [-s, s], 'black', linewidth=t)
    plt.plot([0, 0], [-s, s], 'black', linewidth=t)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim(lim_y)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


    # Square 1

    plt.plot([-l/2, l/2], [-l/2, -l/2], 'black', linewidth=t)
    plt.plot([-l/2, l/2], [l/2, l/2], 'black', linewidth=t)
    plt.plot([-l/2, -l/2], [-l/2, l/2], 'black', linewidth=t)
    plt.plot([l/2, l/2], [-l/2, l/2], 'black', linewidth=t)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim(lim_y)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Square 2

    plt.plot([-l, l], [-l, -l], 'black', linewidth=t)
    plt.plot([-l, l], [l, l], 'black', linewidth=t)
    plt.plot([-l, -l], [-l, l], 'black', linewidth=t)
    plt.plot([l, l], [-l, l], 'black', linewidth=t)
    plt.plot([-l, l], [0, 0], 'black', linewidth=t)
    plt.plot([0, 0], [-l, l], 'black', linewidth=t)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim(lim_y)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Cube 1

    plt.plot([-l/2, l/2], [-l/2, -l/2], 'black', linewidth=t)
    plt.plot([-l/2, l/2], [l/2, l/2], 'black', linewidth=t)
    plt.plot([-l/2, -l/2], [-l/2, l/2], 'black', linewidth=t)
    plt.plot([l/2, l/2], [-l/2, l/2], 'black', linewidth=t)


    plt.plot([-l/2, -l/2+x_dif], [l/2, l/2+y_dif], 'black', linewidth=t)
    plt.plot([l/2, l/2+x_dif], [l/2, l/2+y_dif], 'black', linewidth=t)
    plt.plot([l/2, l/2+x_dif], [-l/2, -l/2+y_dif], 'black', linewidth=t)

    plt.plot([-l/2+x_dif, l/2+x_dif], [l/2+y_dif, l/2+y_dif], 'black', linewidth=t)
    plt.plot([l/2+x_dif, l/2+x_dif], [l/2+y_dif, -l/2+y_dif], 'black', linewidth=t)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim(lim_y)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Cube 2

    plt.plot([-l, l], [-l, -l], 'black', linewidth=t)
    plt.plot([-l, l], [l, l], 'black', linewidth=t)
    plt.plot([-l, -l], [-l, l], 'black', linewidth=t)
    plt.plot([l, l], [-l, l], 'black', linewidth=t)
    plt.plot([-l, l], [0, 0], 'black', linewidth=t)
    plt.plot([0, 0], [-l, l], 'black', linewidth=t)


    plt.plot([-l, -l+2*x_dif], [l, l+2*y_dif], 'black', linewidth=t)
    plt.plot([0, 2*x_dif], [l, l+2*y_dif], 'black', linewidth=t)
    plt.plot([l, l+2*x_dif], [l, l+2*y_dif], 'black', linewidth=t)
    plt.plot([l, l+2*x_dif], [0, 2*y_dif], 'black', linewidth=t)
    plt.plot([l, l+2*x_dif], [-l, -l+2*y_dif], 'black', linewidth=t)

    plt.plot([-l+2*x_dif, l+2*x_dif], [l+2*y_dif, l+2*y_dif], 'black', linewidth=t)
    plt.plot([-l+x_dif, l+x_dif], [l+y_dif, l+y_dif], 'black', linewidth=t)
    plt.plot([l+2*x_dif, l+2*x_dif], [l+2*y_dif, -l+2*y_dif], 'black', linewidth=t)
    plt.plot([l+x_dif, l+x_dif], [l+y_dif, -l+y_dif], 'black', linewidth=t)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim(lim_y)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return 0


def fun_31():
    l = 1
    r = l/2*np.sqrt(2)*1.1
    t = 3
    t2 = 2

    lim_y2 = [-1.5, 1.5]
    # Square cover 1

    plt.plot([-l, l], [-l, -l], 'black', linewidth=t)
    plt.plot([-l, l], [l, l], 'black', linewidth=t)
    plt.plot([-l, -l], [-l, l], 'black', linewidth=t)
    plt.plot([l, l], [-l, l], 'black', linewidth=t)

    ax = plt.gca()

    circle1 = plt.Circle((l/2, l/2), r, color='black', fill=False, linewidth=t2)
    circle2 = plt.Circle((l/2, -l/2), r, color='black', fill=False, linewidth=t2)
    circle3 = plt.Circle((-l/2, l/2), r, color='black', fill=False, linewidth=t2)
    circle4 = plt.Circle((-l/2, -l/2), r, color='black', fill=False, linewidth=t2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)

    ax.set_aspect('equal')
    ax.set_ylim(lim_y2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Square cover 2

    plt.plot([-l, l], [-l, -l], 'black', linewidth=t)
    plt.plot([-l, l], [l, l], 'black', linewidth=t)
    plt.plot([-l, -l], [-l, l], 'black', linewidth=t)
    plt.plot([l, l], [-l, l], 'black', linewidth=t)

    ax = plt.gca()

    circle1 = plt.Circle((l/4+l/2, l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle2 = plt.Circle((l/4+l/2, -l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle3 = plt.Circle((-l/4+l/2, l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle4 = plt.Circle((-l/4+l/2, -l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle5 = plt.Circle((l/4+l/2, l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle6 = plt.Circle((l/4+l/2, -l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle7 = plt.Circle((-l/4+l/2, l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle8 = plt.Circle((-l/4+l/2, -l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle9 = plt.Circle((l/4-l/2, l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle10 = plt.Circle((l/4-l/2, -l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle11 = plt.Circle((-l/4-l/2, l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle12 = plt.Circle((-l/4-l/2, -l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle13 = plt.Circle((l/4-l/2, l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle14 = plt.Circle((l/4-l/2, -l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle15 = plt.Circle((-l/4-l/2, l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle16 = plt.Circle((-l/4-l/2, -l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    ax.add_patch(circle5)
    ax.add_patch(circle6)
    ax.add_patch(circle7)
    ax.add_patch(circle8)
    ax.add_patch(circle9)
    ax.add_patch(circle10)
    ax.add_patch(circle11)
    ax.add_patch(circle12)
    ax.add_patch(circle13)
    ax.add_patch(circle14)
    ax.add_patch(circle15)
    ax.add_patch(circle16)

    ax.set_aspect('equal')
    ax.set_ylim(lim_y2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Square pack 1
    r = l/2*0.9

    plt.plot([-l, l], [-l, -l], 'black', linewidth=t)
    plt.plot([-l, l], [l, l], 'black', linewidth=t)
    plt.plot([-l, -l], [-l, l], 'black', linewidth=t)
    plt.plot([l, l], [-l, l], 'black', linewidth=t)

    ax = plt.gca()

    circle1 = plt.Circle((l/2, l/2), r, color='black', fill=False, linewidth=t2)
    circle2 = plt.Circle((l/2, -l/2), r, color='black', fill=False, linewidth=t2)
    circle3 = plt.Circle((-l/2, l/2), r, color='black', fill=False, linewidth=t2)
    circle4 = plt.Circle((-l/2, -l/2), r, color='black', fill=False, linewidth=t2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)

    ax.set_aspect('equal')
    ax.set_ylim(lim_y2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Square cover 2

    plt.plot([-l, l], [-l, -l], 'black', linewidth=t)
    plt.plot([-l, l], [l, l], 'black', linewidth=t)
    plt.plot([-l, -l], [-l, l], 'black', linewidth=t)
    plt.plot([l, l], [-l, l], 'black', linewidth=t)

    ax = plt.gca()

    circle1 = plt.Circle((l/4+l/2, l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle2 = plt.Circle((l/4+l/2, -l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle3 = plt.Circle((-l/4+l/2, l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle4 = plt.Circle((-l/4+l/2, -l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle5 = plt.Circle((l/4+l/2, l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle6 = plt.Circle((l/4+l/2, -l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle7 = plt.Circle((-l/4+l/2, l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle8 = plt.Circle((-l/4+l/2, -l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle9 = plt.Circle((l/4-l/2, l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle10 = plt.Circle((l/4-l/2, -l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle11 = plt.Circle((-l/4-l/2, l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle12 = plt.Circle((-l/4-l/2, -l/4+l/2), r/2, color='black', fill=False, linewidth=t2)
    circle13 = plt.Circle((l/4-l/2, l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle14 = plt.Circle((l/4-l/2, -l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle15 = plt.Circle((-l/4-l/2, l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    circle16 = plt.Circle((-l/4-l/2, -l/4-l/2), r/2, color='black', fill=False, linewidth=t2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    ax.add_patch(circle5)
    ax.add_patch(circle6)
    ax.add_patch(circle7)
    ax.add_patch(circle8)
    ax.add_patch(circle9)
    ax.add_patch(circle10)
    ax.add_patch(circle11)
    ax.add_patch(circle12)
    ax.add_patch(circle13)
    ax.add_patch(circle14)
    ax.add_patch(circle15)
    ax.add_patch(circle16)

    ax.set_aspect('equal')
    ax.set_ylim(lim_y2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return 0


def fun_32():
    # Function to calculate Euclidean distance between two points
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Generate random points
    def generate_random_points(n):
        return np.random.rand(n, 2)

    # Create a graph with Euclidean distances as edge weights
    def create_graph(points):
        n = len(points)
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distance = euclidean_distance(points[i], points[j])
                G[i, j] = distance
                G[j, i] = distance
        return G

    # Prim's algorithm to find the minimum spanning tree
    def prim_mst(G):
        n = len(G)
        INF = float('inf')
        visited = [False] * n
        parent = [-1] * n
        key = [INF] * n
        key[0] = 0
        
        for _ in range(n):
            u = min((key[i], i) for i in range(n) if not visited[i])[1]
            visited[u] = True
            for v in range(n):
                if G[u, v] > 0 and not visited[v] and G[u, v] < key[v]:
                    parent[v] = u
                    key[v] = G[u, v]
        
        mst_edges = [(parent[i], i) for i in range(1, n)]
        return mst_edges

    # Plot the minimum spanning tree
    def plot_mst(points, mst_edges):
        plt.scatter(points[:,0], points[:,1], color='black', zorder=5, marker='o')
        for edge in mst_edges:
            plt.plot([points[edge[0],0], points[edge[1],0]], [points[edge[0],1], points[edge[1],1]], color='black', zorder=1, linewidth=2)
        plt.tight_layout()
        plt.axis('off')
        plt.show()

    # Number of vertices
    n = 10

    # Generate random points

    np.random.seed(0)
    points = generate_random_points(n)

    plt.scatter(points[:,0], points[:,1], color='black', zorder=5, marker='o')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return 0


def fun_33():
    def generate_gaussian_subset(mean, cov_matrix, num_points):
        subset = np.random.multivariate_normal(mean, cov_matrix, num_points)
        return subset

    def plot_subsets(x, y, pointx, pointy):
        plt.plot([pointx[0], pointy[0]], [pointx[1], pointy[1]], color='black', linewidth=2)
        plt.scatter(x[:,0], x[:,1], color='blue', label='X', marker='o')
        plt.scatter(y[:,0], y[:,1], color='red', label='Y', marker='o')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([0,1])
        plt.gca().set_ylim([0,1])
        plt.tight_layout()
        plt.axis('off')
        plt.show()
        plt.scatter(x[:,0], x[:,1], color='blue', label='X', marker='o')
        plt.scatter(y[:,0], y[:,1], color='red', label='Y', marker='o')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([0,1])
        plt.gca().set_ylim([0,1])
        plt.tight_layout()
        plt.axis('off')
        plt.show()
        return 1

    # Mean and covariance matrices for Gaussian distributions
    mean_x = [0.5, 0.5]  # Mean for subset X
    mean_y = [0.5, 0.5]  # Mean for subset Y
    cov_matrix = [[0.01, 0], [0, 0.01]]  # Covariance matrix (assumed isotropic)

    # Number of points in each subset
    n = 5
    m = 10

    # Generate subsets
    x = generate_gaussian_subset(mean_x, cov_matrix, n)
    y = generate_gaussian_subset(mean_y, cov_matrix, m)

    def dist_p_s(point, set):
        dist_min = np.inf
        pt_min = 0
        for p in set:
            d = np.linalg.norm(point-p)
            if d < dist_min:
                dist_min = d
                pt_min = p
        return pt_min, dist_min

    def h_points(x, y):
        dist_max_x = 0
        cand_ptx_x = 0
        cand_pty_x = 0
        for ptx in x:
            pty, dist_ptx_y = dist_p_s(ptx, y)
            if dist_ptx_y > dist_max_x:
                dist_max_x = dist_ptx_y
                cand_ptx_x = ptx
                cand_pty_x = pty
        
        dist_max_y = 0
        cand_ptx_y = 0
        cand_pty_y = 0
        for pty in y:
            ptx, dist_pty_x = dist_p_s(pty, x)
            if dist_pty_x > dist_max_y:
                dist_max_y = dist_pty_x
                cand_ptx_y = ptx
                cand_pty_y = pty

        if dist_max_x > dist_max_y:
            return cand_ptx_x, cand_pty_x
        
        return cand_ptx_y, cand_pty_y
        

    pointx, pointy = h_points(x, y)

    # Plot subsets
    plot_subsets(x, y, pointx, pointy)
    return 0