import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from itertools import combinations
import sys
sys.path.append("../fractal-dimensions")

from plot import loglog_plot



# compute persistence lifetime intervals
def persistence(points, diameter=2.5, max_dim=3, min_pers=0.001):
    # create vietoris-rips complex
    rips_complex = gd.RipsComplex(points=points, max_edge_length=diameter)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)

    # compute the lifetime intercals
    return simplex_tree.persistence(min_persistence=min_pers)

# sum of persistence lifetime intervals
def sum_lifetime(persistence):
    total_sum = 0
    for (_, (a_i, b_i)) in persistence:
        if b_i != np.inf:
            total_sum += (b_i - a_i)
    return total_sum

###--- TODO ---###
# estimate slope on loglog plot
def asymptotic_slope(coord_x, coord_y):
    # naive; just take slope of line for last 5 x values
    slope = slope_line_fit(coord_x[-5:], coord_y[-5:])
    return slope

# estimate fractal dimension using PH-dim
def frac_dim(points, sizes):
    l = len(sizes)
    Ls = np.zeros(l)

    for i in range(l):
        pers = persistence(points[:sizes[i]], max_dim=0)
        Ls[i] = sum_lifetime(pers)

    # estimate asymptotic slope
    alpha = asymptotic_slope(np.log(sizes), np.log(Ls))

    return 1/(1 - alpha)

# construct vietoris-rips complex
def vietoris_rips(points, threshold):
    n = len(points)
    edges = []
    triangles = []
    
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    
    # edges
    for i in range(n):
        for j in range(i+1, n):
            if distances[i, j] <= threshold:
                edges.append((i, j))
                
    # triangles
    for edge_pair in combinations(range(n), 3):
        edge_distances = [distances[i, j] for i, j in combinations(edge_pair, 2)]
        if all(distance <= threshold for distance in edge_distances):
            triangles.append(edge_pair)
    
    return edges, triangles


# plot vietoris-rips complex
def plot_vietoris_rips(points, thresholds, ppl=4):
    _, axes = plt.subplots(int(len(thresholds)/ppl), ppl)

    for i, threshold in enumerate(thresholds):
        edges, triangles = vietoris_rips(points, threshold)
        ax = axes[int(i/ppl), i%ppl]
        
        # plot vertices
        ax.scatter(points[:, 0], points[:, 1], color='black', marker='o')
        
        # plot edges
        for edge in edges:
            start = points[edge[0]]
            end = points[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], color='black')
        
        # plot triangles
        for triangle_points in triangles:
            verts = [points[i] for i in triangle_points]
            verts.append(points[triangle_points[0]])
            ax.fill(*zip(*verts), alpha=0.1, linewidth=1, color='blue', edgecolor='black')
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout(pad=2.0)
    plt.savefig('example_annulus/vr_complex.png')
    plt.show()
    return 0


# find slope of line fint
def slope_line_fit(coord_x, coord_y):
    slope, _ = np.polyfit(coord_x, coord_y, 1)
    return slope


# draw barcode
def plot_barcode(intervals):
    x_max = max([x for _, x in intervals if x != np.inf])
    plt.figure(figsize=(8, 6))
    h = 0
    for [birth, death] in intervals:
        if death == np.inf:
            plt.plot([birth, x_max+10], [h, h], color='black')
        else:
            plt.plot([birth, death], [h, h], color='black')
        h += 0.5
    plt.xlabel("Scale Parameter")
    plt.ylim(-0.1, h)
    plt.xlim(-0.1, x_max+1)
    plt.yticks([])
    plt.xticks([])
    plt.box(False)
    plt.show()
    return 0