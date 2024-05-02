import numpy as np
from sklearn import datasets



# classical magnitude example
def toy_data_01():
    return [[0, 0], [10, 0.01], [10, -0.01]]

# d-dimensional space embedded in R^n
def toy_data_affine(d, n, N=5000):
    points_first = np.random.uniform(-1, 1, (N, d))
    points_last = np.zeros((N, n - d))
    return np.hstack((points_first, points_last))

# d-dimensional sphere in R^n
def toy_data_sphere(d, n, N=5000):
    points = np.zeros((N, n))
    points[:, :d+1] = np.random.normal(size=(N, d+1))
    points[:, :d+1] /= np.linalg.norm(points[:, :d+1], axis=1)[:, np.newaxis]
    return points

# s curve
def toy_data_s_curve(N):
    return np.column_stack(datasets.make_s_curve(N))

# swiss roll
def toy_data_swiss_roll(N):
    return np.column_stack(datasets.make_swiss_roll(N))

# sierpinski triangle
def sierpinski_triangle(N):
    # vertices of the sierpinski triangle
    vertices = [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]]

    point = np.array([0, 0])

    points = np.zeros((N, 2))

    # sample ploints
    for i in range(N):
        vertex = vertices[np.random.randint(3)]
        point = (point + vertex) / 2
        points[i] = point

    return points

# annulus
def toy_annulus(N, r1, r2):
    angles = 2*np.pi*np.random.uniform(0, 2, N)
    ds = np.random.uniform(r1, r2, N)
    x = ds*np.cos(angles)
    y = ds*np.sin(angles)
    return np.column_stack([x, y])

# cantor set
def cantor_set(N, level=100):
    points = np.zeros(N)
    binary = np.random.randint(2, size=(N, level))  # Corrected random binary generation
    for i in range(level):
        points = points + 2 * binary[:, i] / (3 ** (i+1))  # Corrected exponentiation and division
    return points

# cantor dust
def cantor_dust(N, level=100):
    points = np.stack((cantor_set(N, level), cantor_set(N, level)), axis=-1)
    return points