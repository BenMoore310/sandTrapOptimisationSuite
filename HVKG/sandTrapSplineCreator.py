import numpy as np
import matplotlib.pyplot as plt
import random


def catmull(P):
    """
    A method to generate subdivision curves with given control points.

    Args:
        P (numpy array): control points.

    Returns:
        Q (numpy array): generated points on the subdivision curve.
    """
    N = P.shape[0]
    # print('N=', N)
    Q = np.zeros((2 * N - 1, 2), "d")
    # print('Q shape =', Q.shape)
    Q[0, :] = P[0, :]
    for i in range(0, N - 1):
        if i > 0:
            Q[2 * i, :] = (P[i - 1, :] + 6 * P[i, :] + P[i + 1, :]) / 8
        Q[2 * i + 1, :] = (P[i, :] + P[i + 1, :]) / 2
    Q[-1, :] = P[-1, :]

    return Q


def decision_vector_2d(n_points, ymin, ymax, xmin, xmax):
    y0 = ymax
    yfinal = -0.54
    x0 = xmin
    xfinal = xmax

    y_bounds = [ymin, ymax]
    x_bounds = [xmin, xmax]
    vertices = np.zeros((n_points, 2))
    vertices[0] = [x0, random.uniform(y_bounds[0], y_bounds[1])]
    vertices[-1] = [xfinal, random.uniform(y_bounds[0], y_bounds[1])]
    for i in range(1, n_points - 1):
        random_x = random.uniform(x_bounds[0], x_bounds[1])
        random_y = random.uniform(y_bounds[0], y_bounds[1])
        vertices[i] = [random_x, random_y]
    # print('unsorted', vertices)
    # vertices = np.sort(vertices, axis=0)

    print(vertices.shape)

    xCoords = vertices[:, 0]
    yCoords = vertices[:, 1]
    sortedX = np.sort(xCoords)

    # print(xCoords)
    # print(sortedX)

    # sortedVertices = [sortedX[i], yCoords[i]]

    sortedVertices = np.array((sortedX, yCoords)).T
    print(sortedVertices.shape)

    # sorted_indices = np.argsort(vertices[:,0])
    # vertices = vertices[sorted_indices]
    # vertices = np.flip(vertices, axis=0)
    # print('sorted', vertices)

    return sortedVertices


vertices = decision_vector_2d(5, -1.54, 1, 3.7, 23.5)
# vertices = decision_vector_2d(7, -0.636436, -0.252769, -0.383667, 0, -0.383667, -0.383667, 0, 0 )

print(vertices)


catmull_points = vertices
catmull_curve_validity = True
i = 0
while i < 4:
    catmull_points = catmull(catmull_points)
    i += 1
    print(i)


np.savetxt("spline.txt", catmull_points)
