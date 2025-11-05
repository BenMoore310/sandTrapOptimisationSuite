import numpy as np
import random
import argparse
import matplotlib.pyplot as plt


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


def decision_vector_2d(catmullPoints, ymin, ymax, xmin, xmax, xmaxLim):
    n_points = len(catmullPoints)

    y0 = ymax
    yfinal = 0
    x0 = xmin
    xfinal = xmax
    xfinalLim = xmaxLim

    y_bounds = [ymin, ymax]
    x_bounds = [xmin, xmaxLim]
    vertices = np.zeros((n_points + 2, 2))
    vertices[0] = [x0, y0]
    vertices[-1] = [xfinal, yfinal]
    for i in range(1, n_points + 1):
        vertices[i] = catmullPoints[i - 1]
        # print(vertices)

    # print('unsorted', vertices)
    # vertices = np.sort(vertices, axis=0)

    # print(vertices.shape)

    # xCoords = vertices[:,0]
    # yCoords = vertices[:,1]
    # sortedX = np.sort(xCoords)

    # print(xCoords)
    # print(sortedX)

    # sortedVertices = [sortedX[i], yCoords[i]]

    # sortedVertices = np.array((sortedX, yCoords)).T
    # print(sortedVertices.shape)

    # sorted_indices = np.argsort(vertices[:,0])
    # vertices = vertices[sorted_indices]
    # vertices = np.flip(vertices, axis=0)
    # print('sorted', vertices)

    # I AM NO LONGER RETURNING SORTED VERTICES - LETTING THE OPTIMISER SORT IT OUT FOR ITSELF
    # print('returned vertices:', vertices)
    return vertices


def main(catmullPoints, numBasis):
    sample = np.reshape(catmullPoints, (numBasis, 2))

    vertices = decision_vector_2d(sample, -1.54, 1, 3.7, 23.5, 21)
    # vertices = decision_vector_2d(7, -0.636436, -0.252769, -0.383667, 0, -0.383667, -0.383667, 0, 0 )

    # print('vertices', vertices)
    # print(vertices.shape)

    catmull_points = vertices
    catmull_curve_validity = True
    i = 0
    while i < numBasis:
        catmull_points = catmull(catmull_points)
        i += 1
        # print(i)
    np.savetxt("spline.txt", catmull_points)

    plt.figure(figsize=(10, 2))
    plt.plot(catmull_points[:, 0], catmull_points[:, 1])
    plt.scatter(vertices[:, 0], vertices[:, 1])
    plt.ylim(-1.7, 1.1)
    plt.xlim(3.5, 24)
    plt.title("Tank Bottom Profile")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.savefig("currentSpline.png")


# if __name__ == "main":
#     parser = argparse.ArgumentParser(
#         description="Run the sand trap shape optimisation."
#     )
#     parser.add_argument(
#         "--catmullPoints",
#         type=int,
#         default=4,
#         help="Number of Catmull-Clark control points used during spline generation process."
#     )

#     args = parser.parse_args()

#     main(args.catmullPoints)
