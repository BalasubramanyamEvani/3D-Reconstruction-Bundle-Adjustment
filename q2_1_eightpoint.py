import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here


"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T. - Done
    (2) Setup the eight point algorithm's equation. - Done
    (3) Solve for the least square solution using SVD. - Done
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. - Done
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. - Done
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix - Done
"""


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # ----- TODO -----
    N = pts1.shape[0]
    pts1, pts2 = pts1 / M, pts2 / M
    A = np.zeros((N, 9))
    xm, ym = pts1[:, 0][:, np.newaxis], pts1[:, 1][:, np.newaxis]
    xmp, ymp = pts2[:, 0][:, np.newaxis], pts2[:, 1][:, np.newaxis]
    A = np.hstack(
        (xm * xmp, ym * xmp, xmp, xm * ymp, ym * ymp, ymp, xm, ym, np.ones((N, 1)))
    )
    u, s, vt = np.linalg.svd(A)
    F = vt[-1].T.reshape((3, 3))
    F = _singularize(F)
    F = refineF(F, pts1, pts2)
    T = np.diag((1 / M, 1 / M, 1))
    F = T.T @ F @ T
    F = F / F[2, 2]
    return F


if __name__ == "__main__":
    # Loading correspondences
    correspondence = np.load("data/some_corresp.npz")
    # Loading the intrinscis of the camera
    intrinsics = np.load("data/intrinsics.npz")
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    print(f"Recoverd F: {F}")

    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1

    np.savez("q2_1.npz", M=np.max([*im1.shape, *im2.shape]), F=F)
