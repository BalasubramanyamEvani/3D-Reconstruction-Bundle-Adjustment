import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here


"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M. - Done
    (2) Setup the seven point algorithm's equation. - Done
    (3) Solve for the least square solution using SVD. - Done
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2) - Done
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots - Done
    (6) Unscale the fundamental matrixes and return as Farray
"""


def sevenpoint(pts1, pts2, M):
    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
    N = pts1.shape[0]
    pts1, pts2 = pts1 / M, pts2 / M
    A = np.zeros((N, 9))
    xm, ym = pts1[:, 0][:, np.newaxis], pts1[:, 1][:, np.newaxis]
    xmp, ymp = pts2[:, 0][:, np.newaxis], pts2[:, 1][:, np.newaxis]
    A = np.hstack(
        (xm * xmp, ym * xmp, xmp, xm * ymp, ym * ymp, ymp, xm, ym, np.ones((N, 1)))
    )
    u, s, vt = np.linalg.svd(A)
    f1, f2 = vt[-1].T.reshape((3, 3)), vt[-2].T.reshape((3, 3))
    f1 = f1 / f1[2, 2]
    f2 = f2 / f2[2, 2]
    alphas = np.linspace(0.5, 1, num=4)
    Bs = np.array([np.linalg.det(alpha * f1 + (1 - alpha) * f2) for alpha in alphas])
    avs = [[alpha**3, alpha**2, alpha, 1] for alpha in alphas]
    avs = np.vstack(avs)
    ks = np.linalg.solve(avs, Bs).T
    roots = np.polynomial.polynomial.polyroots(ks)
    T = np.diag((1 / M, 1 / M, 1))
    for root in roots:
        root = root.real
        F = root * f1 + (1 - root) * f2
        # assert np.isclose(np.linalg.det(F), 0.)
        F = _singularize(F)
        F = refineF(F, pts1, pts2)
        F = T.T @ F @ T
        F = F / F[2, 2]
        Farray.append(F)
    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]

    np.savez("q2_2.npz", F=F, M=M)
    print(f"Recovered F: {F}")

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
