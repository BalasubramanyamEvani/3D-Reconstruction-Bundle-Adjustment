import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2
import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
"""


def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # TODO: Replace pass by your implementation
    pts1_homo = toHomogenous(pts1)
    pts2_homo = toHomogenous(pts2)
    best_inlier = None
    N = pts1.shape[0]
    best_count = -1

    for itr in range(nIters):
        print(f"--------------{itr}---------------")
        indices = np.random.choice(N, 7)
        Fs = sevenpoint(pts1[indices, :], pts2[indices, :], M)
        errs = []
        for F in Fs:
            err = calc_epi_error(pts1_homo, pts2_homo, F)
            errs.append(np.mean(err))

        min_idx = np.argmin(np.abs(np.array(errs)))
        curr_best_F = Fs[min_idx]

        err = calc_epi_error(pts1_homo, pts2_homo, curr_best_F)
        inliers = np.where(err < tol, True, False)
        curr_best_inliers_count = np.sum(inliers)

        if curr_best_inliers_count > best_count:
            best_count = curr_best_inliers_count
            best_inlier = inliers

    F = eightpoint(pts1[best_inlier, :], pts2[best_inlier, :], M)
    return F, best_inlier


"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""


def rodrigues(r):
    # TODO: Replace pass by your implementation
    theta = np.linalg.norm(r)
    u = r / theta
    u = u[:, np.newaxis]
    I = np.eye(3)

    def getuindex(i):
        return u.item(i)

    ux = np.array(
        [
            [0.0, -getuindex(2), getuindex(1)],
            [getuindex(2), 0.0, -getuindex(0)],
            [-getuindex(1), getuindex(0), 0.0],
        ]
    )

    return I * np.cos(theta) + (1 - np.cos(theta)) * (u @ u.T) + np.sin(theta) * ux


"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""


def invRodrigues(R):
    # TODO: Replace pass by your implementation
    A = (R - R.T) / 2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]])
    s = np.linalg.norm(rho)
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2

    def shalf(r):
        if np.linalg.norm(r) == np.pi and (
            (r[0] == 0 and r[1] == 0 and r[2] < 0)
            or (r[0] == 0 and r[1] < 0)
            or r[0] < 0
        ):
            return -r
        return r

    if s == 0 and c == 0:
        return np.zeros((3,))

    if s == 0 and c == -1:
        Rp = R + np.eye(3)
        for col in range(Rp.shape[1]):
            if np.any(Rp[:, col] != 0):
                v = Rp[:, v][:, np.newaxis]
                u = v / np.linalg.norm(v)
                u = u.flatten()
                r = shalf(u * np.pi)
                return r

    u = rho / s
    theta = np.arctan2(s, c)
    r = u * theta
    return r


"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    C1 = K1 @ M1
    num_points = p1.shape[0]
    P = x[: 3 * num_points].reshape((num_points, 3))
    P = np.hstack((P, np.ones((num_points, 1))))

    r2 = x[3 * num_points : 3 * num_points + 3]
    R2 = rodrigues(r2)

    t2 = x[3 * num_points + 3 :][:, np.newaxis]

    C1 = K1 @ M1

    M2 = np.hstack((R2, t2))
    C2 = K2 @ M2

    x1hat = (C1 @ P.T).T
    x1hat = (x1hat.T / x1hat[:, -1]).T[:, :-1]

    x2hat = (C2 @ P.T).T
    x2hat = (x2hat.T / x2hat[:, -1]).T[:, :-1]

    residuals1 = (p1 - x1hat).flatten()
    residuals2 = (p2 - x2hat).flatten()

    return np.concatenate((residuals1, residuals2))


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # ----- TODO -----
    R2_init = M2_init[:, :-1]
    r2_init = invRodrigues(R2_init)
    x = np.concatenate((P_init.flatten(), r2_init.flatten(), M2_init[:, -1].flatten()))

    func = lambda z: np.sum(rodriguesResidual(K1, M1, p1, K2, p2, z) ** 2)
    obj_start = func(x)
    res = scipy.optimize.minimize(func, x)
    x = res.x
    obj_end = func(x)

    num_points = p1.shape[0]
    P = x[: 3 * num_points].reshape((num_points, 3))
    P = np.hstack((P, np.ones((num_points, 1))))

    r2 = x[3 * num_points : 3 * num_points + 3]
    R2 = rodrigues(r2)

    t2 = x[3 * num_points + 3 :][:, np.newaxis]
    M2 = np.hstack((R2, t2))

    return M2, P, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F, inliers = ransacF(
        noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=100
    )
    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(
        noisy_pts2
    )

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization:
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """
    M2_init, _, P_init = findM2(F, pts1[inliers, :], pts2[inliers, :], intrinsics)
    print(f"M2_init: {M2_init}")
    print(f"P_init: {P_init}")

    M1 = np.eye(4)[:-1]
    M2, P, obj_start, obj_end = bundleAdjustment(
        K1, M1, pts1[inliers, :], K2, M2_init, pts2[inliers, :], P_init
    )

    print(f"Reprojection Error: before: {obj_start}, after: {obj_end}")
    print(f"Optimized M2: {M2}")
    print(f"Optimized P: {P}")

    np.savez(
        "q5.npz",
        M2_init=M2_init,
        P_init=P_init,
        M2_opt=M2,
        P_opt=P,
        obj_start=obj_start,
        obj_end=obj_end,
    )

    plot_3D_dual(P_init, P)
