"""Rotation parameterization and relative pose recovery from E."""
import numpy as np
from sfmtoolbox.core.triangulation import triangulate_points


def rodrigues(rvec):
    """Axis-angle -> rotation matrix."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rodrigues_batch(rvecs):
    """(N,3) axis-angle -> (N,3,3)."""
    rvecs = np.asarray(rvecs, float).reshape(-1, 3)
    theta = np.linalg.norm(rvecs, axis=1)
    small = theta < 1e-12
    k = rvecs / np.where(small, 1.0, theta)[:, None]
    K = np.zeros((len(k), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    R[small] = np.eye(3)
    return R


def rodrigues_inv(R):
    """Rotation matrix -> axis-angle."""
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(tr)
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(M), 0))
        i = np.argmax(k)
        k = M[i] / max(k[i], 1e-12)
        k /= np.linalg.norm(k)
        return theta * k
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2 * np.sin(theta)) * w


def decompose_essential(E):
    """The 4 candidate (R, t) pairs (t unit-norm)."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    R1, R2 = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def recover_pose(E, x1n, x2n):
    """Cheirality disambiguation on normalized coords. Returns (R, t, mask) with
    X_cam2 = R X_cam1 + t."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    for R, t in decompose_essential(E):
        X = triangulate_points(P1, np.hstack([R, t[:, None]]), x1n, x2n)
        good = (X[:, 2] > 0) & ((X @ R[2] + t[2]) > 0)
        if best is None or good.sum() > best[2].sum():
            best = (R, t, good)
    return best
