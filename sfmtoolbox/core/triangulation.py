"""Linear (DLT) triangulation."""
import numpy as np


def _dlt_solve(A):
    """Null vector of A as the least eigenvector of the 4x4 Gram A^T A (agrees with
    the SVD solution to ~1e-12 relative; ~2x faster batched)."""
    X = np.linalg.eigh(np.swapaxes(A, -1, -2) @ A)[1][..., :, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        return X[..., :3] / X[..., 3:]


def triangulate_point(Ps, xs):
    """Multi-view DLT. Ps: 3x4 projections, xs: (2,) coords in the frame of the Ps."""
    Ps = np.asarray(Ps)
    xs = np.asarray(xs)
    A = np.empty((2 * len(Ps), 4))
    A[0::2] = xs[:, :1] * Ps[:, 2] - Ps[:, 0]
    A[1::2] = xs[:, 1:] * Ps[:, 2] - Ps[:, 1]
    return _dlt_solve(A)


def triangulate_points(P1, P2, x1, x2):
    """Batched two-view DLT for N points. Returns (N,3)."""
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    A = np.empty((len(x1), 4, 4))
    A[:, 0] = x1[:, :1] * P1[2] - P1[0]
    A[:, 1] = x1[:, 1:] * P1[2] - P1[1]
    A[:, 2] = x2[:, :1] * P2[2] - P2[0]
    A[:, 3] = x2[:, 1:] * P2[2] - P2[1]
    return _dlt_solve(A) if len(x1) else np.zeros((0, 3))


def triangulate_multiview(P_obs, x_obs, seg, pos, n_pts, k_max):
    """Batched multi-view DLT. Observation o belongs to point seg[o] at slot pos[o];
    P_obs (M,3,4), x_obs (M,2). Zero-padded rows leave A^T A, hence the null vector,
    unchanged."""
    A = np.zeros((n_pts, 2 * k_max, 4))
    A[seg, 2 * pos] = x_obs[:, :1] * P_obs[:, 2] - P_obs[:, 0]
    A[seg, 2 * pos + 1] = x_obs[:, 1:] * P_obs[:, 2] - P_obs[:, 1]
    return _dlt_solve(A)


def triangulation_angles(X, C1, C2):
    """Angle (rad) between rays C1->X and C2->X."""
    r1 = X - C1
    r2 = X - C2
    r1 = r1 / (np.linalg.norm(r1, axis=1, keepdims=True) + 1e-12)
    r2 = r2 / (np.linalg.norm(r2, axis=1, keepdims=True) + 1e-12)
    return np.arccos(np.clip(np.einsum('ij,ij->i', r1, r2), -1, 1))


def project(X, R, t, K):
    """World points -> pixels. Returns (N,2) and depths (N,)."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    uv = Xc[:, :2] / z[:, None] @ K[:2, :2].T + K[:2, 2]
    return uv, z
