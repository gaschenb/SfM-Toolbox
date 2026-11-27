"""Two-view epipolar geometry. Convention: x2^T F x1 = 0 (homogeneous pixels)."""
import numpy as np


def to_h(x):
    return np.hstack([x, np.ones((len(x), 1))])


def from_h(x):
    return x[:, :-1] / x[:, -1:]


def normalize_points(pts):
    """Hartley normalization: zero centroid, mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1.0]])
    return (T @ to_h(pts).T).T, T


def _hartley_batch(x):
    c = x.mean(axis=1)
    d = np.sqrt(((x - c[:, None]) ** 2).sum(-1)).mean(-1)
    s = np.sqrt(2) / np.maximum(d, 1e-12)
    T = np.zeros((len(x), 3, 3))
    T[:, 0, 0] = T[:, 1, 1] = s
    T[:, :2, 2] = -s[:, None] * c
    T[:, 2, 2] = 1.0
    return (x - c[:, None]) * s[:, None, None], T


def eight_point_batch(x1, x2):
    """Normalized 8-point on B samples at once. x1, x2: (B,N>=8,2) -> (B,3,3)."""
    n1, T1 = _hartley_batch(x1)
    n2, T2 = _hartley_batch(x2)
    u1, v1 = n1[..., 0], n1[..., 1]
    u2, v2 = n2[..., 0], n2[..., 1]
    A = np.stack([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2,
                  u1, v1, np.ones_like(u1)], axis=-1)
    F = np.linalg.svd(A)[2][:, -1].reshape(-1, 3, 3)
    U, S, Vt = np.linalg.svd(F)
    F = (U * np.concatenate([S[:, :2], np.zeros((len(S), 1))], 1)[:, None]) @ Vt
    F = T2.transpose(0, 2, 1) @ F @ T1
    with np.errstate(divide='ignore', invalid='ignore'):
        return F / F[:, 2:3, 2:3]


def eight_point(x1, x2):
    """Normalized 8-point algorithm. x1, x2: (N>=8, 2) pixel coords."""
    return eight_point_batch(x1[None], x2[None])[0]


def sampson_distance(F, x1, x2):
    """Squared first-order geometric error (px^2). F may be (3,3) or (B,3,3)."""
    x1h, x2h = to_h(x1), to_h(x2)
    Fx1 = x1h @ np.swapaxes(F, -1, -2)
    Ftx2 = x2h @ F
    num = np.einsum('...ij,ij->...i', Fx1, x2h) ** 2
    den = (Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2 + Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2)
    return num / np.maximum(den, 1e-12)


def _adaptive_iters(w, s, confidence):
    w = max(w, 1e-8)
    denom = min(np.log1p(-min(w ** s, 1 - 1e-12)), -1e-12)
    return int(np.log(1 - confidence) / denom) + 1


def ransac_fundamental(x1, x2, thresh=1.0, max_iters=2000, confidence=0.999, rng=None,
                       batch=64):
    """Adaptive RANSAC over the 8-point algorithm, hypotheses scored in batches.
    Sample sequence and acceptance logic match the sequential version exactly.
    Returns (F, inlier_mask)."""
    rng = rng or np.random.default_rng(0)
    n = len(x1)
    if n < 8:
        raise ValueError("need >= 8 correspondences")
    best_inl = np.zeros(n, bool)
    best_cnt, best_F = 0, None
    t2 = thresh ** 2
    it, needed = 0, max_iters
    while it < min(needed, max_iters):
        b = min(batch, min(needed, max_iters) - it)
        idx = np.stack([rng.choice(n, 8, replace=False) for _ in range(b)])
        try:
            Fs = eight_point_batch(x1[idx], x2[idx])
            ok = np.ones(b, bool)
        except np.linalg.LinAlgError:
            Fs, ok = np.zeros((b, 3, 3)), np.zeros(b, bool)
            for k in range(b):
                try:
                    Fs[k] = eight_point(x1[idx[k]], x2[idx[k]])
                    ok[k] = True
                except np.linalg.LinAlgError:
                    pass
        with np.errstate(invalid='ignore'):
            inl = sampson_distance(Fs, x1, x2) < t2
        cnt = inl.sum(1)
        for k in range(b):
            if it >= min(needed, max_iters):
                break
            if ok[k] and cnt[k] > best_cnt:
                best_inl, best_cnt, best_F = inl[k], cnt[k], Fs[k]
                needed = _adaptive_iters(cnt[k] / n, 8, confidence)
            it += 1
    if best_F is None:
        raise RuntimeError("RANSAC failed")
    if best_cnt >= 8:
        F = eight_point(x1[best_inl], x2[best_inl])
        with np.errstate(invalid='ignore'):
            inl = sampson_distance(F, x1, x2) < t2
        if inl.sum() >= best_cnt:
            best_F, best_inl = F, inl
    return best_F, best_inl


def essential_from_fundamental(F, K1, K2=None):
    K2 = K1 if K2 is None else K2
    U, S, Vt = np.linalg.svd(K2.T @ F @ K1)
    s = (S[0] + S[1]) / 2
    return (U * [s, s, 0]) @ Vt
