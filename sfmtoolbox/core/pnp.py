"""Perspective-n-Point: DLT + adaptive RANSAC (batched hypotheses) + robust refinement."""
import numpy as np
from scipy.optimize import least_squares
from sfmtoolbox.core.pose import rodrigues, rodrigues_inv


def _extract_batch(P):
    U, S, Vt = np.linalg.svd(P[:, :, :3])
    R = U @ Vt
    scale = S.mean(1)
    neg = np.linalg.det(R) < 0
    R[neg] *= -1
    scale[neg] *= -1
    return R, P[:, :, 3] / scale[:, None]


def pnp_dlt_batch(X, xn):
    """Linear PnP for B samples. X (B,N>=6,3), xn (B,N,2) normalized -> R (B,3,3), t (B,3)."""
    B, n = X.shape[:2]
    Xh = np.concatenate([X, np.ones((B, n, 1))], axis=2)
    A = np.zeros((B, 2 * n, 12))
    A[:, 0::2, 0:4] = Xh
    A[:, 0::2, 8:12] = -xn[:, :, 0:1] * Xh
    A[:, 1::2, 4:8] = Xh
    A[:, 1::2, 8:12] = -xn[:, :, 1:2] * Xh
    P = np.linalg.svd(A)[2][:, -1].reshape(B, 3, 4)
    R, t = _extract_batch(P)
    flip = np.median(np.einsum('bnj,bj->bn', X, R[:, 2]) + t[:, 2:3], axis=1) < 0
    if flip.any():
        R[flip], t[flip] = _extract_batch(-P[flip])
    return R, t


def pnp_dlt(X, xn):
    """Linear PnP. X (N>=6,3) world, xn (N,2) normalized; x ~ [R|t] X."""
    R, t = pnp_dlt_batch(X[None], xn[None])
    return R[0], t[0]


def refine_pose(R, t, X, xn, loss='huber', f_scale=1.0):
    r0 = np.concatenate([rodrigues_inv(R), t])

    def resid(p):
        Xc = X @ rodrigues(p[:3]).T + p[3:]
        return ((Xc[:, :2] / Xc[:, 2:3]) - xn).ravel()

    sol = least_squares(resid, r0, method='lm' if loss == 'linear' else 'trf',
                        loss=loss, f_scale=f_scale, max_nfev=200)
    return rodrigues(sol.x[:3]), sol.x[3:]


def _inliers(R, t, X, xn, thresh_n):
    Xc = X @ np.swapaxes(R, -1, -2) + t[..., None, :]
    ok = Xc[..., 2] > 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        err = np.linalg.norm(Xc[..., :2] / Xc[..., 2:3] - xn, axis=-1)
    return ok & (err < thresh_n)


def ransac_pnp(X, xn, thresh_n, max_iters=1000, confidence=0.999, rng=None, batch=64):
    """RANSAC PnP in normalized coords (thresh_n = px / focal). Same sample sequence and
    acceptance logic as a sequential loop. Returns (R, t, inlier_mask) or None."""
    rng = rng or np.random.default_rng(0)
    n = len(X)
    if n < 6:
        return None
    best_cnt, best = 0, None
    it, needed = 0, max_iters
    while it < min(needed, max_iters):
        b = min(batch, min(needed, max_iters) - it)
        idx = np.stack([rng.choice(n, 6, replace=False) for _ in range(b)])
        try:
            Rs, ts = pnp_dlt_batch(X[idx], xn[idx])
            ok = np.ones(b, bool)
        except np.linalg.LinAlgError:
            Rs, ts, ok = np.zeros((b, 3, 3)), np.zeros((b, 3)), np.zeros(b, bool)
            for k in range(b):
                try:
                    Rs[k], ts[k] = pnp_dlt(X[idx[k]], xn[idx[k]])
                    ok[k] = True
                except np.linalg.LinAlgError:
                    pass
        cnt = _inliers(Rs, ts, X, xn, thresh_n).sum(1)
        for k in range(b):
            if it >= min(needed, max_iters):
                break
            if ok[k] and cnt[k] > best_cnt:
                best_cnt, best = cnt[k], (Rs[k], ts[k])
                w = max(cnt[k] / n, 1e-8)
                denom = min(np.log1p(-min(w ** 6, 1 - 1e-12)), -1e-12)
                needed = int(np.log(1 - confidence) / denom) + 1
            it += 1
    if best is None or best_cnt < 6:
        return None
    inl = _inliers(*best, X, xn, thresh_n)
    R, t = refine_pose(*best, X[inl], xn[inl], loss='huber', f_scale=thresh_n)
    return R, t, _inliers(R, t, X, xn, thresh_n)
