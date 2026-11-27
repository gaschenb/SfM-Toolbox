"""Lens distortion models on normalized coordinates, with analytic Jacobians.

Parameter vectors (full length per model; `free` selects which are refined):
  pinhole  -> []
  radial1  -> [k1]                  s = 1 + k1 r^2
  radial2  -> [k1, k2]
  radial3  -> [k1, k2, k3]
  brown    -> [k1, k2, p1, p2, k3]  Brown-Conrady / OpenCV:
              x_d = s x + 2 p1 x y + p2 (r^2 + 2 x^2)
              y_d = s y + p1 (r^2 + 2 y^2) + 2 p2 x y,   s = 1 + k1 r^2 + k2 r^4 + k3 r^6
  fisheye  -> [k1, k2, k3, k4]      equidistant (OpenCV fisheye):
              theta = atan r,  theta_d = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8),
              x_d = (theta_d / r) x
"""
import numpy as np

MODELS = {'pinhole': 0, 'radial1': 1, 'radial2': 2, 'radial3': 3, 'brown': 5, 'fisheye': 4}
PARAM_NAMES = {'pinhole': [], 'radial1': ['k1'], 'radial2': ['k1', 'k2'],
               'radial3': ['k1', 'k2', 'k3'], 'brown': ['k1', 'k2', 'p1', 'p2', 'k3'],
               'fisheye': ['k1', 'k2', 'k3', 'k4']}


def n_params(model):
    return MODELS[model]


def _brown_full(model, k):
    full = np.zeros(5)
    if model == 'brown':
        full[:] = k
    else:
        full[[0, 1, 4][:len(k)]] = k
    return full


def distort(xn, model, k, jac=False):
    """xn (N,2) undistorted normalized -> xd (N,2). With jac: also dxd/dxn (N,2,2) and
    dxd/dk (N,2,len(k))."""
    xn = np.asarray(xn, float)
    k = np.asarray(k, float)
    N = len(xn)
    if model == 'pinhole' or not len(k):
        if jac:
            return xn.copy(), np.broadcast_to(np.eye(2), (N, 2, 2)).copy(), np.zeros((N, 2, 0))
        return xn.copy()
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    if model == 'fisheye':
        r = np.sqrt(r2)
        th = np.arctan(r)
        t2 = th * th
        pw = np.stack([t2, t2 ** 2, t2 ** 3, t2 ** 4], 1)
        poly = 1 + pw @ k
        thd = th * poly
        small = r < 1e-8
        rs = np.where(small, 1.0, r)
        g = np.where(small, 1.0, thd / rs)
        xd = xn * g[:, None]
        if not jac:
            return xd
        dthd = 1 + pw @ (k * [3, 5, 7, 9])
        dg_dr = (dthd / (1 + r2) * r - thd) / rs ** 2
        h = np.where(small, 2 * (k[0] - 1 / 3), dg_dr / rs)
        Jx = g[:, None, None] * np.eye(2) + h[:, None, None] * np.einsum('ni,nj->nij', xn, xn)
        Jk = (th[:, None] * pw / rs[:, None])[:, None, :] * xn[:, :, None]
        Jk[small] = 0
        return xd, Jx, Jk
    k1, k2, p1, p2, k3 = _brown_full(model, k)
    s = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = np.empty_like(xn)
    xd[:, 0] = s * x + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    xd[:, 1] = s * y + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    if not jac:
        return xd
    ds = k1 + r2 * (2 * k2 + 3 * k3 * r2)
    Jx = np.empty((N, 2, 2))
    Jx[:, 0, 0] = s + 2 * x * x * ds + 2 * p1 * y + 6 * p2 * x
    Jx[:, 0, 1] = Jx[:, 1, 0] = 2 * x * y * ds + 2 * p1 * x + 2 * p2 * y
    Jx[:, 1, 1] = s + 2 * y * y * ds + 6 * p1 * y + 2 * p2 * x
    full = np.empty((N, 2, 5))
    full[:, :, 0] = xn * r2[:, None]
    full[:, :, 1] = xn * (r2 ** 2)[:, None]
    full[:, :, 4] = xn * (r2 ** 3)[:, None]
    full[:, 0, 2], full[:, 1, 2] = 2 * x * y, r2 + 2 * y * y
    full[:, 0, 3], full[:, 1, 3] = r2 + 2 * x * x, 2 * x * y
    cols = [0, 1, 2, 3, 4] if model == 'brown' else [0, 1, 4][:len(k)]
    return xd, Jx, full[:, :, cols]


def undistort(xd, model, k, iters=30, tol=1e-12):
    """Invert distort() by Newton's method (vectorized)."""
    xd = np.asarray(xd, float)
    if model == 'pinhole' or (model != 'fisheye' and not np.any(k)):
        return xd.copy()
    xn = xd.copy()
    for _ in range(iters):
        f, J, _ = distort(xn, model, k, jac=True)
        res = f - xd
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        dx = (J[:, 1, 1] * res[:, 0] - J[:, 0, 1] * res[:, 1]) / det
        dy = (-J[:, 1, 0] * res[:, 0] + J[:, 0, 0] * res[:, 1]) / det
        xn[:, 0] -= dx
        xn[:, 1] -= dy
        if np.max(np.abs(res)) < tol:
            break
    return xn


def undistort_maps(cam, out_K=None):
    """cv2.remap maps producing a pinhole image with intrinsics out_K (default cam.K)."""
    w, h = cam.width, cam.height
    Kp = cam.K if out_K is None else out_K
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    xn = np.stack([(u - Kp[0, 2]) / Kp[0, 0], (v - Kp[1, 2]) / Kp[1, 1]], -1).reshape(-1, 2)
    uv = cam.denormalize(xn).reshape(h, w, 2).astype(np.float32)
    return uv[..., 0], uv[..., 1]


def undistort_image(img, cam, out_K=None):
    import cv2
    mx, my = undistort_maps(cam, out_K)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def max_displacement_px(cam):
    """Largest |distorted - undistorted| pixel displacement over the image border."""
    if not cam.distorted:
        return 0.0
    w, h = cam.width, cam.height
    t = np.linspace(0, 1, 50)
    border = np.concatenate([np.c_[t * w, 0 * t], np.c_[t * w, h + 0 * t],
                             np.c_[0 * t, t * h], np.c_[w + 0 * t, t * h]])
    xd = (border - [cam.cx, cam.cy]) / [cam.fx, cam.fy]
    xn = undistort(xd, cam.model, cam.dist)
    return float(np.max(np.linalg.norm((xd - xn) * [cam.fx, cam.fy], axis=1)))
