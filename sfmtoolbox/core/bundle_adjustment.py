"""Sparse bundle adjustment with per-camera-group intrinsics and analytic Jacobian.

Parameters: 6 per free image [axis-angle | t] (gauge: fixed_poses), 3 per point,
and per camera group optionally f (fx with fy/fx held fixed), (cx, cy) and the lens
distortion coefficients (see distortion.py).
Residuals are pixel reprojection errors; robust loss as in scipy (huber/linear).
Default solver: Levenberg-Marquardt on the Schur-complemented (reduced camera)
system; solver='scipy' uses scipy's TRF with the same analytic Jacobian.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.optimize import least_squares, OptimizeResult
from scipy.sparse import bsr_matrix, csr_matrix, coo_matrix, hstack
from sfmtoolbox.core.pose import rodrigues, rodrigues_inv, rodrigues_batch
from sfmtoolbox.core.distortion import distort


def _rotate(points, rvecs):
    """Row-wise Rodrigues rotation."""
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        k = np.where(theta > 1e-12, rvecs / theta, 0.0)
    cos, sin = np.cos(theta), np.sin(theta)
    return (points * cos
            + np.cross(k, points) * sin
            + k * np.einsum('ij,ij->i', k, points)[:, None] * (1 - cos))


def _skew(v):
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def _rot_and_jac(rvecs):
    """R(r) and G(r) = R (r r^T + (R^T - I)[r]x) / |r|^2, so that
    d(R p)/dr = -[R p]x G  (Gallego & Yezzi 2015); G -> I as r -> 0."""
    R = rodrigues_batch(rvecs)
    th2 = np.einsum('ij,ij->i', rvecs, rvecs)
    small = th2 < 1e-20
    M = (np.einsum('ni,nj->nij', rvecs, rvecs)
         + (R.transpose(0, 2, 1) - np.eye(3)) @ _skew(rvecs))
    M /= np.where(small, 1.0, th2)[:, None, None]
    G = R @ M
    G[small] = np.eye(3)
    return R, G


def _csr_pattern(rows, cols, shape):
    """CSR structure for fixed COO (rows, cols) plus the COO->CSR data permutation."""
    m = coo_matrix((np.arange(1, len(rows) + 1, dtype=float), (rows, cols)),
                   shape=shape).tocsr()
    return m.data.astype(np.int64) - 1, m.indices, m.indptr


def _inv3_sym(A):
    """Batched inverse of symmetric 3x3 matrices via the adjugate."""
    a, b, c = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    d, e, f = A[:, 1, 1], A[:, 1, 2], A[:, 2, 2]
    c00, c01, c02 = d * f - e * e, c * e - b * f, b * e - c * d
    c11, c12, c22 = a * f - c * c, b * c - a * e, a * d - b * b
    det = a * c00 + b * c01 + c * c02
    if not np.all(np.isfinite(det)) or np.any(np.abs(det) < 1e-300):
        raise np.linalg.LinAlgError('singular point block')
    out = np.empty_like(A)
    out[:, 0, 0], out[:, 0, 1], out[:, 0, 2] = c00, c01, c02
    out[:, 1, 0], out[:, 1, 1], out[:, 1, 2] = c01, c11, c12
    out[:, 2, 0], out[:, 2, 1], out[:, 2, 2] = c02, c12, c22
    return out / det[:, None, None]


def _robust(r, loss, f_scale):
    """scipy-compatible robust cost 0.5 f^2 sum rho((r/f)^2) and IRLS weights rho'."""
    if loss == 'linear':
        return 0.5 * float(r @ r), np.ones_like(r)
    z = (r / f_scale) ** 2
    big = z > 1
    rho = np.where(big, 2 * np.sqrt(z) - 1, z)
    w = np.where(big, 1 / np.sqrt(np.where(big, z, 1)), 1.0)
    return 0.5 * f_scale ** 2 * float(rho.sum()), w


class _Problem:
    def __init__(self, pose_params, points3d, img_idx, pt_idx, obs_uv, cameras,
                 cam_of_image, fixed_poses):
        order = np.argsort(pt_idx, kind='stable')
        self.order = order
        self.img_idx = np.asarray(img_idx, np.int64)[order]
        self.pt_idx = np.asarray(pt_idx, np.int64)[order]
        self.obs_uv = np.asarray(obs_uv, float)[order]
        self.pose_params = np.asarray(pose_params, float)
        n_imgs, self.n_pts = len(pose_params), len(points3d)
        self.n_obs = n = len(self.obs_uv)
        cam_of_image = np.asarray(cam_of_image)
        self.obs_group = cam_of_image[self.img_idx]

        fixed = np.zeros(n_imgs, bool)
        fixed[list(fixed_poses)] = True
        self.free_img = np.flatnonzero(~fixed)
        slot = np.full(n_imgs, -1)
        slot[self.free_img] = np.arange(len(self.free_img))
        self.nP, self.nX = 6 * len(self.free_img), 3 * self.n_pts

        groups = np.unique(self.obs_group)
        ng = int(cam_of_image.max()) + 1
        self.base = np.zeros((4, ng))
        for g in groups:
            c = cameras[g]
            self.base[:, g] = c.fx, c.fy, c.cx, c.cy
        self.models = {g: (cameras[g].model, np.asarray(cameras[g].dist, float).copy())
                       for g in groups.tolist()}
        self.dist_groups = [g for g, (m, _) in self.models.items() if m != 'pinhole']
        self.ratio = np.divide(self.base[1], self.base[0], out=np.ones(ng),
                               where=self.base[0] != 0)
        self.intr_offset, self.intr_layout = {}, {}
        off = 0
        for g in sorted(groups.tolist()):
            lay = (['f'] if cameras[g].refine_focal else []) + \
                  (['cx', 'cy'] if cameras[g].refine_principal else []) + \
                  ([f'd{j}' for j in range(len(self.models[g][1]))]
                   if getattr(cameras[g], 'refine_dist', False) else [])
            if lay:
                self.intr_offset[g], self.intr_layout[g] = off, lay
                off += len(lay)
        self.nI = off
        self.nC = self.nP + self.nI
        self.n_params = self.nP + self.nX + self.nI

        obs = np.arange(n)
        self.fobs = obs[slot[self.img_idx] >= 0]
        rows, cols = [], []
        cblocks = []
        if len(self.fobs):
            cblocks.append((self.fobs, (6 * slot[self.img_idx[self.fobs]])[:, None]
                            + np.arange(6)))
        self.intr_obs = []
        for g, o in self.intr_offset.items():
            og = obs[self.obs_group == g]
            k = len(self.intr_layout[g])
            cblocks.append((og, np.broadcast_to(self.nP + o + np.arange(k), (len(og), k))))
            self.intr_obs.append((g, og))
        for ob, cl in cblocks:
            w = cl.shape[1]
            for a in (0, 1):
                rows.append(np.repeat(2 * ob + a, w))
                cols.append(np.ascontiguousarray(cl).ravel())
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        self.c_pat = _csr_pattern(rows, cols, (2 * n, self.nC))
        prow = np.repeat(np.r_[2 * obs, 2 * obs + 1], 3)
        pcol = np.tile((3 * self.pt_idx[:, None] + np.arange(3)).ravel(), 2)
        self.p_pat = _csr_pattern(prow, pcol, (2 * n, self.nX))
        self.pt_start = np.searchsorted(self.pt_idx, np.arange(self.n_pts))
        self.dense_pts = n > 0 and bool(np.all(np.bincount(self.pt_idx, minlength=self.n_pts) > 0))
        self._cache_x = None

    def x0(self, points3d):
        intr0 = [self.models[g][1][int(name[1:])] if name[0] == 'd' else
                 self.base[{'f': 0, 'cx': 2, 'cy': 3}[name], g]
                 for g in self.intr_offset for name in self.intr_layout[g]]
        return np.concatenate([self.pose_params[self.free_img].ravel(),
                               np.asarray(points3d, float).ravel(), intr0])

    def intrinsics(self, x):
        return self.intrinsics_full(x)[:4]

    def intrinsics_full(self, x):
        fx, fy, cx, cy = (a.copy() for a in self.base)
        dist = {g: d.copy() for g, (_, d) in self.models.items()}
        xi = x[self.nP + self.nX:]
        for g, o in self.intr_offset.items():
            for k, name in enumerate(self.intr_layout[g]):
                if name == 'f':
                    fx[g], fy[g] = xi[o + k], xi[o + k] * self.ratio[g]
                elif name == 'cx':
                    cx[g] = xi[o + k]
                elif name == 'cy':
                    cy[g] = xi[o + k]
                else:
                    dist[g][int(name[1:])] = xi[o + k]
        return fx, fy, cx, cy, dist

    def unpack(self, x):
        poses = self.pose_params.copy()
        poses[self.free_img] = x[:self.nP].reshape(-1, 6)
        return poses, x[self.nP:self.nP + self.nX].reshape(-1, 3)

    def forward(self, x):
        if self._cache_x is not None and np.array_equal(self._cache_x, x):
            return self._c
        poses, pts = self.unpack(x)
        R, G = _rot_and_jac(poses[:, :3])
        Ro = R[self.img_idx]
        q = np.matmul(Ro, pts[self.pt_idx][:, :, None])[:, :, 0]
        Xc = q + poses[self.img_idx, 3:]
        clipped = Xc[:, 2] < 1e-9
        z = np.maximum(Xc[:, 2], 1e-9)
        *intr, dist = self.intrinsics_full(x)
        fx, fy, cx, cy = (a[self.obs_group] for a in intr)
        xn, yn = Xc[:, 0] / z, Xc[:, 1] / z
        xd, yd = xn, yn
        Jd, Jk = None, {}
        if self.dist_groups:
            xd, yd = xn.copy(), yn.copy()
            Jd = np.zeros((self.n_obs, 2, 2))
            Jd[:, 0, 0] = Jd[:, 1, 1] = 1.0
            for g in self.dist_groups:
                m = self.obs_group == g
                d, Jd[m], Jk[g] = distort(np.c_[xn[m], yn[m]], self.models[g][0], dist[g],
                                          jac=True)
                xd[m], yd[m] = d[:, 0], d[:, 1]
        r = np.empty(2 * self.n_obs)
        r[0::2] = fx * xd + cx - self.obs_uv[:, 0]
        r[1::2] = fy * yd + cy - self.obs_uv[:, 1]
        self._cache_x = x.copy()
        self._c = dict(Ro=Ro, G=G, q=q, z=z, xn=xn, yn=yn, xd=xd, yd=yd, Jd=Jd, Jk=Jk,
                       fx=fx, fy=fy, clipped=clipped, r=r)
        return self._c

    def residuals(self, x):
        return self.forward(x)['r']

    def jac_blocks(self, x):
        """-> (Jc csr (2n, nC) over [poses | intrinsics], Jp (n,2,3) d r / d X)."""
        c = self.forward(x)
        n = self.n_obs
        z, xn, yn, fx, fy = c['z'], c['xn'], c['yn'], c['fx'], c['fy']
        Jx = np.zeros((n, 2, 3))
        Jx[:, 0, 0] = 1 / z
        Jx[:, 0, 2] = -xn / z
        Jx[:, 1, 1] = 1 / z
        Jx[:, 1, 2] = -yn / z
        Jx[c['clipped'], :, 2] = 0.0
        if c['Jd'] is not None:
            Jx = np.matmul(c['Jd'], Jx)
        Jx[:, 0] *= fx[:, None]
        Jx[:, 1] *= fy[:, None]
        vals = []
        if len(self.fobs):
            f = self.fobs
            Jr = -Jx[f] @ (_skew(c['q'][f]) @ c['G'][self.img_idx[f]])
            vals.append(np.concatenate([Jr, Jx[f]], axis=2))
        for g, og in self.intr_obs:
            d = np.zeros((len(og), 2, len(self.intr_layout[g])))
            for k, name in enumerate(self.intr_layout[g]):
                if name == 'f':
                    d[:, 0, k], d[:, 1, k] = c['xd'][og], self.ratio[g] * c['yd'][og]
                elif name == 'cx':
                    d[:, 0, k] = 1.0
                elif name == 'cy':
                    d[:, 1, k] = 1.0
                else:
                    j = int(name[1:])
                    d[:, 0, k] = fx[og] * c['Jk'][g][:, 0, j]
                    d[:, 1, k] = fy[og] * c['Jk'][g][:, 1, j]
            vals.append(d)
        data = (np.concatenate([np.concatenate([v[:, 0].ravel(), v[:, 1].ravel()])
                                for v in vals]) if vals else np.zeros(0))
        perm, ind, ptr = self.c_pat
        Jc = csr_matrix((data[perm], ind, ptr), shape=(2 * n, self.nC))
        return Jc, Jx @ c['Ro']

    def dense_cam(self, Jc):
        D = np.zeros((Jc.shape[0], self.nC))
        D[np.repeat(np.arange(Jc.shape[0]), np.diff(Jc.indptr)), Jc.indices] = Jc.data
        return D

    def jac_full(self, x):
        Jc, Jp = self.jac_blocks(x)
        perm, ind, ptr = self.p_pat
        data = np.concatenate([Jp[:, 0].ravel(), Jp[:, 1].ravel()])
        Jpts = csr_matrix((data[perm], ind, ptr), shape=(2 * self.n_obs, self.nX))
        J = hstack([Jc[:, :self.nP], Jpts, Jc[:, self.nP:]], format='csr')
        return J

    def to_cam_pt(self, v):
        return np.r_[v[:self.nP], v[self.nP + self.nX:]], v[self.nP:self.nP + self.nX]

    def from_cam_pt(self, vc, vp):
        return np.r_[vc[:self.nP], vp, vc[self.nP:]]


def _lm_schur(prob, x0, loss, f_scale, max_nfev, ftol=1e-8, xtol=1e-10, gtol=1e-10,
              dense_max=512):
    """Levenberg-Marquardt (Marquardt diagonal scaling) with IRLS robust weights.
    Points are eliminated per 3x3 block: S = U - W V^-1 W^T on the camera block."""
    x = x0.copy()
    r = prob.residuals(x)
    cost, w = _robust(r, loss, f_scale)
    nfev, njev, lam, nu = 1, 0, None, 2.0
    n = prob.n_obs
    status = 0
    while nfev < max_nfev:
        Jc, Jp = prob.jac_blocks(x)
        njev += 1
        wr = (w * r).reshape(n, 2)
        w2 = w.reshape(n, 2)
        dense = prob.nC <= dense_max
        if dense:
            Jcd = prob.dense_cam(Jc)
            U = Jcd.T @ (w[:, None] * Jcd)
            gc = Jcd.T @ (w * r)
        else:
            U = (Jc.multiply(w[:, None]).tocsr().T @ Jc).toarray()
            gc = Jc.T @ (w * r)
        JpW = Jp * w2[:, :, None]
        Vo = np.matmul(JpW.transpose(0, 2, 1), Jp)
        go = (Jp * wr[:, :, None]).sum(1)
        V = np.zeros((prob.n_pts, 3, 3))
        gp = np.zeros((prob.n_pts, 3))
        if prob.dense_pts:
            V[:] = np.add.reduceat(Vo, prob.pt_start, axis=0)
            gp[:] = np.add.reduceat(go, prob.pt_start, axis=0)
        else:
            np.add.at(V, prob.pt_idx, Vo)
            np.add.at(gp, prob.pt_idx, go)
        if prob.nC:
            perm, ind, ptr = prob.p_pat
            data = np.concatenate([JpW[:, 0].ravel(), JpW[:, 1].ravel()])
            JpW_s = csr_matrix((data[perm], ind, ptr), shape=(2 * n, prob.nX))
            if dense:
                WT = np.asarray(JpW_s.T @ Jcd)
                WT3 = WT.reshape(prob.n_pts, 3, prob.nC)
            else:
                Wm = (Jc.T @ JpW_s).tocsr()
        g_full = np.r_[gc, gp.ravel()]
        if np.max(np.abs(g_full), initial=0.0) < gtol:
            status = 1
            break
        dU = np.diag(U).copy() if prob.nC else np.zeros(0)
        dV = np.diagonal(V, axis1=1, axis2=2).copy()
        dU = np.maximum(dU, 1e-12)
        dV = np.maximum(dV, 1e-12)
        if lam is None:
            lam = 1e-4
        while nfev < max_nfev:
            Vd = V.copy()
            Vd[:, [0, 1, 2], [0, 1, 2]] += lam * dV
            try:
                Vinv = _inv3_sym(Vd)
            except np.linalg.LinAlgError:
                lam *= nu
                nu *= 2
                continue
            Vg = np.matmul(Vinv, gp[:, :, None])[:, :, 0]
            if prob.nC:
                if dense:
                    S = U + np.diag(lam * dU) - WT.T @ np.matmul(Vinv, WT3).reshape(prob.nX, -1)
                    rhs = -gc + WT.T @ Vg.ravel()
                else:
                    Vinv_s = bsr_matrix((Vinv, np.arange(prob.n_pts), np.arange(prob.n_pts + 1)),
                                        shape=(prob.nX, prob.nX)).tocsr()
                    S = U + np.diag(lam * dU) - (Wm @ Vinv_s @ Wm.T).toarray()
                    rhs = -gc + Wm @ Vg.ravel()
                try:
                    dc = cho_solve(cho_factor(S), rhs)
                except LinAlgError:
                    lam *= nu
                    nu *= 2
                    continue
                Wdc = WT @ dc if dense else Wm.T @ dc
                dp = -(Vg + np.matmul(Vinv, Wdc.reshape(-1, 3, 1))[:, :, 0])
            else:
                dc = np.zeros(0)
                dp = -Vg
            step = prob.from_cam_pt(dc, dp.ravel())
            x_new = x + step
            r_new = prob.residuals(x_new)
            nfev += 1
            cost_new, w_new = _robust(r_new, loss, f_scale)
            if np.isfinite(cost_new) and cost_new < cost:
                dcost = cost - cost_new
                x, r, w = x_new, r_new, w_new
                prev, cost = cost, cost_new
                lam = max(lam / 3.0, 1e-12)
                nu = 2.0
                if dcost < ftol * prev:
                    status = 2
                elif np.linalg.norm(step) < xtol * (xtol + np.linalg.norm(x)):
                    status = 3
                break
            lam *= nu
            nu *= 2
            if lam > 1e16:
                status = 3
                break
        if status:
            break
    return OptimizeResult(x=x, fun=r, cost=cost, nfev=nfev, njev=njev,
                          status=status, success=status > 0)


def bundle_adjust(pose_params, points3d, img_idx, pt_idx, obs_uv,
                  cameras, cam_of_image, fixed_poses=(0,),
                  huber_px=2.0, max_nfev=100, verbose=0, solver='lm'):
    """
    pose_params (n_images,6) world->camera; points3d (n_pts,3);
    img_idx, pt_idx, obs_uv (n_obs,), (n_obs,), (n_obs,2); cam_of_image (n_images,).
    Returns (pose_params, points3d, result); result.fun is in the caller's
    observation order. Refined intrinsics are written to `cameras`.
    """
    prob = _Problem(pose_params, points3d, img_idx, pt_idx, obs_uv, cameras,
                    cam_of_image, fixed_poses)
    x0 = prob.x0(points3d)
    r0 = np.abs(prob.residuals(x0))
    s_robust = 1.4826 * np.median(np.abs(r0 - np.median(r0))) + np.median(r0)
    stages = ['linear', 'huber'] if s_robust > 3 * huber_px else ['huber']

    res = None
    for loss in stages:
        if solver == 'scipy':
            res = least_squares(prob.residuals, x0, jac=prob.jac_full, method='trf',
                                tr_solver='lsmr', loss=loss, f_scale=huber_px,
                                x_scale='jac', max_nfev=max_nfev, verbose=verbose,
                                ftol=1e-6)
        else:
            res = _lm_schur(prob, x0, loss, huber_px, max_nfev)
        x0 = res.x
    poses, pts = prob.unpack(res.x)
    inv = np.empty_like(prob.order)
    inv[prob.order] = np.arange(len(inv))
    res.fun = res.fun.reshape(-1, 2)[inv].ravel()

    fx, fy, cx, cy, dist = prob.intrinsics_full(res.x)
    for g in prob.intr_offset:
        cameras[g].fx, cameras[g].fy = fx[g], fy[g]
        cameras[g].cx, cameras[g].cy = cx[g], cy[g]
        if len(dist[g]):
            cameras[g].dist = dist[g]
    return poses, pts, res


def pack_pose(R, t):
    return np.concatenate([rodrigues_inv(R), t])


def unpack_pose(p):
    return rodrigues(p[:3]), p[3:]
