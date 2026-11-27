"""Incremental Structure-from-Motion with per-camera-group intrinsics.

Poses map world -> camera: X_cam = R X + t, camera center C = -R^T t.
"""
import time
from collections.abc import MutableMapping

import numpy as np
from sfmtoolbox.core import geometry as geo
from sfmtoolbox.core.pose import recover_pose
from sfmtoolbox.core.triangulation import (triangulate_points, triangulate_multiview,
                                           triangulation_angles, project)
from sfmtoolbox.core.pnp import ransac_pnp
from sfmtoolbox.core.bundle_adjustment import bundle_adjust, pack_pose, unpack_pose


class PointMap(MutableMapping):
    """{track_id: X} backed by a dense (n_tracks,3) array and a validity mask."""

    def __init__(self, n):
        self.X = np.zeros((n, 3))
        self.has = np.zeros(n, bool)

    def __getitem__(self, tid):
        if not self.__contains__(tid):
            raise KeyError(tid)
        return self.X[tid]

    def __setitem__(self, tid, X):
        self.X[tid] = X
        self.has[tid] = True

    def __delitem__(self, tid):
        if not self.__contains__(tid):
            raise KeyError(tid)
        self.has[tid] = False

    def __contains__(self, tid):
        try:
            return 0 <= tid < len(self.has) and bool(self.has[tid])
        except TypeError:
            return False

    def __iter__(self):
        return iter(np.flatnonzero(self.has).tolist())

    def __len__(self):
        return int(np.count_nonzero(self.has))

    def ids(self):
        return np.flatnonzero(self.has)

    def array(self):
        return self.X[self.has]


class Reconstruction:
    def __init__(self, keypoints, tracks, cameras, cam_of_image,
                 ransac_px=2.0, min_tri_angle_deg=1.5, max_reproj_px=4.0,
                 verbose=True, log_fn=None):
        self.kps = keypoints
        self.tracks = tracks
        self.cameras = cameras
        self.cam_of_image = cam_of_image
        self.ransac_px = ransac_px
        self.min_angle = np.deg2rad(min_tri_angle_deg)
        self.max_reproj = max_reproj_px
        self.verbose = verbose
        self.log_fn = log_fn or print

        self.n_images = len(keypoints)
        self.poses = {}
        self.points = PointMap(len(tracks))
        self.error_history = []
        self.registration = {}
        self.ba_log = []
        self.timings = {}
        self.min_init_points = 30
        self.max_init_attempts = 10
        self.init_pair = None

        lens = np.fromiter((len(t) for t in tracks), np.int64, len(tracks))
        self.obs_tid = np.repeat(np.arange(len(tracks)), lens)
        self.obs_img = np.fromiter((i for t in tracks for i in t), np.int64, lens.sum())
        self.obs_feat = np.fromiter((f for t in tracks for f in t.values()), np.int64,
                                    lens.sum())
        self.obs_uv = np.zeros((len(self.obs_img), 2))
        for img in range(self.n_images):
            m = self.obs_img == img
            if m.any():
                self.obs_uv[m] = np.asarray(keypoints[img])[self.obs_feat[m]]
        self._blocked = np.zeros(self.n_images, bool)

    @property
    def img_tracks(self):
        out = {i: [] for i in range(self.n_images)}
        for tid, img, feat in zip(self.obs_tid.tolist(), self.obs_img.tolist(),
                                  self.obs_feat.tolist()):
            if not self._blocked[img]:
                out[img].append((tid, feat))
        return out

    def cam(self, img):
        return self.cameras[self.cam_of_image[img]]

    def K(self, img):
        return self.cam(img).K

    def _norm(self, img, feats):
        return self.cam(img).normalize(self.kps[img][feats])

    def _center(self, img):
        R, t = self.poses[img]
        return -R.T @ t

    def _log(self, *a):
        if self.verbose:
            self.log_fn(' '.join(str(x) for x in a))

    def _registered(self):
        reg = np.zeros(self.n_images, bool)
        reg[list(self.poses)] = True
        return reg

    def _pose_arrays(self):
        R = np.tile(np.eye(3), (self.n_images, 1, 1))
        t = np.zeros((self.n_images, 3))
        for img, (Ri, ti) in self.poses.items():
            R[img], t[img] = Ri, ti
        return R, t, self._registered()

    def _by_camera(self, img):
        cam_ids = np.asarray(self.cam_of_image)[img]
        for g in np.unique(cam_ids):
            yield self.cameras[g], cam_ids == g

    def _project_obs(self, Xc, img):
        """Camera-frame points observed in images `img` -> distorted pixels."""
        uv = np.empty((len(Xc), 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            for cam, m in self._by_camera(img):
                uv[m] = cam.project(Xc[m])
        return uv

    def _normalize_obs(self, uv, img):
        xn = np.empty_like(uv)
        for cam, m in self._by_camera(img):
            xn[m] = cam.normalize(uv[m])
        return xn

    def _intrinsics(self):
        f = np.array([[self.cam(i).fx, self.cam(i).fy, self.cam(i).cx, self.cam(i).cy]
                      for i in range(self.n_images)])
        return f[:, :2], f[:, 2:]

    def _gather(self, tids, reg):
        """Observations of sorted `tids` in registered images, grouped by track."""
        q = np.zeros(len(self.tracks), bool)
        q[tids] = True
        sel = np.flatnonzero(q[self.obs_tid] & reg[self.obs_img])
        seg = np.searchsorted(tids, self.obs_tid[sel])
        counts = np.bincount(seg, minlength=len(tids))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = np.arange(len(sel)) - starts[seg]
        return sel, seg, pos, counts

    def _check(self, tids, X, R, t, reg, gathered=None):
        """Vectorized acceptance: positive depth, max reprojection error, and
        max pairwise ray angle over all registered observers."""
        if not len(tids):
            return np.zeros(0, bool)
        sel, seg, pos, counts = gathered or self._gather(tids, reg)
        img = self.obs_img[sel]
        Xc = np.einsum('nij,nj->ni', R[img], X[seg]) + t[img]
        uv = self._project_obs(Xc, img)
        err = np.linalg.norm(uv - self.obs_uv[sel], axis=1)
        bad = np.zeros(len(tids), bool)
        np.logical_or.at(bad, seg, ~(Xc[:, 2] > 0) | ~(err <= self.max_reproj))
        C = -np.einsum('nji,nj->ni', R, t)
        rays = X[seg] - C[img]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
        k = int(counts.max())
        Rp = np.zeros((len(tids), k, 3))
        Rp[seg, pos] = rays
        valid = np.arange(k)[None] < counts[:, None]
        cos = np.einsum('nai,nbi->nab', Rp, Rp)
        cos[~(valid[:, :, None] & valid[:, None, :])] = 1.0
        max_ang = np.arccos(np.clip(cos.reshape(len(tids), -1).min(1), -1, 1))
        return ~bad & (counts >= 2) & (max_ang >= self.min_angle)

    def _accept_point(self, X, obs):
        R, t, reg = self._pose_arrays()
        depths, errs, centers = [], [], []
        for img, feat in obs:
            Xc = X[None] @ R[img].T + t[img]
            uv, z = self.cam(img).project(Xc), Xc[:, 2]
            depths.append(z[0])
            errs.append(np.linalg.norm(uv[0] - self.kps[img][feat]))
            centers.append(-R[img].T @ t[img])
        if min(depths) <= 0 or max(errs) > self.max_reproj:
            return False
        max_ang = max((triangulation_angles(X[None], centers[a], centers[b])[0]
                       for a in range(len(centers)) for b in range(a + 1, len(centers))),
                      default=0.0)
        return max_ang >= self.min_angle

    def select_init_pair(self, pair_data):
        c = self.init_candidates(pair_data)
        return c[0][1:] if c else None

    def init_candidates(self, pair_data):
        """Valid initial pairs as (score, i, j, R, t), best first. Score = inliers x
        clipped median triangulation angle."""
        out = []
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        for (i, j), d in sorted(pair_data.items()):
            n_inl = d['inliers'].sum()
            if n_inl < 50:
                continue
            x1n = self.cam(i).normalize(d['x1'][d['inliers']])
            x2n = self.cam(j).normalize(d['x2'][d['inliers']])
            R, t, good = recover_pose(d['E'], x1n, x2n)
            if good.sum() < 0.7 * n_inl:
                continue
            X = triangulate_points(P1, np.hstack([R, t[:, None]]), x1n[good], x2n[good])
            med = np.median(triangulation_angles(X, np.zeros(3), -R.T @ t))
            score = n_inl * min(med / self.min_angle, 3.0)
            if med > 0.5 * self.min_angle:
                out.append((score, i, j, R, t))
        out.sort(key=lambda c: -c[0])
        return out

    def initialize(self, i, j, R, t, tracks_ij, relaxed=False):
        """relaxed: skip the reprojection gate (cheirality and angle only) so that an
        initial pose biased by unmodelled distortion or a poor focal can still be
        refined by BA before strict filtering."""
        self.poses[i] = (np.eye(3), np.zeros(3))
        self.poses[j] = (R, t)
        keep = self.max_reproj
        if relaxed:
            self.max_reproj = np.inf
        try:
            self._triangulate_tracks(tracks_ij)
        finally:
            self.max_reproj = keep
        self._log(f"[init] pair ({i},{j}): {len(self.points)} points")

    def _triangulate_tracks(self, track_ids):
        tids = np.unique(np.asarray(track_ids, dtype=np.int64))
        tids = tids[~self.points.has[tids]]
        if not len(tids):
            return 0
        R, t, reg = self._pose_arrays()
        sel, seg, pos, counts = self._gather(tids, reg)
        keep = counts >= 2
        if not keep.any():
            return 0
        if not keep.all():
            tids = tids[keep]
            sel, seg, pos, counts = self._gather(tids, reg)
        img = self.obs_img[sel]
        xn = self._normalize_obs(self.obs_uv[sel], img)
        P = np.concatenate([R, t[:, :, None]], axis=2)
        X = triangulate_multiview(P[img], xn, seg, pos, len(tids), int(counts.max()))
        ok = self._check(tids, X, R, t, reg, (sel, seg, pos, counts))
        self.points.X[tids[ok]] = X[ok]
        self.points.has[tids[ok]] = True
        return int(ok.sum())

    def next_image(self):
        n = np.bincount(self.obs_img[self.points.has[self.obs_tid]],
                        minlength=self.n_images)
        n[self._registered() | self._blocked] = 0
        if n.max() <= 0:
            return None, 0
        best = int(np.argmax(n))
        return best, int(n[best])

    def register_image(self, img):
        m = (self.obs_img == img) & self.points.has[self.obs_tid]
        tids = self.obs_tid[m]
        if len(tids) < 6:
            return False
        X = self.points.X[tids]
        xn = self.cam(img).normalize(self.obs_uv[m])
        out = ransac_pnp(X, xn, thresh_n=self.ransac_px / self.cam(img).f,
                         rng=np.random.default_rng(img))
        if out is None or out[2].sum() < 10:
            self.registration[img] = dict(n_corr=len(tids), n_inliers=0 if out is None
                                          else int(out[2].sum()), registered=False)
            return False
        R, t, inl = out
        self.poses[img] = (R, t)
        self.registration[img] = dict(n_corr=len(tids), n_inliers=int(inl.sum()),
                                      registered=True, order=len(self.poses) - 1)
        self._log(f"[register] image {img}: {inl.sum()}/{len(tids)} PnP inliers")
        return True

    def filter_points(self):
        tids = self.points.ids()
        if not len(tids):
            return 0
        R, t, reg = self._pose_arrays()
        ok = self._check(tids, self.points.X[tids], R, t, reg)
        self.points.has[tids[~ok]] = False
        return int((~ok).sum())

    def _obs_mask(self):
        return self.points.has[self.obs_tid] & self._registered()[self.obs_img]

    def _observations(self):
        sel = np.flatnonzero(self._obs_mask())
        imgs = sorted(self.poses)
        img_slot = np.full(self.n_images, -1)
        img_slot[imgs] = np.arange(len(imgs))
        tids = self.points.ids()
        return (imgs, tids.tolist(), img_slot[self.obs_img[sel]],
                np.searchsorted(tids, self.obs_tid[sel]), self.obs_uv[sel].copy())

    def observation_errors(self):
        """(obs indices, per-observation reprojection error px, depth) for all
        observations of triangulated points in registered images."""
        sel = np.flatnonzero(self._obs_mask())
        R, t, _ = self._pose_arrays()
        img = self.obs_img[sel]
        Xc = np.einsum('nij,nj->ni', R[img], self.points.X[self.obs_tid[sel]]) + t[img]
        uv = self._project_obs(Xc, img)
        return sel, np.linalg.norm(uv - self.obs_uv[sel], axis=1), Xc[:, 2]

    def reprojection_errors(self):
        sel, err, _ = self.observation_errors()
        img = self.obs_img[sel]
        out = {}
        for i in self.poses:
            e = err[img == i]
            if len(e):
                out[i] = e
        return out

    def mean_reprojection_error(self):
        _, err, _ = self.observation_errors()
        return float(err.mean()) if len(err) else np.nan

    def run_ba(self, label=''):
        imgs, tids, img_idx, pt_idx, uv = self._observations()
        if len(uv) < 20:
            return
        t0 = time.perf_counter()
        poses = np.array([pack_pose(*self.poses[im]) for im in imgs])
        pts = self.points.X[tids]
        group = [self.cam_of_image[im] for im in imgs]
        poses, pts, res = bundle_adjust(poses, pts, img_idx, pt_idx, uv,
                                        self.cameras, group, fixed_poses=(0,),
                                        huber_px=self.ransac_px)
        for k, im in enumerate(imgs):
            self.poses[im] = unpack_pose(poses[k])
        self.points.X[tids] = pts
        err = self.mean_reprojection_error()
        self.error_history.append((label, err))
        self.ba_log.append(dict(label=label, n_images=len(imgs), n_points=len(tids),
                                n_obs=len(uv), nfev=int(res.nfev), cost=float(res.cost),
                                mean_reproj=err, seconds=time.perf_counter() - t0))
        refined = [c for c in self.cameras if c.refine_focal or c.refine_principal]
        extra = (' | ' + ', '.join(f'cam{c.cam_id} f={c.f:.1f}' for c in refined[:4])
                 if refined else '')
        self._log(f"[BA {label}] mean reproj = {err:.3f} px, "
                  f"{len(self.points)} pts, {len(self.poses)} cams{extra}")

    def _focal_search(self, pair_data):
        """Grid search of a shared focal scale on the strongest pair (median
        reprojection after E -> pose -> triangulation). Avoids the focal/depth
        local minimum when the initial focal is only heuristic."""
        (i, j), d = max(pair_data.items(), key=lambda kv: kv[1]['inliers'].sum())
        x1 = d['x1'][d['inliers']]
        x2 = d['x2'][d['inliers']]
        base_f = {c.cam_id: c.f for c in self.cameras}
        best_s, best_err = 1.0, np.inf
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        for s in np.linspace(0.5, 1.6, 23):
            for c in self.cameras:
                c.fx = c.fy = base_f[c.cam_id] * s
            E = geo.essential_from_fundamental(d['F'], self.K(i), self.K(j))
            x1n = self.cam(i).normalize(x1)
            x2n = self.cam(j).normalize(x2)
            R, t, good = recover_pose(E, x1n, x2n)
            if good.sum() < 0.5 * len(x1n):
                continue
            P2 = np.hstack([R, t[:, None]])
            X = triangulate_points(P1, P2, x1n[good], x2n[good])
            errs = [np.linalg.norm(self.cam(img).project(X @ P[:, :3].T + P[:, 3]) - xs, axis=1)
                    for img, P, xs in ((i, P1, x1[good]), (j, P2, x2[good]))]
            err = np.median(np.concatenate(errs))
            if err < best_err:
                best_err, best_s = err, s
        for c in self.cameras:
            c.fx = c.fy = base_f[c.cam_id] * best_s
            c.source += '+search'
        self._log(f"[focal search] pair ({i},{j}): scale {best_s:.3f} "
                  f"(median reproj {best_err:.2f}px) -> "
                  + ', '.join(f'cam{c.cam_id} f={c.f:.1f}' for c in self.cameras[:4]))
        for (a, b), dd in pair_data.items():
            dd['E'] = geo.essential_from_fundamental(dd['F'], self.K(a), self.K(b))

    def run(self, pair_data, progress_fn=None):
        T = self.timings
        t0 = time.perf_counter()
        if any(c.refine_focal and c.source == 'heuristic' for c in self.cameras):
            self._focal_search(pair_data)
            T['focal_search'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cands = self.init_candidates(pair_data)
        if not cands:
            raise RuntimeError("no valid initialization pair")
        T['ba'] = T['register'] = T['triangulate'] = T['filter'] = 0.0
        best_state = None
        for attempt, (_, i, j, R, t) in enumerate(cands[:self.max_init_attempts]):
            self.poses, self.points.has[:] = {}, False
            self.registration, self.error_history, self.ba_log = {}, [], []
            both = np.intersect1d(self.obs_tid[self.obs_img == i],
                                  self.obs_tid[self.obs_img == j])
            self.initialize(i, j, R, t, both, relaxed=True)
            self.registration[i] = dict(n_corr=0, n_inliers=0, registered=True, order=0,
                                        init=True)
            self.registration[j] = dict(n_corr=int(len(both)), n_inliers=int(len(both)),
                                        registered=True, order=1, init=True)
            self.error_history.append(('pre-BA', self.mean_reprojection_error()))
            t1 = time.perf_counter()
            self.run_ba('init')
            T['ba'] += time.perf_counter() - t1
            self.filter_points()
            n = len(self.points)
            self._log(f"[init] pair ({i},{j}) after BA + filtering: {n} points")
            if best_state is None or n > best_state[0]:
                best_state = (n, i, j, dict(self.poses), self.points.has.copy(),
                              self.points.X.copy(), dict(self.registration),
                              list(self.error_history), list(self.ba_log),
                              [(c.fx, c.fy, c.cx, c.cy, c.dist.copy()) for c in self.cameras])
            if n >= self.min_init_points:
                break
            self._log(f"[init] too few points (< {self.min_init_points}); trying next pair")
        n, i, j, poses, has, X, reg, hist, bal, intr = best_state
        self.poses, self.registration = poses, reg
        self.points.has[:], self.points.X[:] = has, X
        self.error_history, self.ba_log = hist, bal
        for c, (fx, fy, cx, cy, d) in zip(self.cameras, intr):
            c.fx, c.fy, c.cx, c.cy, c.dist = fx, fy, cx, cy, d
        self.init_pair = (i, j)
        T['init'] = time.perf_counter() - t0 - T['ba']

        def timed(key, fn, *a):
            t1 = time.perf_counter()
            out = fn(*a)
            T[key] += time.perf_counter() - t1
            return out

        while True:
            img, n = self.next_image()
            if img is None or n < 6:
                break
            if not timed('register', self.register_image, img):
                self._blocked[img] = True
                continue
            timed('triangulate', self._triangulate_tracks,
                  self.obs_tid[self.obs_img == img])
            timed('ba', self.run_ba, f'img{img}')
            timed('filter', self.filter_points)
            if progress_fn:
                progress_fn(self)
        timed('filter', self.filter_points)
        timed('ba', self.run_ba, 'final')
        return self


def verify_pairs(keypoints, pair_matches, cameras, cam_of_image,
                 thresh_px=1.5, min_inliers=30, verbose=True, log_fn=print):
    """F-RANSAC per pair, E = K_j^T F K_i."""
    pair_data, filtered = {}, {}
    for (i, j), m in sorted(pair_matches.items()):
        if len(m) < 8:
            continue
        x1 = keypoints[i][m[:, 0]]
        x2 = keypoints[j][m[:, 1]]
        ci, cj = cameras[cam_of_image[i]], cameras[cam_of_image[j]]
        try:
            F, inl = geo.ransac_fundamental(ci.undistort_pixels(x1), cj.undistort_pixels(x2),
                                            thresh=thresh_px,
                                            rng=np.random.default_rng(i * 1000 + j))
        except RuntimeError:
            continue
        if inl.sum() < min_inliers:
            continue
        E = geo.essential_from_fundamental(F, cameras[cam_of_image[i]].K,
                                           cameras[cam_of_image[j]].K)
        pair_data[(i, j)] = dict(x1=x1, x2=x2, matches=m, inliers=inl, F=F, E=E)
        filtered[(i, j)] = m[inl]
        if verbose:
            log_fn(f"[verify] pair ({i},{j}): {inl.sum()}/{len(m)} inliers")
    return pair_data, filtered
