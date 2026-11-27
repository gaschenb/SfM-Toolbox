"""Render views of a textured corner (3 orthogonal planes) with known
cameras via per-plane homographies -> real perspective images for
trying the pipeline on known ground truth."""
import numpy as np
import cv2
from pathlib import Path
from sfmtoolbox.core.pose import rodrigues
from sfmtoolbox.core.triangulation import project

rng = np.random.default_rng(7)
W, H = 960, 720
F = 900.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])

def texture(size=512):
    img = np.full((size, size, 3), 255, np.uint8)
    for _ in range(400):
        c = rng.integers(0, 255, 3).tolist()
        p = rng.integers(0, size, 2)
        r = int(rng.integers(4, 25))
        if rng.random() < 0.5:
            cv2.circle(img, tuple(p), r, c, -1)
        else:
            cv2.rectangle(img, tuple(p), tuple(p + r), c, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

S = 3.2
planes = [
    (np.array([-1.6, -1.6, 6.5]), np.array([S, 0, 0.]), np.array([0, S, 0.])),
    (np.array([-1.6, -1.6, 6.5]), np.array([0, 0, -S]), np.array([0, S, 0.])),
    (np.array([-1.6, -1.6, 6.5]), np.array([S, 0, 0.]), np.array([0, 0, -S])),
]
textures = [texture() for _ in planes]

def render(R, t):
    out = np.zeros((H, W, 3), np.uint8)
    depth = np.full((H, W), np.inf)
    for (O, U, V), tex in zip(planes, textures):
        s = tex.shape[0]
        corners_tex = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float32)
        corners_3d = np.array([O, O + U, O + U + V, O + V])
        uv, z = project(corners_3d, R, t, K)
        if np.any(z <= 0.1):
            continue
        Hm, _ = cv2.findHomography(corners_tex, uv.astype(np.float32))
        warped = cv2.warpPerspective(tex, Hm, (W, H))
        mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), Hm, (W, H)) > 0
        n = np.cross(U, V)
        d = -n @ O
        ys, xs = np.where(mask)
        rays = np.linalg.inv(K) @ np.vstack([xs, ys, np.ones_like(xs)])
        rays_w = R.T @ rays
        C = -R.T @ t
        denom = n @ rays_w
        lam = -(n @ C + d) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        zc = (R @ (C[:, None] + lam * rays_w) + t[:, None])[2]
        closer = zc < depth[ys, xs]
        sel = np.where(closer)[0]
        out[ys[sel], xs[sel]] = warped[ys[sel], xs[sel]]
        depth[ys[sel], xs[sel]] = zc[sel]
    return out

outdir = Path('synthetic_images')
outdir.mkdir(exist_ok=True)
gt = {}
n_views = 8
for i in range(n_views):
    ang = -0.25 + 0.5 * i / (n_views - 1)
    R = rodrigues(np.array([0.03 * np.sin(3 * i), ang, 0.0]))
    C = np.array([3.0 * np.sin(ang), 0.25 * np.sin(2 * i),
                  0.6 * (1 - np.cos(ang))])
    t = -R @ C
    img = render(R, t)
    cv2.imwrite(str(outdir / f'view_{i:02d}.png'), img)
    gt[f'view_{i:02d}.png'] = (R, t)
np.save('gt_poses.npy', {k: (R.tolist(), t.tolist()) for k, (R, t) in gt.items()},
        allow_pickle=True)
print(f"rendered {n_views} views, f={F}px")
