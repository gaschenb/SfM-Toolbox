"""Regenerate the README figures: python -m sfmtoolbox.examples.gallery docs/figures"""
import os
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sfmtoolbox.core.pipeline import run_pipeline
from sfmtoolbox.core.analytics import plot_analytics
from sfmtoolbox.core.camera import Camera
from sfmtoolbox.core.pairs import select_pairs, vocab_similarity
from sfmtoolbox.core.features import get_extractor

INK, INK2, GRID = '#191919', '#666666', '#e5e5e5'
S = ['#257de0', '#f47230', '#00b894', '#f2a600']
quiet = lambda *a: None  # noqa: E731


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7)


def synthetic(work):
    cwd = os.getcwd()
    os.chdir(work)
    try:
        runpy.run_module('sfmtoolbox.examples.make_synthetic_images', run_name='__main__')
    finally:
        os.chdir(cwd)
    return work / 'synthetic_images', np.load(work / 'gt_poses.npy', allow_pickle=True).item()


def distort_set(src, dst, cam):
    dst.mkdir(exist_ok=True)
    u, v = np.meshgrid(np.arange(cam.width, dtype=float), np.arange(cam.height, dtype=float))
    m = cam.undistort_pixels(np.c_[u.ravel(), v.ravel()]).reshape(cam.height, cam.width, 2)
    m = m.astype(np.float32)
    for f in sorted(src.glob('*.png')):
        cv2.imwrite(str(dst / f.name), cv2.remap(cv2.imread(str(f)), m[..., 0], m[..., 1],
                                                 cv2.INTER_LINEAR))


def fig_scene(recon, names, colors, gt, out):
    tids = recon.points.ids()
    X = recon.points.X[tids]
    first = np.searchsorted(recon.obs_tid, tids)
    img, uv = recon.obs_img[first], np.round(recon.obs_uv[first]).astype(int)
    rgb = np.array([colors[i][min(y, 719), min(x, 959)][::-1] for i, (x, y) in zip(img, uv)]) / 255
    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 2], -X[:, 1], s=1.2, c=rgb, depthshade=False)
    for i, (R, t) in recon.poses.items():
        C = -R.T @ t
        corners = np.array([[-.16, -.12, .3], [.16, -.12, .3], [.16, .12, .3], [-.16, .12, .3]])
        W = corners @ R + C
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)] + [(k, 4) for k in range(4)]:
            P = np.vstack([W, C])[[a, b]]
            ax.plot(P[:, 0], P[:, 2], -P[:, 1], color=S[1], lw=0.9)
    ax.view_init(elev=22, azim=-62)
    ax.set_axis_off()
    ax.set_title(f'{len(tids)} points, {len(recon.poses)} cameras', loc='left', fontsize=10)
    ax2 = fig.add_subplot(122)
    ims = sorted(recon.poses)
    R0e, R0g = recon.poses[ims[0]][0], np.array(gt[names[ims[0]]][0])
    err = []
    for i in ims:
        Re = recon.poses[i][0] @ R0e.T
        Rg = np.array(gt[names[i]][0]) @ R0g.T
        c = np.clip((np.trace(Re.T @ Rg) - 1) / 2, -1, 1)
        err.append(np.degrees(np.arccos(c)))
    errs = recon.reprojection_errors()
    x = np.arange(len(ims))
    ax2.bar(x - 0.2, [errs[i].mean() for i in ims], 0.38, color=S[0], label='mean reprojection error (px)')
    ax2.bar(x + 0.2, [100 * e for e in err], 0.38, color=S[1], label='rotation error vs ground truth (0.01°)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([names[i].replace('.png', '') for i in ims], rotation=45, ha='right')
    ax2.legend(fontsize=7, frameon=False, loc='upper left')
    ax2.grid(True, axis='y', color=GRID)
    ax2.set_axisbelow(True)
    _style(ax2, 'Per-view accuracy on the synthetic scene (f = 900 px)')
    fig.tight_layout()
    fig.savefig(out / 'scene.png', dpi=130)
    plt.close(fig)


def fig_distortion(clean_dir, dist_dir, work, out):
    rows = []
    for model in ('pinhole', 'radial1', 'radial2', 'brown'):
        r, _, _ = run_pipeline(dist_dir, spec={'focal_px': 900, 'model': model,
                                               'refine_dist': model != 'pinhole'},
                               use_exif=False, log=quiet, analytics=False,
                               out_dir=work / f'dist_{model}', undistort_out=(model == 'radial2'))
        rows.append((model, r.mean_reprojection_error(), np.asarray(r.cameras[0].dist)))
    a = cv2.imread(str(dist_dir / 'view_03.png'))[..., ::-1]
    b = cv2.imread(str(work / 'dist_radial2' / 'undistorted' / 'view_03.png'))[..., ::-1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for k, (im, title) in enumerate([(a, 'Input with Brown distortion (k1 = -0.18, k2 = 0.04)'),
                                     (b, 'Undistorted with refined radial2 model')]):
        ax[k].imshow(im)
        ax[k].set_title(title, loc='left', fontsize=10)
        ax[k].axis('off')
        h, w = im.shape[:2]
        for yy in np.linspace(0, h - 1, 7):
            ax[k].axhline(yy, color='white', lw=0.4, alpha=0.6)
        for xx in np.linspace(0, w - 1, 9):
            ax[k].axvline(xx, color='white', lw=0.4, alpha=0.6)
    labels = [m for m, _, _ in rows]
    vals = [e for _, e, _ in rows]
    ax[2].barh(labels[::-1], vals[::-1], color=[S[0]] * len(vals), height=0.6)
    for yi, (m, e, d) in enumerate(rows[::-1]):
        txt = f'{e:.3f}px' + (f'   k1={d[0]:.3f}' if len(d) else '')
        ax[2].annotate(txt, (e, yi), xytext=(4, -3), textcoords='offset points', fontsize=8)
    ax[2].set_xlim(0, max(vals) * 1.8)
    _style(ax[2], 'Mean reprojection error by lens model', 'px')
    fig.tight_layout()
    fig.savefig(out / 'distortion.png', dpi=130)
    plt.close(fig)


def fig_pairs(img_dir, out):
    paths = sorted(img_dir.glob('*.png'))
    ext = get_extractor('sift', n_features=4000)
    descs = [ext.detect(cv2.imread(str(p), 0))[1] for p in paths]
    n = len(paths)
    Ssim = vocab_similarity(descs, 'l2')
    sel = {m: set(select_pairs(m, n, descs, 'l2', k=3, window=2, loop=False)[0])
           for m in ('exhaustive', 'sequential', 'vocab', 'vlad')}
    stats = {}
    for m in ('exhaustive', 'sequential', 'vocab', 'hybrid'):
        r, _, _ = run_pipeline(img_dir, spec={'focal_px': 900}, use_exif=False, log=quiet,
                               pairs=m, pairs_k=3, seq_window=2,
                               expand_transitive_iters=1 if m != 'exhaustive' else 0)
        ps = r.analytics['pair_selection']
        stats[m] = (ps['n_candidates'], ps['n_tried'], ps['n_verified'], len(r.poses),
                    r.analytics['timings'].get('matching', 0))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    M = Ssim.copy()
    np.fill_diagonal(M, np.nan)
    im = ax[0].imshow(M, cmap='Blues')
    for (i, j) in sel['vocab']:
        for a, b in ((i, j), (j, i)):
            ax[0].add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False, ec=S[1], lw=1.6))
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    _style(ax[0], 'Vocabulary-tree similarity (boxes: k = 3 selection)')
    ax[0].set_xticks(range(n))
    ax[0].set_yticks(range(n))
    for k, m in enumerate(('sequential', 'vlad')):
        G = np.full((n, n), np.nan)
        for i, j in sel['exhaustive']:
            G[i, j] = G[j, i] = 0.15
        for i, j in sel[m]:
            G[i, j] = G[j, i] = 1
        ax[1].imshow(G, cmap='Blues', vmin=0, vmax=1) if k == 0 else None
    _style(ax[1], 'Sequential (window 2) candidate pairs')
    ax[1].set_xticks(range(n))
    ax[1].set_yticks(range(n))
    ms = list(stats)
    x = np.arange(len(ms))
    ax[2].bar(x - 0.27, [stats[m][0] for m in ms], 0.26, color=S[0], label='selected')
    ax[2].bar(x, [stats[m][1] for m in ms], 0.26, color=S[1], label='matched (after expansion)')
    ax[2].bar(x + 0.27, [stats[m][2] for m in ms], 0.26, color=S[2], label='verified')
    for xi, m in zip(x, ms):
        ax[2].annotate(f'{stats[m][3]}/{n} reg.', (xi, stats[m][1]), xytext=(0, 4),
                       textcoords='offset points', ha='center', fontsize=7, color=INK)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(ms)
    ax[2].legend(fontsize=7, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    _style(ax[2], 'Pairs per method (k = 3, +1 expansion round)\n', None, 'pairs')
    fig.tight_layout()
    fig.savefig(out / 'pairs.png', dpi=130)
    plt.close(fig)


def main(out='docs/figures'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        img_dir, gt = synthetic(work)
        recon, names, colors = run_pipeline(img_dir, out_dir=work / 'run', spec={'focal_px': 900},
                                            use_exif=False, log=quiet)
        fig_scene(recon, names, colors, gt, out)
        plot_analytics(recon.analytics, out / 'analytics.png')
        cam = Camera(0, 960, 720, 900, 900, 480, 360, model='brown',
                     dist=[-0.18, 0.04, 5e-4, -8e-4, 0])
        distort_set(img_dir, work / 'distorted', cam)
        fig_distortion(img_dir, work / 'distorted', work, out)
        fig_pairs(img_dir, out)
    print(f'wrote figures to {out}/')


if __name__ == '__main__':
    main(*sys.argv[1:2])
