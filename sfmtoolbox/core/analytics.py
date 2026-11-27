"""Reconstruction analytics: statistics, text summary, JSON/CSV dump and plots."""
import csv
import json
from pathlib import Path

import numpy as np
from sfmtoolbox.core.distortion import PARAM_NAMES, max_displacement_px

_Q = (50, 90, 95, 99)


def _stats(a):
    a = np.asarray(a, float)
    if not len(a):
        return dict(n=0)
    out = dict(n=int(len(a)), mean=float(a.mean()), std=float(a.std()),
               min=float(a.min()), max=float(a.max()), rms=float(np.sqrt((a ** 2).mean())))
    out.update({f'p{q}': float(v) for q, v in zip(_Q, np.percentile(a, _Q))})
    return out


def _grid_coverage(uv, w, h, n=10):
    if not len(uv):
        return 0.0
    gx = np.clip((uv[:, 0] / w * n).astype(int), 0, n - 1)
    gy = np.clip((uv[:, 1] / h * n).astype(int), 0, n - 1)
    return len(np.unique(gy * n + gx)) / n ** 2


def compute_analytics(recon, names=None, timings=None, n_features=None,
                      pair_matches=None, pair_data=None, pair_info=None, grid=10):
    """Collect statistics from a finished Reconstruction. All inputs besides
    `recon` are optional; sections without data are omitted."""
    n_img = recon.n_images
    names = list(names) if names is not None else [str(i) for i in range(n_img)]
    sel, err, depth = recon.observation_errors()
    o_img, o_tid = recon.obs_img[sel], recon.obs_tid[sel]
    tids = recon.points.ids()
    X = recon.points.X[tids]
    reg = recon._registered()

    track_len_all = np.fromiter((len(t) for t in recon.tracks), np.int64, len(recon.tracks))
    pt_len = np.bincount(np.searchsorted(tids, o_tid), minlength=len(tids)) if len(tids) \
        else np.zeros(0, int)

    R, t, _ = recon._pose_arrays()
    C = -np.einsum('nji,nj->ni', R, t)
    tri_ang = np.zeros(len(tids))
    if len(tids):
        seg = np.searchsorted(tids, o_tid)
        counts = np.bincount(seg, minlength=len(tids))
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        pos = np.arange(len(seg)) - starts[seg]
        rays = X[seg] - C[o_img]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
        k = int(counts.max())
        Rp = np.zeros((len(tids), k, 3))
        Rp[seg, pos] = rays
        valid = np.arange(k)[None] < counts[:, None]
        cos = np.einsum('nai,nbi->nab', Rp, Rp)
        cos[~(valid[:, :, None] & valid[:, None, :])] = 1.0
        tri_ang = np.degrees(np.arccos(np.clip(cos.reshape(len(tids), -1).min(1), -1, 1)))

    matches_per_img = np.zeros(n_img, int)
    verified_per_img = np.zeros(n_img, int)
    pairs = []
    if pair_matches is not None:
        for (i, j), m in pair_matches.items():
            matches_per_img[[i, j]] += len(m)
    if pair_data is not None:
        for (i, j), d in pair_data.items():
            ninl = int(d['inliers'].sum())
            verified_per_img[[i, j]] += ninl
    if pair_matches is not None:
        for (i, j), m in sorted(pair_matches.items()):
            ninl = int(pair_data[(i, j)]['inliers'].sum()) if pair_data and (i, j) in pair_data else 0
            pairs.append(dict(i=int(i), j=int(j), matches=int(len(m)), inliers=ninl,
                              inlier_ratio=ninl / max(len(m), 1)))

    per_image = []
    for i in range(n_img):
        cam = recon.cam(i)
        m = o_img == i
        regi = recon.registration.get(i, {})
        e = err[m]
        row = dict(image=names[i], index=i, registered=bool(reg[i]),
                   n_features=int(n_features[i]) if n_features is not None else None,
                   n_matches=int(matches_per_img[i]) if pair_matches is not None else None,
                   n_verified=int(verified_per_img[i]) if pair_data is not None else None,
                   init=bool(regi.get('init', False)),
                   pnp_correspondences=None if regi.get('init') else regi.get('n_corr'),
                   pnp_inliers=None if regi.get('init') else regi.get('n_inliers'),
                   pnp_inlier_ratio=(regi['n_inliers'] / regi['n_corr']
                                     if regi.get('n_corr') and not regi.get('init') else None),
                   n_observations=int(m.sum()),
                   reproj_mean=float(e.mean()) if len(e) else None,
                   reproj_median=float(np.median(e)) if len(e) else None,
                   coverage=_grid_coverage(recon.obs_uv[sel][m], cam.width, cam.height, grid),
                   focal_px=float(cam.f),
                   center=C[i].tolist() if reg[i] else None)
        per_image.append(row)

    pc = dict(n_points=int(len(tids)))
    if len(tids) >= 2:
        from scipy.spatial import cKDTree
        lo, hi = np.percentile(X, [2, 98], axis=0)
        ext = hi - lo
        vol = float(np.prod(np.maximum(ext, 1e-12)))
        nn = cKDTree(X).query(X, k=2)[0][:, 1]
        inside = np.all((X >= lo) & (X <= hi), axis=1).sum()
        pc.update(centroid=X.mean(0).tolist(), robust_bbox_min=lo.tolist(),
                  robust_bbox_max=hi.tolist(), robust_extent=ext.tolist(),
                  density_robust_bbox=float(inside / vol), nn_distance=_stats(nn),
                  depth=_stats(depth), triangulation_angle_deg=_stats(tri_ang))
        if reg.sum() >= 2:
            pc['baseline_span'] = float(np.ptp(C[reg], axis=0).max())

    summary = dict(
        n_images=n_img, n_registered=int(reg.sum()),
        registration_rate=float(reg.mean()) if n_img else 0.0,
        n_tracks=int(len(recon.tracks)), n_points=int(len(tids)),
        track_utilization=float(len(tids) / max(len(recon.tracks), 1)),
        n_observations=int(len(err)),
        mean_track_length=float(pt_len.mean()) if len(pt_len) else 0.0,
        reproj_mean=float(err.mean()) if len(err) else None,
        reproj_median=float(np.median(err)) if len(err) else None,
        reproj_rms=float(np.sqrt((err ** 2).mean())) if len(err) else None,
        mean_coverage=float(np.mean([r['coverage'] for r in per_image if r['registered']]))
        if reg.any() else 0.0)
    out = dict(
        summary=summary,
        reprojection_error=_stats(err),
        track_length=dict(
            all=_stats(track_len_all),
            triangulated=_stats(pt_len),
            histogram_triangulated={int(k): int(v) for k, v in
                                    zip(*np.unique(pt_len, return_counts=True))}),
        per_image=per_image, pairs=pairs, point_cloud=pc,
        bundle_adjustment=list(recon.ba_log),
        error_history=[[lab, float(v)] for lab, v in recon.error_history],
        cameras=[dict(cam_id=c.cam_id, label=c.label, fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy,
                      source=c.source, model=c.model,
                      distortion=dict(zip(PARAM_NAMES[c.model], np.asarray(c.dist).tolist())),
                      max_distortion_px=max_displacement_px(c)) for c in recon.cameras],
        pair_selection=dict(pair_info or {}),
        timings={**(timings or {}), **{f'reconstruct.{k}': v
                                       for k, v in recon.timings.items()}})
    out['_raw'] = dict(errors=err, obs_img=o_img, tri_angle=tri_ang, track_len=pt_len,
                       nn=nn if len(tids) >= 2 else np.zeros(0))
    return out


def format_summary(a):
    s, e, tl, pc = a['summary'], a['reprojection_error'], a['track_length'], a['point_cloud']
    lines = [
        f"images      {s['n_registered']}/{s['n_images']} registered "
        f"({100 * s['registration_rate']:.0f}%)",
        f"points      {s['n_points']} from {s['n_tracks']} tracks "
        f"({100 * s['track_utilization']:.0f}% used), {s['n_observations']} observations",
    ]
    if e.get('n'):
        lines.append(f"reproj px   mean {e['mean']:.3f}  median {e['p50']:.3f}  rms {e['rms']:.3f}  "
                     f"p95 {e['p95']:.3f}  max {e['max']:.3f}")
    if tl['triangulated'].get('n'):
        t = tl['triangulated']
        lines.append(f"track len   mean {t['mean']:.2f}  median {t['p50']:.0f}  max {t['max']:.0f}")
    if 'triangulation_angle_deg' in pc:
        ta, nn = pc['triangulation_angle_deg'], pc['nn_distance']
        lines.append(f"tri angle   median {ta['p50']:.2f} deg  p90 {ta['p90']:.2f} deg")
        lines.append(f"cloud       extent {np.round(pc['robust_extent'], 3).tolist()}  "
                     f"NN median {nn['p50']:.4g}  density {pc['density_robust_bbox']:.4g}/unit^3")
    ps = a.get('pair_selection') or {}
    if ps.get('method'):
        exp = sum(e['verified'] for e in ps.get('expansion_added', []))
        lines.append(f"pairs       {ps['method']}: {ps.get('n_candidates', 0)}/{ps['n_exhaustive']} "
                     f"candidates, {ps.get('n_verified', 0)} verified"
                     + (f" (+{exp} via transitive expansion)" if exp else ''))
    for c in a['cameras']:
        if c['model'] != 'pinhole':
            ks = ' '.join(f"{k}={v:.4g}" for k, v in c['distortion'].items())
            lines.append(f"lens        cam{c['cam_id']} {c['model']}: {ks}  "
                         f"(max {c['max_distortion_px']:.1f}px at border)")
    lines.append(f"coverage    mean {100 * s['mean_coverage']:.0f}% of image grid cells")
    tm = {k: v for k, v in a['timings'].items() if '.' not in k}
    if tm:
        lines.append("time s      " + "  ".join(f"{k} {v:.2f}" for k, v in tm.items()))
    hdr = f"{'image':<22}{'reg':>4}{'feat':>7}{'match':>7}{'verif':>7}{'pnp':>10}{'obs':>6}" \
          f"{'err':>7}{'cov':>6}"
    lines += ['', hdr]
    for r in a['per_image']:
        pnp = ('init' if r.get('init') else f"{r['pnp_inliers']}/{r['pnp_correspondences']}"
               if r['pnp_correspondences'] else '-')
        err_s = f"{r['reproj_mean']:.3f}" if r['reproj_mean'] is not None else '-'
        lines.append(f"{r['image'][:21]:<22}{'y' if r['registered'] else 'n':>4}"
                     f"{r['n_features'] if r['n_features'] is not None else '-':>7}"
                     f"{r['n_matches'] if r['n_matches'] is not None else '-':>7}"
                     f"{r['n_verified'] if r['n_verified'] is not None else '-':>7}"
                     f"{pnp:>10}{r['n_observations']:>6}"
                     f"{err_s:>7}"
                     f"{100 * r['coverage']:>5.0f}%")
    return '\n'.join(lines)


_INK, _INK2, _GRID = '#191919', '#666666', '#e5e5e5'
_S = ['#257de0', '#f47230', '#00b894']


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc='left', fontsize=10, color=_INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=_INK2, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=_INK2, fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=7)
    ax.grid(True, color=_GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_analytics(a, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw = a['_raw']
    pi = a['per_image']
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    ax = axes.ravel()

    e = raw['errors']
    if len(e):
        hi = np.percentile(e, 99.5)
        ax[0].hist(np.clip(e, 0, hi), bins=60, color=_S[0], edgecolor='white', lw=0.5)
        for v, ls, lab in ((np.median(e), '-', 'median'), (e.mean(), '--', 'mean')):
            ax[0].axvline(v, color=_INK, ls=ls, lw=1, label=f'{lab} {v:.2f}px')
        ax[0].legend(fontsize=7, frameon=False)
        _style(ax[0], 'Reprojection error', 'px', 'observations')
        es = np.sort(e)
        ax[1].plot(es, np.arange(1, len(es) + 1) / len(es), color=_S[0], lw=2)
        for q in (0.5, 0.9, 0.99):
            v = es[min(int(q * len(es)), len(es) - 1)]
            ax[1].plot([v], [q], 'o', ms=5, color=_S[0], mec='white')
            ax[1].annotate(f'p{int(q * 100)} {v:.2f}', (v, q), fontsize=7, color=_INK,
                           xytext=(5, -10), textcoords='offset points')
        ax[1].set_xlim(0, np.percentile(e, 99.9))
        _style(ax[1], 'Reprojection error CDF', 'px', 'fraction ≤ x')

    tl = raw['track_len']
    if len(tl):
        ks, cs = np.unique(tl, return_counts=True)
        ax[2].bar(ks, cs, color=_S[0], width=0.8, edgecolor='white')
        _style(ax[2], 'Track length (triangulated points)', 'observing images', 'points')
        ax[2].set_xticks(ks)

    x = np.arange(len(pi))
    labels = [r['image'] for r in pi]
    series = [('features', [r['n_features'] or 0 for r in pi]),
              ('verified matches', [r['n_verified'] or 0 for r in pi]),
              ('PnP inliers', [r['pnp_inliers'] or 0 for r in pi])]
    wdt = 0.27
    for k, (lab, vals) in enumerate(series):
        ax[3].bar(x + (k - 1) * wdt, vals, wdt * 0.92, color=_S[k], label=lab)
    ax[3].set_yscale('symlog', linthresh=10)
    ax[3].legend(fontsize=7, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    _style(ax[3], 'Per-image features / matches / inliers\n', None, 'count (log)')
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(labels, rotation=60, ha='right', fontsize=6)

    me = [r['reproj_mean'] or 0 for r in pi]
    ax[4].bar(x, me, color=_S[0], width=0.7)
    _style(ax[4], 'Per-image mean reprojection error', None, 'px')
    ax[4].set_xticks(x)
    ax[4].set_xticklabels(labels, rotation=60, ha='right', fontsize=6)

    ax[5].bar(x, [100 * r['coverage'] for r in pi], color=_S[0], width=0.7)
    ax[5].set_ylim(0, 100)
    _style(ax[5], 'Per-image observation coverage (10×10 grid)', None, '% cells')
    ax[5].set_xticks(x)
    ax[5].set_xticklabels(labels, rotation=60, ha='right', fontsize=6)

    n = len(pi)
    M = np.full((n, n), np.nan)
    for p in a['pairs']:
        M[p['i'], p['j']] = M[p['j'], p['i']] = p['inliers']
    im = ax[6].imshow(M, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax[6], fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    _style(ax[6], 'Verified inliers per image pair')
    ax[6].grid(False)
    ax[6].set_xticks(range(n))
    ax[6].set_yticks(range(n))
    ax[6].tick_params(labelsize=6)

    ta = raw['tri_angle']
    if len(ta):
        ax[7].hist(np.clip(ta, 0, np.percentile(ta, 99.5)), bins=50, color=_S[0],
                   edgecolor='white', lw=0.5)
        _style(ax[7], 'Max triangulation angle per point', 'deg', 'points')

    tm = {k: v for k, v in a['timings'].items() if not k.startswith('total')}
    if tm:
        ks = list(tm)[::-1]
        ax[8].barh(ks, [tm[k] for k in ks], color=_S[0], height=0.6)
        for yi, k in enumerate(ks):
            ax[8].annotate(f'{tm[k]:.2f}s', (tm[k], yi), xytext=(3, -3),
                           textcoords='offset points', fontsize=7, color=_INK)
        _style(ax[8], 'Time per stage', 's')
        ax[8].tick_params(axis='y', labelsize=7)

    s = a['summary']
    fig.suptitle(f"{s['n_registered']}/{s['n_images']} images · {s['n_points']} points · "
                 f"mean reproj {s['reproj_mean'] or float('nan'):.3f}px · "
                 f"mean track {s['mean_track_length']:.2f}",
                 x=0.01, ha='left', fontsize=12, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k != '_raw'}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def write_analytics(a, out_dir, plots=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'analytics.json', 'w') as f:
        json.dump(_jsonable(a), f, indent=2)
    (out / 'analytics_summary.txt').write_text(format_summary(a) + '\n')
    if a['per_image']:
        with open(out / 'per_image.csv', 'w', newline='') as f:
            cols = [k for k in a['per_image'][0] if k != 'center']
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(a['per_image'])
    if plots:
        plot_analytics(a, out / 'analytics.png')
    return out
