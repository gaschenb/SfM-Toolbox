"""End-to-end pipeline driver shared by the CLI and the GUI."""
import time
from pathlib import Path

import numpy as np
import cv2

from sfmtoolbox.core.features import get_extractor, match_features, build_tracks
from sfmtoolbox.core.reconstruction import Reconstruction, verify_pairs
from sfmtoolbox.core.camera import build_cameras, exif_focal_spec
from sfmtoolbox.core.distortion import undistort_maps
from sfmtoolbox.core.pairs import select_pairs, expand_transitive
from sfmtoolbox.core.analytics import compute_analytics, write_analytics, format_summary
from sfmtoolbox.core.visualize import (plot_reprojection_errors, plot_scene, export_ply,
                                       export_poses)

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)


def load_images(paths, downsample=1.0):
    """downsample: linear reduction factor (1 = full res). Returns
    (grays, colors, names, sizes, scales)."""
    grays, colors, names, sizes, scales = [], [], [], [], []
    s = 1.0 / max(float(downsample), 1.0)
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        colors.append(img)
        names.append(p.name)
        sizes.append((img.shape[1], img.shape[0]))
        scales.append(s)
    return grays, colors, names, sizes, scales


def make_cameras(paths, sizes, scales, camera_mode='shared', spec=None,
                 specs=None, use_exif=True, log=print):
    """Pixel focals in specs refer to the original images and are rescaled by the
    load scale; mm specs are resolution-independent via sensor_res_x."""
    spec = dict(spec or {})
    specs = [dict(s) if s else None for s in (specs or [None] * len(paths))]
    for i, (p, s) in enumerate(zip(paths, scales)):
        eff = dict(specs[i] if specs[i] is not None else spec)
        if eff.get('focal_px'):
            eff['focal_px'] = eff['focal_px'] * s
        if use_exif and not any(eff.get(k) for k in ('focal_px', 'focal_mm', 'focal_35mm')):
            eff.update(exif_focal_spec(p))
        specs[i] = eff
    cameras, cam_of_image = build_cameras(sizes, mode=camera_mode, specs=specs,
                                          labels=[p.name for p in paths])
    for c in cameras:
        log(f"[camera {c.cam_id}] {c.label or 'shared'}: "
            f"f={c.f:.1f}px ({c.source}), pp=({c.cx:.1f},{c.cy:.1f}), "
            f"refine_focal={c.refine_focal}, refine_pp={c.refine_principal}")
    return cameras, cam_of_image


def run_pipeline(image_folder, out_dir=None, camera_mode='shared', spec=None,
                 specs=None, feature='sift', downsample=1.0, ransac_px=2.0,
                 min_tri_angle_deg=1.5, use_exif=True, log=print,
                 progress_fn=None, n_features=8000, matcher='bf', ratio=0.8,
                 feature_params=None, analytics=True, pairs='exhaustive', pairs_k=10,
                 seq_window=5, expand_transitive_iters=0, undistort_out=False):
    """Full SfM. Returns (recon, names, colors_img); recon.analytics holds the
    analytics dict. Writes outputs if out_dir is given."""
    T = {}
    t0 = time.perf_counter()
    paths = list_images(image_folder)
    if len(paths) < 2:
        raise RuntimeError("need at least 2 images")
    grays, colors_img, names, sizes, scales = load_images(paths, downsample)
    cameras, cam_of_image = make_cameras(paths, sizes, scales, camera_mode,
                                         spec, specs, use_exif, log)
    T['load'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ext = get_extractor(feature, n_features=n_features, **(feature_params or {}))
    log(f"[features] {ext.name.upper()} on {len(paths)} images")
    kps, descs = [], []
    for g, nm in zip(grays, names):
        p_, d_ = ext.detect(g)
        kps.append(p_)
        descs.append(d_)
        log(f"  {nm}: {len(p_)} features")
    T['features'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cand, pair_info = select_pairs(pairs, len(paths), descs, ext.norm, paths=paths,
                                   k=pairs_k, window=seq_window, log=log)
    pair_info.pop('similarity', None)
    T['pair_selection'] = time.perf_counter() - t0
    log(f"[pairs] {pairs}: {len(cand)}/{pair_info['n_exhaustive']} candidate pairs")

    def match_and_verify(cands):
        t1 = time.perf_counter()
        pm = {}
        for i, j in cands:
            m = match_features(descs[i], descs[j], ext.norm, ratio=ratio, matcher=matcher)
            if len(m) >= 8:
                pm[(i, j)] = m
        T['matching'] = T.get('matching', 0.0) + time.perf_counter() - t1
        t1 = time.perf_counter()
        pd, flt = verify_pairs(kps, pm, cameras, cam_of_image,
                               thresh_px=ransac_px * 0.75, log_fn=log)
        T['verification'] = T.get('verification', 0.0) + time.perf_counter() - t1
        return pm, pd, flt

    log(f"[matching] {len(cand)} pairs ({matcher}, {ext.norm})")
    pair_matches, pair_data, filtered = match_and_verify(cand)
    tried = set(cand)
    pair_info['expansion_added'] = []
    for it in range(expand_transitive_iters):
        new = expand_transitive(list(pair_data), tried, len(paths))
        if not new:
            break
        tried |= set(new)
        pm, pd, flt = match_and_verify(new)
        pair_matches.update(pm)
        pair_data.update(pd)
        filtered.update(flt)
        pair_info['expansion_added'].append(dict(tried=len(new), verified=len(pd)))
        log(f"[pairs] transitive expansion {it + 1}: {len(pd)}/{len(new)} new pairs verified")
    pair_matches = dict(sorted(pair_matches.items()))
    filtered = dict(sorted(filtered.items()))
    pair_info.update(n_tried=len(tried), n_matched=len(pair_matches), n_verified=len(pair_data))
    if not pair_data:
        raise RuntimeError("no geometrically consistent pairs found")
    t0 = time.perf_counter()
    tracks = build_tracks(filtered, len(paths))
    T['tracks'] = time.perf_counter() - t0
    log(f"[tracks] {len(tracks)} tracks")

    t0 = time.perf_counter()
    recon = Reconstruction(kps, tracks, cameras, cam_of_image, ransac_px=ransac_px,
                           min_tri_angle_deg=min_tri_angle_deg, log_fn=log)
    recon.run(pair_data, progress_fn=progress_fn)
    T['reconstruction'] = time.perf_counter() - t0
    T['total'] = sum(T.values())

    recon.analytics = None
    if analytics:
        recon.analytics = compute_analytics(
            recon, names, timings=T, n_features=[len(k) for k in kps],
            pair_matches=pair_matches, pair_data=pair_data, pair_info=pair_info)
        log('[analytics]\n' + format_summary(recon.analytics))
    if out_dir is not None:
        write_outputs(recon, names, colors_img, out_dir, log)
        if undistort_out:
            write_undistorted(recon, names, colors_img, Path(out_dir) / 'undistorted', log)
    return recon, names, colors_img


def write_undistorted(recon, names, colors_img, out_dir, log=print):
    """Undistorted (pinhole, same K) copies of all images using the refined models."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    maps = {}
    for i, (nm, img) in enumerate(zip(names, colors_img)):
        cam = recon.cam(i)
        if cam.cam_id not in maps:
            maps[cam.cam_id] = undistort_maps(cam)
        mx, my = maps[cam.cam_id]
        cv2.imwrite(str(out / nm), cv2.remap(img, mx, my, cv2.INTER_LINEAR))
    log(f"[output] wrote {len(names)} undistorted images to {out}/")


def point_colors(recon, colors_img):
    """{tid: (r,g,b)} sampled at the first observation of each track."""
    tids = recon.points.ids()
    first = np.searchsorted(recon.obs_tid, tids)
    img, uv = recon.obs_img[first], np.round(recon.obs_uv[first]).astype(int)
    rgb = np.zeros((len(tids), 3), int)
    for i in np.unique(img):
        m = img == i
        h, w = colors_img[i].shape[:2]
        bgr = colors_img[i][np.clip(uv[m, 1], 0, h - 1), np.clip(uv[m, 0], 0, w - 1)]
        rgb[m] = bgr[:, ::-1]
    return {int(t): tuple(int(c) for c in v) for t, v in zip(tids, rgb)}


def write_outputs(recon, names, colors_img, out_dir, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_ply(recon, out / 'points.ply', point_colors(recon, colors_img))
    export_poses(recon, names, out / 'poses.json')
    plot_reprojection_errors(recon, names, out / 'reprojection_errors.png')
    plot_scene(recon, out / 'scene.png')
    files = 'points.ply, poses.json, reprojection_errors.png, scene.png'
    if getattr(recon, 'analytics', None):
        write_analytics(recon.analytics, out)
        files += ', analytics.{json,png}, analytics_summary.txt, per_image.csv'
    log(f"[output] wrote {files} to {out}/")
