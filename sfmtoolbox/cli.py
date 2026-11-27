"""Structure-from-Motion CLI.

Intrinsics (per shared camera), any one of:
  --focal-px F
  --focal-35mm F
  --focal-mm F --sensor-width-mm W [--sensor-res-x N]
  --focal-mm F --pixel-size-mm P [--sensor-res-x N]
  (none)    EXIF if available, else 1.2*max(w,h)

Examples:
  sfm-toolbox --images imgs/ --out out/ --focal-mm 24 --sensor-width-mm 23.5
  sfm-toolbox --images imgs/ --out out/ --feature akaze --matcher flann
  sfm-toolbox --images imgs/ --out out/ --cameras per-image --refine focal
  sfm-toolbox --images imgs/ --out out/ --camera-model radial2 --refine-dist --undistort
  sfm-toolbox --images imgs/ --out out/ --pairs vocab --pairs-k 15 --expand 1
  sfm-toolbox --list-features
"""
import argparse
import sys

from sfmtoolbox.core.features import available_extractors, _REGISTRY
from sfmtoolbox.core.distortion import MODELS
from sfmtoolbox.core.pairs import PAIR_METHODS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images')
    ap.add_argument('--out', default='sfm_output')
    ap.add_argument('--cameras', choices=['shared', 'per-image'], default='shared')
    ap.add_argument('--focal-px', type=float)
    ap.add_argument('--focal-35mm', type=float)
    ap.add_argument('--focal-mm', type=float)
    ap.add_argument('--sensor-width-mm', type=float)
    ap.add_argument('--pixel-size-mm', type=float)
    ap.add_argument('--sensor-res-x', type=int,
                    help='native sensor width (px) for the mm spec; needed if images are resized')
    ap.add_argument('--refine', choices=['none', 'focal', 'full'], default='none',
                    help='refine intrinsics in BA (full = focal + principal point)')
    ap.add_argument('--no-exif', action='store_true')
    ap.add_argument('--camera-model', choices=list(MODELS), default='pinhole',
                    help='lens distortion model')
    ap.add_argument('--dist', type=lambda v: [float(x) for x in v.split(',')], default=None,
                    help='initial distortion coefficients, comma-separated, in model order')
    ap.add_argument('--refine-dist', action='store_true',
                    help='refine distortion coefficients in BA')
    ap.add_argument('--undistort', action='store_true',
                    help='also write undistorted images to OUT/undistorted')
    ap.add_argument('--pairs', choices=PAIR_METHODS, default='exhaustive',
                    help='candidate pair selection')
    ap.add_argument('--pairs-k', type=int, default=10,
                    help='neighbours per image for vocab/vlad/gps')
    ap.add_argument('--seq-window', type=int, default=5, help='window for sequential')
    ap.add_argument('--expand', type=int, default=0,
                    help='rounds of transitive pair expansion after verification')
    ap.add_argument('--feature', default='sift', choices=available_extractors(),
                    help='feature extractor')
    ap.add_argument('--n-features', type=int, default=8000)
    ap.add_argument('--matcher', choices=['bf', 'flann'], default='bf')
    ap.add_argument('--ratio', type=float, default=0.8, help='Lowe ratio')
    ap.add_argument('--downsample', type=float, default=1.0,
                    help='linear downsample factor (1 = full resolution)')
    ap.add_argument('--ransac-px', type=float, default=2.0)
    ap.add_argument('--min-tri-angle', type=float, default=1.5)
    ap.add_argument('--no-analytics', action='store_true')
    ap.add_argument('--list-features', action='store_true')
    a = ap.parse_args(argv)

    if a.list_features:
        for k in available_extractors():
            c = _REGISTRY[k]
            print(f"{k:<11}{c.norm:<9}{'ok' if c.available() else 'missing: ' + c.requires}")
        return
    if not a.images:
        ap.error('--images is required')

    from sfmtoolbox.core.pipeline import run_pipeline
    spec = {k: v for k, v in dict(
        focal_px=a.focal_px, focal_35mm=a.focal_35mm, focal_mm=a.focal_mm,
        sensor_width_mm=a.sensor_width_mm, pixel_size_mm=a.pixel_size_mm,
        sensor_res_x=a.sensor_res_x).items() if v}
    spec['refine_focal'] = a.refine in ('focal', 'full')
    spec['refine_principal'] = a.refine == 'full'
    spec['model'] = a.camera_model
    spec['refine_dist'] = a.refine_dist and a.camera_model != 'pinhole'
    if a.dist:
        spec['dist'] = a.dist

    try:
        recon, names, _ = run_pipeline(
            a.images, out_dir=a.out, camera_mode=a.cameras.replace('-', '_'), spec=spec,
            feature=a.feature, n_features=a.n_features, matcher=a.matcher, ratio=a.ratio,
            downsample=a.downsample, ransac_px=a.ransac_px,
            min_tri_angle_deg=a.min_tri_angle, use_exif=not a.no_exif,
            analytics=not a.no_analytics, pairs=a.pairs, pairs_k=a.pairs_k,
            seq_window=a.seq_window, expand_transitive_iters=a.expand,
            undistort_out=a.undistort)
    except (RuntimeError, ImportError, ValueError) as e:
        sys.exit(str(e))
    print(f"\nDone. {len(recon.poses)}/{len(names)} cameras registered, "
          f"{len(recon.points)} points, "
          f"mean reproj {recon.mean_reprojection_error():.3f}px")


if __name__ == '__main__':
    main()
