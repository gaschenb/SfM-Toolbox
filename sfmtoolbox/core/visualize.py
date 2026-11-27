"""Outputs: reprojection-error plots, PLY point cloud, pose dump."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_reprojection_errors(recon, image_names, out_path):
    errs = recon.reprojection_errors()
    if not errs:
        return
    all_e = np.concatenate(list(errs.values()))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(np.clip(all_e, 0, 10), bins=60, color='steelblue')
    axes[0].axvline(np.mean(all_e), color='r', ls='--',
                    label=f'mean = {np.mean(all_e):.2f}px')
    axes[0].axvline(np.median(all_e), color='k', ls=':',
                    label=f'median = {np.median(all_e):.2f}px')
    axes[0].set(title='Reprojection error histogram', xlabel='error (px)',
                ylabel='count')
    axes[0].legend()

    imgs = sorted(errs)
    means = [errs[i].mean() for i in imgs]
    axes[1].bar(range(len(imgs)), means, color='steelblue')
    axes[1].set_xticks(range(len(imgs)))
    axes[1].set_xticklabels([image_names[i] for i in imgs], rotation=60,
                            ha='right', fontsize=7)
    axes[1].set(title='Mean error per image', ylabel='error (px)')

    labels, vals = zip(*recon.error_history) if recon.error_history else ((), ())
    axes[2].plot(range(len(vals)), vals, 'o-', color='steelblue')
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    axes[2].set(title='Mean error over reconstruction (after each BA)',
                ylabel='error (px)')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

def plot_scene(recon, out_path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    X = recon.points.array() if hasattr(recon.points, "array") else np.array(list(recon.points.values()))
    if len(X):
        lo, hi = np.percentile(X, [2, 98], axis=0)
        keep = np.all((X > lo - 0.5 * (hi - lo)) & (X < hi + 0.5 * (hi - lo)), axis=1)
        ax.scatter(*X[keep].T, s=1, c=X[keep, 2], cmap='viridis', alpha=0.6)
    for img, (R, t) in recon.poses.items():
        C = -R.T @ t
        ax.scatter(*C, color='red', s=30)
        z = R.T @ np.array([0, 0, 1])
        ax.quiver(*C, *z, length=0.25, color='red')
        ax.text(*C, str(img), fontsize=8)
    ax.set(title=f'{len(recon.points)} points, {len(recon.poses)} cameras')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

def export_ply(recon, out_path, colors=None):
    """colors: optional {track_id: (r,g,b)}."""
    tids = sorted(recon.points)
    X = np.array([recon.points[t] for t in tids]).reshape(-1, 3)
    c = np.array([(colors or {}).get(t, (200, 200, 200)) for t in tids], int).reshape(-1, 3)
    with open(out_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(tids)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        if len(tids):
            np.savetxt(f, np.hstack([X, c]), fmt='%.6f %.6f %.6f %d %d %d')


def export_poses(recon, image_names, out_path):
    data = {'images': {}, 'cameras': {}}
    for img, (R, t) in sorted(recon.poses.items()):
        data['images'][image_names[img]] = {
            'camera_id': int(recon.cam_of_image[img]),
            'R_world_to_cam': R.tolist(),
            't': t.tolist(),
            'camera_center': (-R.T @ t).tolist(),
        }
    for c in recon.cameras:
        data['cameras'][c.cam_id] = {
            'label': c.label, 'width': c.width, 'height': c.height,
            'fx': c.fx, 'fy': c.fy, 'cx': c.cx, 'cy': c.cy,
            'focal_source': c.source,
            'model': c.model, 'distortion': np.asarray(c.dist).tolist(),
            'refined_in_ba': bool(c.refine_focal or c.refine_principal),
        }
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
