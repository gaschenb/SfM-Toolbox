"""Pinhole intrinsics per COLMAP-style camera group.

Focal resolution order: focal_px; focal_35mm (f_px = f35/36 * w); focal_mm with
pixel_size_mm (f_px = f_mm / pixel_size, rescaled by w / sensor_res_x); focal_mm with
sensor_width_mm (f_px = f_mm / sensor_width * w).
"""
from dataclasses import dataclass, field
import numpy as np
from sfmtoolbox.core.distortion import distort, undistort, n_params

def focal_px_from_spec(width_px, focal_px=None, focal_35mm=None,
                       focal_mm=None, sensor_width_mm=None,
                       pixel_size_mm=None, sensor_res_x=None):
    """Focal in pixels for an image of width width_px, or None if underspecified."""
    if focal_px:
        return float(focal_px)
    if focal_35mm:
        return focal_35mm / 36.0 * width_px
    if focal_mm:
        if pixel_size_mm is None and sensor_width_mm and sensor_res_x:
            pixel_size_mm = sensor_width_mm / sensor_res_x
        if pixel_size_mm:
            f_native = focal_mm / pixel_size_mm
            native_w = sensor_res_x or width_px
            return f_native * (width_px / native_w)
        if sensor_width_mm:
            return focal_mm / sensor_width_mm * width_px
    return None

@dataclass
class Camera:
    """Pinhole camera (fx, fy, cx, cy). One per COLMAP-style camera group."""
    cam_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    refine_focal: bool = False
    refine_principal: bool = False
    label: str = ''
    source: str = field(default='heuristic')
    model: str = 'pinhole'
    dist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    refine_dist: bool = False

    def __post_init__(self):
        n = n_params(self.model)
        d = np.zeros(n)
        given = np.asarray(self.dist, float).ravel()[:n]
        d[:len(given)] = given
        self.dist = d

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1.0]])

    @property
    def f(self):
        return 0.5 * (self.fx + self.fy)

    def set_K(self, K):
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

    @property
    def distorted(self):
        return self.model == 'fisheye' or (self.model != 'pinhole' and bool(np.any(self.dist)))

    def normalize(self, uv):
        """Pixels -> undistorted normalized coordinates."""
        xd = (np.asarray(uv, float) - [self.cx, self.cy]) / [self.fx, self.fy]
        return undistort(xd, self.model, self.dist) if self.distorted else xd

    def denormalize(self, xn):
        """Undistorted normalized coordinates -> pixels."""
        xn = np.asarray(xn, float)
        if self.distorted:
            xn = distort(xn, self.model, self.dist)
        return xn * [self.fx, self.fy] + [self.cx, self.cy]

    def undistort_pixels(self, uv):
        """Pixels -> pinhole pixels with the same K."""
        if not self.distorted:
            return np.asarray(uv, float)
        return self.normalize(uv) * [self.fx, self.fy] + [self.cx, self.cy]

    def project(self, Xc):
        """Camera-frame points (N,3) -> pixels (N,2)."""
        return self.denormalize(Xc[:, :2] / Xc[:, 2:3])

    @classmethod
    def from_spec(cls, cam_id, width, height, spec, label=''):
        f = focal_px_from_spec(width,
                               focal_px=spec.get('focal_px'),
                               focal_35mm=spec.get('focal_35mm'),
                               focal_mm=spec.get('focal_mm'),
                               sensor_width_mm=spec.get('sensor_width_mm'),
                               pixel_size_mm=spec.get('pixel_size_mm'),
                               sensor_res_x=spec.get('sensor_res_x'))
        source = ('user-px' if spec.get('focal_px') else
                  'user-35mm' if spec.get('focal_35mm') else
                  'user-mm' if f is not None else 'heuristic')
        if f is None:
            f = 1.2 * max(width, height)
        return cls(cam_id, width, height, fx=f, fy=f,
                   cx=spec.get('cx', width / 2), cy=spec.get('cy', height / 2),
                   refine_focal=spec.get('refine_focal', False),
                   refine_principal=spec.get('refine_principal', False),
                   label=label, source=source,
                   model=spec.get('model', 'pinhole'),
                   dist=np.asarray(spec.get('dist', ()), float),
                   refine_dist=spec.get('refine_dist', False))

def exif_focal_spec(path):
    """Best-effort EXIF -> partial spec dict."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        exif = Image.open(path)._getexif() or {}
        tags = {TAGS.get(k, k): v for k, v in exif.items()}
        out = {}
        if tags.get('FocalLength'):
            out['focal_mm'] = float(tags['FocalLength'])
        return out
    except Exception:
        return {}

def build_cameras(image_sizes, mode='shared', spec=None, specs=None,
                  labels=None):
    """mode 'shared': one camera per distinct image size; 'per_image': one per image.
    Returns (cameras, cam_of_image)."""
    n = len(image_sizes)
    spec = spec or {}
    specs = specs or [None] * n
    labels = labels or [''] * n
    cameras, cam_of_image = [], []
    if mode == 'shared':
        groups = {}
        for i, (w, h) in enumerate(image_sizes):
            key = (w, h)
            if key not in groups:
                cam = Camera.from_spec(len(cameras), w, h,
                                       specs[i] or spec,
                                       label=f'shared {w}x{h}')
                groups[key] = cam.cam_id
                cameras.append(cam)
            cam_of_image.append(groups[key])
    elif mode == 'per_image':
        for i, (w, h) in enumerate(image_sizes):
            cam = Camera.from_spec(i, w, h, specs[i] or spec,
                                   label=labels[i])
            cameras.append(cam)
            cam_of_image.append(i)
    else:
        raise ValueError(mode)
    return cameras, cam_of_image
