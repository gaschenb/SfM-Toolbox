"""Pluggable feature extraction, matching, and track building."""
from abc import ABC, abstractmethod

import numpy as np
import cv2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_REGISTRY = {}


def register_extractor(name):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def available_extractors(installed_only=False):
    return sorted(k for k, c in _REGISTRY.items()
                  if not installed_only or c.available())


def get_extractor(name, **kwargs):
    if isinstance(name, FeatureExtractor):
        return name
    cls = _REGISTRY.get(str(name).lower())
    if cls is None:
        raise ValueError(f"unknown feature extractor {name!r}; "
                         f"available: {', '.join(available_extractors())}")
    if not cls.available():
        raise ImportError(f"extractor {name!r} needs optional dependencies: {cls.requires}")
    return cls(**kwargs)


class FeatureExtractor(ABC):
    """detect(gray uint8 HxW) -> (pts (N,2) float64 pixel xy, desc (N,D) or None)."""
    name = 'base'
    norm = 'l2'
    requires = ''

    def __init__(self, n_features=8000, **params):
        self.n_features = n_features
        self.params = params

    @classmethod
    def available(cls):
        return True

    @abstractmethod
    def detect(self, gray):
        ...

    def __call__(self, gray):
        return self.detect(gray)

    def __repr__(self):
        return f"{type(self).__name__}(n_features={self.n_features})"


def _cv_factory(name):
    """OpenCV >= 5 moved AKAZE/BRISK/KAZE to contrib's xfeatures2d."""
    for mod in (cv2, getattr(cv2, 'xfeatures2d', None)):
        f = getattr(mod, name, None) if mod is not None else None
        if f is not None:
            return f
    return None


class _OpenCVExtractor(FeatureExtractor):
    requires = 'opencv-contrib-python-headless (OpenCV >= 5 moved this to contrib)'
    cap = False
    factory = None
    defaults = {}

    @classmethod
    def available(cls):
        return _cv_factory(cls.factory) is not None

    def _make(self):
        f = _cv_factory(self.factory)
        if f is None:
            raise ImportError(f"cv2.{self.factory} not found; install opencv-contrib-python-headless")
        kw = dict(self.defaults)
        if 'nfeatures' in kw:
            kw['nfeatures'] = self.n_features
        return f(**{**kw, **self.params})

    def detect(self, gray):
        kps, desc = self._make().detectAndCompute(gray, None)
        if not kps:
            return np.zeros((0, 2)), None
        pts = np.array([k.pt for k in kps], dtype=np.float64)
        if self.cap and len(kps) > self.n_features:
            resp = np.array([k.response for k in kps])
            keep = np.sort(np.argsort(-resp, kind='stable')[:self.n_features])
            pts, desc = pts[keep], desc[keep]
        return pts, desc


@register_extractor('sift')
class SIFT(_OpenCVExtractor):
    factory = 'SIFT_create'
    defaults = {'nfeatures': None}


@register_extractor('rootsift')
class RootSIFT(SIFT):
    """L1-normalise then sqrt: L2 distance on the result is the Hellinger distance on SIFT."""
    def detect(self, gray):
        pts, d = super().detect(gray)
        if d is not None:
            d = np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1e-12)).astype(np.float32)
        return pts, d


@register_extractor('orb')
class ORB(_OpenCVExtractor):
    norm = 'hamming'
    factory = 'ORB_create'
    defaults = {'nfeatures': None}


@register_extractor('akaze')
class AKAZE(_OpenCVExtractor):
    norm = 'hamming'
    cap = True
    factory = 'AKAZE_create'
    defaults = {'threshold': 5e-4}


@register_extractor('brisk')
class BRISK(_OpenCVExtractor):
    norm = 'hamming'
    cap = True
    factory = 'BRISK_create'


@register_extractor('kaze')
class KAZE(_OpenCVExtractor):
    cap = True
    factory = 'KAZE_create'


class _TorchExtractor(FeatureExtractor):
    requires = 'torch, kornia'
    _modules = ('torch', 'kornia')

    @classmethod
    def available(cls):
        import importlib.util
        return all(importlib.util.find_spec(m) is not None for m in cls._modules)

    def __init__(self, n_features=8000, device=None, **params):
        super().__init__(n_features, **params)
        import torch
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None

    def _tensor(self, gray, channels=1):
        t = self.torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)[None, None]
        if channels == 3:
            t = t.expand(1, 3, *gray.shape).contiguous()
        return t.to(self.device)

    def _np(self, t, dtype):
        return t.detach().cpu().numpy().astype(dtype)


@register_extractor('disk')
class DISK(_TorchExtractor):
    def detect(self, gray):
        import kornia.feature as KF
        if self._model is None:
            self._model = KF.DISK.from_pretrained('depth').to(self.device).eval()
        with self.torch.inference_mode():
            f = self._model(self._tensor(gray, 3), n=self.n_features,
                            pad_if_not_divisible=True)[0]
        return self._np(f.keypoints, np.float64), self._np(f.descriptors, np.float32)


@register_extractor('superpoint')
class SuperPoint(_TorchExtractor):
    requires = 'torch, lightglue (pip install git+https://github.com/cvg/LightGlue)'
    _modules = ('torch', 'lightglue')

    def detect(self, gray):
        from lightglue import SuperPoint as _SP
        if self._model is None:
            self._model = _SP(max_num_keypoints=self.n_features).to(self.device).eval()
        with self.torch.inference_mode():
            f = self._model({'image': self._tensor(gray)})
        return self._np(f['keypoints'][0], np.float64), self._np(f['descriptors'][0], np.float32)


def detect_features(gray, method='sift', n_features=8000):
    return get_extractor(method, n_features=n_features).detect(gray)


def _norm_of(method):
    if method in ('l2', 'hamming'):
        return method
    if isinstance(method, FeatureExtractor):
        return method.norm
    cls = _REGISTRY.get(str(method).lower())
    return cls.norm if cls else 'l2'


def _knn_arrays(matcher, a, b):
    """-> (M,4) [query, train, d1, d2] for queries with two neighbours."""
    knn = matcher.knnMatch(a, b, k=2)
    out = np.array([(m[0].queryIdx, m[0].trainIdx, m[0].distance, m[1].distance)
                    for m in knn if len(m) == 2], dtype=np.float64)
    return out.reshape(-1, 4)


def make_matcher(norm, kind='bf'):
    if kind == 'flann':
        idx = (dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
               if norm == 'hamming' else dict(algorithm=1, trees=4))
        return cv2.FlannBasedMatcher(idx, dict(checks=64))
    return cv2.BFMatcher(cv2.NORM_HAMMING if norm == 'hamming' else cv2.NORM_L2)


def match_features(d1, d2, method='sift', ratio=0.8, matcher='bf'):
    """Lowe ratio + mutual cross-check. Returns (M,2) index pairs (i into d1, j into d2)."""
    empty = np.zeros((0, 2), int)
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return empty
    mt = make_matcher(_norm_of(method), matcher)
    f = _knn_arrays(mt, d1, d2)
    f = f[f[:, 2] < ratio * f[:, 3]]
    if not len(f):
        return empty
    qi, tj = f[:, 0].astype(int), f[:, 1].astype(int)
    uj = np.unique(tj)
    b = _knn_arrays(mt, d2[uj], d1)
    b = b[b[:, 2] < ratio * b[:, 3]]
    back = np.full(len(d2), -1)
    back[uj[b[:, 0].astype(int)]] = b[:, 1].astype(int)
    keep = back[tj] == qi
    return np.column_stack([qi[keep], tj[keep]]) if keep.any() else empty


def build_tracks(pair_matches, n_images):
    """{(i,j): (M,2)} -> list of {img: feat}. Tracks seeing an image twice are dropped.
    Track order and key order follow first appearance in pair_matches."""
    seq_img, seq_feat = [], []
    for (i, j), m in pair_matches.items():
        m = np.asarray(m, dtype=np.int64).reshape(-1, 2)
        if not len(m):
            continue
        im = np.empty(2 * len(m), np.int64)
        im[0::2], im[1::2] = i, j
        seq_img.append(im)
        seq_feat.append(m.ravel())
    if not seq_img:
        return []
    seq_img = np.concatenate(seq_img)
    seq_feat = np.concatenate(seq_feat)
    stride = int(seq_feat.max()) + 1
    nodes, first, inv = np.unique(seq_img * stride + seq_feat,
                                  return_index=True, return_inverse=True)
    n = len(nodes)
    g = coo_matrix((np.ones(len(inv) // 2, np.int8), (inv[0::2], inv[1::2])), shape=(n, n))
    _, lab = connected_components(g, directed=False)

    rank = np.empty(n, np.int64)
    rank[np.argsort(first, kind='stable')] = np.arange(n)
    comp_first = np.full(lab.max() + 1, n)
    np.minimum.at(comp_first, lab, rank)
    order = np.lexsort((rank, comp_first[lab]))
    lab_s = lab[order]
    img_s = nodes[order] // stride
    feat_s = nodes[order] % stride

    starts = np.flatnonzero(np.r_[True, lab_s[1:] != lab_s[:-1]])
    ends = np.r_[starts[1:], n]
    seg = np.repeat(np.arange(len(starts)), ends - starts)
    by_img = np.lexsort((img_s, seg))
    dupe = (seg[by_img][1:] == seg[by_img][:-1]) & (img_s[by_img][1:] == img_s[by_img][:-1])
    bad = np.zeros(len(starts), bool)
    bad[seg[by_img][1:][dupe]] = True

    img_l, feat_l = img_s.tolist(), feat_s.tolist()
    return [dict(zip(img_l[s:e], feat_l[s:e]))
            for s, e, b in zip(starts.tolist(), ends.tolist(), bad.tolist())
            if e - s >= 2 and not b]
