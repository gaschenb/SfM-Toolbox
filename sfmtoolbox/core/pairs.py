"""Candidate image-pair selection (replaces exhaustive O(n^2) matching).

Methods: exhaustive, sequential (window), vocab (hierarchical k-means vocabulary tree,
TF-IDF bag of words, Nister & Stewenius 2006), vlad (global VLAD descriptor kNN,
Jegou et al. 2010), gps (EXIF GPS kNN), hybrid (union of sequential + vocab [+ gps]).
expand_transitive() adds (i,k) for verified (i,j),(j,k) without a direct candidate.
"""
import itertools

import numpy as np
import cv2
from scipy.sparse import csr_matrix

PAIR_METHODS = ('exhaustive', 'sequential', 'vocab', 'vlad', 'gps', 'hybrid')


def _as_float(d, norm):
    if d is None or not len(d):
        return np.zeros((0, 1), np.float32)
    if norm == 'hamming':
        return np.unpackbits(np.asarray(d, np.uint8), axis=1).astype(np.float32)
    return np.asarray(d, np.float32)


def _kmeans(X, k, seed):
    k = min(k, len(X))
    if k <= 1:
        return np.asarray(X[:1], np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    _, _, C = cv2.kmeans(np.ascontiguousarray(X, np.float32), k, None, crit, 1,
                         cv2.KMEANS_PP_CENTERS)
    return C


def _nearest(X, C):
    d = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None]
    return np.argmin(d, axis=1)


class VocabularyTree:
    """Hierarchical k-means tree; leaves are visual words."""

    def __init__(self, branching=8, depth=4, max_train=200000, seed=0):
        self.b, self.L, self.max_train, self.seed = branching, depth, max_train, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            X = X[rng.choice(len(X), self.max_train, replace=False)]
        self.levels = []
        nodes = [np.arange(len(X))]
        for lvl in range(self.L):
            centers, parents, nxt = [], [], []
            for p, idx in enumerate(nodes):
                if len(idx) == 0:
                    continue
                C = _kmeans(X[idx], self.b, self.seed + lvl)
                lab = _nearest(X[idx], C)
                for c in range(len(C)):
                    centers.append(C[c])
                    parents.append(p)
                    nxt.append(idx[lab == c])
            self.levels.append((np.array(centers, np.float32), np.array(parents)))
            nodes = nxt
        self.n_words = len(self.levels[-1][0])
        return self

    def quantize(self, X):
        node = np.zeros(len(X), np.int64)
        for C, parents in self.levels:
            out = np.empty(len(X), np.int64)
            order = np.argsort(parents, kind='stable')
            starts = np.searchsorted(parents[order], np.arange(parents.max() + 2))
            for p in np.unique(node):
                m = node == p
                kids = order[starts[p]:starts[p + 1]]
                out[m] = kids[_nearest(X[m], C[kids])]
            node = out
        return node


def _bow_similarity(word_lists, n_words):
    n = len(word_lists)
    rows = np.concatenate([np.full(len(w), i) for i, w in enumerate(word_lists)])
    cols = np.concatenate(word_lists)
    tf = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n, n_words))
    tf.sum_duplicates()
    df = np.bincount(tf.indices, minlength=n_words)
    idf = np.log(n / np.maximum(df, 1))
    v = tf.multiply(1.0 / np.maximum(np.asarray(tf.sum(1)).ravel(), 1)[:, None]).multiply(idf[None])
    v = csr_matrix(v)
    nrm = np.sqrt(np.asarray(v.multiply(v).sum(1)).ravel())
    v = csr_matrix(v.multiply(1.0 / np.maximum(nrm, 1e-12)[:, None]))
    return (v @ v.T).toarray()


def _knn_pairs(S, k):
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(S) - 1)
    pairs = set()
    if k <= 0:
        return pairs
    nn = np.argsort(-S, axis=1)[:, :k]
    for i, row in enumerate(nn):
        for j in row:
            pairs.add((min(i, j), max(i, j)))
    return pairs


def vocab_similarity(descs, norm='l2', branching=8, depth=4, seed=0, per_image=2000):
    rng = np.random.default_rng(seed)
    F = [_as_float(d, norm) for d in descs]
    train = np.concatenate([f[rng.choice(len(f), min(per_image, len(f)), replace=False)]
                            for f in F if len(f)])
    tree = VocabularyTree(branching, depth, seed=seed).fit(train)
    words = [tree.quantize(f) if len(f) else np.zeros(0, np.int64) for f in F]
    return _bow_similarity(words, tree.n_words)


def vlad_similarity(descs, norm='l2', k=64, seed=0, per_image=2000):
    rng = np.random.default_rng(seed)
    F = [_as_float(d, norm) for d in descs]
    train = np.concatenate([f[rng.choice(len(f), min(per_image, len(f)), replace=False)]
                            for f in F if len(f)])
    C = _kmeans(train, k, seed)
    V = np.zeros((len(F), C.size), np.float32)
    for i, f in enumerate(F):
        if not len(f):
            continue
        a = _nearest(f, C)
        r = f - C[a]
        acc = np.zeros_like(C)
        np.add.at(acc, a, r)
        acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
        v = acc.ravel()
        v = np.sign(v) * np.sqrt(np.abs(v))
        V[i] = v / max(np.linalg.norm(v), 1e-12)
    return V @ V.T


def exif_gps(path):
    """(lat, lon, alt) in degrees/metres from EXIF, or None."""
    try:
        from PIL import Image
        ex = Image.open(path).getexif()
        g = ex.get_ifd(0x8825)
        if not g or 2 not in g or 4 not in g:
            return None

        def deg(v, ref):
            d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            return -d if ref in ('S', 'W') else d
        lat = deg(g[2], g.get(1, 'N'))
        lon = deg(g[4], g.get(3, 'E'))
        alt = float(g.get(6, 0.0)) * (-1 if g.get(5, 0) == 1 else 1)
        return lat, lon, alt
    except Exception:
        return None


def gps_to_enu(lla):
    """Local tangent-plane metres (equirectangular; fine for scenes << Earth radius)."""
    lla = np.asarray(lla, float)
    lat0, lon0 = np.radians(lla[:, 0].mean()), np.radians(lla[:, 1].mean())
    R = 6378137.0
    e = R * np.cos(lat0) * (np.radians(lla[:, 1]) - lon0)
    n = R * (np.radians(lla[:, 0]) - lat0)
    return np.c_[e, n, lla[:, 2] - lla[:, 2].mean()]


def select_pairs(method, n_images, descs=None, norm='l2', paths=None, gps=None, k=10,
                 window=5, loop=True, branching=8, depth=4, vlad_k=64, log=None):
    """-> (sorted list of (i,j) with i<j, info dict)."""
    log = log or (lambda *a: None)
    all_pairs = n_images * (n_images - 1) // 2
    info = dict(method=method, n_candidates=0, n_exhaustive=all_pairs)
    if method == 'exhaustive' or (method != 'sequential' and n_images <= k + 1):
        pairs = set(itertools.combinations(range(n_images), 2))
    elif method == 'sequential':
        pairs = {(i, j) for i in range(n_images)
                 for j in range(i + 1, min(i + window + 1, n_images))}
        if loop and n_images > 2 * window:
            pairs |= {(i, j) for i in range(window) for j in range(n_images - window, n_images)}
    elif method == 'vocab':
        S = vocab_similarity(descs, norm, branching, depth)
        pairs = _knn_pairs(S, k)
        info['similarity'] = S
    elif method == 'vlad':
        S = vlad_similarity(descs, norm, vlad_k)
        pairs = _knn_pairs(S, k)
        info['similarity'] = S
    elif method == 'gps':
        if gps is None and paths is not None:
            gps = [exif_gps(p) for p in paths]
        if gps is None or any(g is None for g in gps):
            log('[pairs] GPS missing for some images; falling back to vocab tree')
            return select_pairs('vocab', n_images, descs, norm, k=k, branching=branching,
                                depth=depth, log=log)
        P = gps_to_enu(gps)
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        pairs = _knn_pairs(-D, k)
    elif method == 'hybrid':
        pairs = set(select_pairs('sequential', n_images, window=window, loop=loop)[0])
        pairs |= set(select_pairs('vocab', n_images, descs, norm, k=k, branching=branching,
                                  depth=depth)[0])
        if paths is not None:
            g = [exif_gps(p) for p in paths]
            if all(x is not None for x in g):
                pairs |= set(select_pairs('gps', n_images, gps=g, k=k)[0])
    else:
        raise ValueError(f"unknown pair method {method!r}; choose from {PAIR_METHODS}")
    out = sorted(pairs)
    info['n_candidates'] = len(out)
    return out, info


def expand_transitive(verified_pairs, existing, n_images, max_new=None):
    """Candidates (i,k) such that (i,j) and (j,k) are verified for some j and (i,k) was
    not already tried, ranked by number of supporting j."""
    adj = [set() for _ in range(n_images)]
    for i, j in verified_pairs:
        adj[i].add(j)
        adj[j].add(i)
    tried = set(existing)
    support = {}
    for j in range(n_images):
        for i, k in itertools.combinations(sorted(adj[j]), 2):
            if (i, k) not in tried:
                support[(i, k)] = support.get((i, k), 0) + 1
    out = sorted(support, key=lambda p: (-support[p], p))
    return out[:max_new] if max_new else out
