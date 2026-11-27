"""COLMAP-style PyQt GUI for the SfM toolbox."""
import sys
import argparse
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from sfmtoolbox.core import pipeline
from sfmtoolbox.core.camera import focal_px_from_spec
from sfmtoolbox.core.visualize import export_ply, export_poses
from sfmtoolbox.core.pipeline import point_colors
from sfmtoolbox.core.features import available_extractors, _REGISTRY as _EXTRACTORS
from sfmtoolbox.core.analytics import format_summary
from sfmtoolbox.core.distortion import MODELS as DIST_MODELS
from sfmtoolbox.core.pairs import PAIR_METHODS

try:
    import pyqtgraph.opengl as gl
    HAVE_GL = True

    class PickableGLView(gl.GLViewWidget):
        """GLViewWidget that emits clicked(x, y) for click-without-drag,
        while keeping normal orbit/pan interaction."""
        clicked = QtCore.pyqtSignal(float, float)

        def mousePressEvent(self, ev):
            self._press = ev.pos()
            super().mousePressEvent(ev)

        def mouseReleaseEvent(self, ev):
            if hasattr(self, '_press') and \
                    (ev.pos() - self._press).manhattanLength() < 4:
                self.clicked.emit(ev.pos().x(), ev.pos().y())
            super().mouseReleaseEvent(ev)
except Exception:
    HAVE_GL = False

PAIR_HELP = {
    'exhaustive': (
        'Exhaustive: match every image against every other, n(n-1)/2 pairs.\n'
        'Most complete and slowest; matching cost grows quadratically. Fine for up to '
        'a few dozen images.',
        'Not used by exhaustive.', False),
    'sequential': (
        'Sequential: match each image with the next k images in filename order, plus '
        'the first k against the last k (loop closure).\nBest for video or ordered '
        'captures; about n*k pairs.',
        'k = window size: how many following images each image is matched against.\n'
        'Higher k: more overlap found and more robust to skipped frames, but more '
        'matching.\nLower k: faster, but gaps in the sequence can split the '
        'reconstruction.', True),
    'vocab': (
        'Vocabulary tree: cluster all descriptors into a tree of visual words '
        '(hierarchical k-means), describe each image as a TF-IDF bag of words, and '
        'match each image only with its k most similar images.\nScales to large '
        'unordered collections; about n*k pairs.',
        'k = retrieved neighbours per image.\nHigher k: higher recall of overlapping '
        'pairs (fewer images left unregistered), cost grows linearly in k.\nLower k: '
        'faster, but weakly overlapping views may never be matched. 10-20 is typical; '
        'combine small k with "expand".', True),
    'vlad': (
        'VLAD: one global descriptor per image (residuals to 64 visual words, power- '
        'and L2-normalised); match each image with its k nearest neighbours by cosine '
        'similarity.\nCheaper to build than the vocabulary tree, slightly less '
        'discriminative.',
        'k = nearest neighbours per image.\nHigher k: more recall, more matching '
        '(linear in k).\nLower k: faster, risk of missing overlaps.', True),
    'gps': (
        'GPS: match each image with its k spatially nearest images using EXIF GPS '
        'positions.\nGood for drone/outdoor captures; falls back to the vocabulary '
        'tree if any image lacks GPS.',
        'k = nearest images by GPS distance.\nHigher k: tolerates GPS noise and '
        'wide-baseline overlaps, more matching.\nLower k: faster, but views pointing '
        'different ways at the same spot can be missed.', True),
    'hybrid': (
        'Hybrid: union of sequential (window k), vocabulary tree (k neighbours) and, '
        'when every image has GPS, GPS kNN.\nMost robust non-exhaustive option.',
        'k is used both as the sequential window and as the number of retrieved '
        'neighbours.\nHigher k: more recall, more matching. Lower k: faster.', True),
}


class CameraDialog(QtWidgets.QDialog):
    """Intrinsics specification dialog."""

    MODES = ['Auto (EXIF / heuristic)',
             'Focal length (pixels)',
             'Focal (mm) + sensor width (mm)',
             'Focal (mm) + pixel size (mm)',
             'Focal (mm) + sensor width (mm) + sensor resolution (px)']

    def __init__(self, parent=None, image_size=None, spec=None,
                 show_assignment=True, camera_mode='shared'):
        super().__init__(parent)
        self.setWindowTitle('Camera parameters')
        self.image_size = image_size or (0, 0)
        spec = spec or {}
        form = QtWidgets.QFormLayout(self)

        if show_assignment:
            self.assign = QtWidgets.QComboBox()
            self.assign.addItems(['Shared camera for all images',
                                  'Individual camera per image'])
            self.assign.setCurrentIndex(1 if camera_mode == 'per_image' else 0)
            form.addRow('Camera assignment:', self.assign)
        else:
            self.assign = None

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(self.MODES)
        form.addRow('Focal specification:', self.mode)

        def spin(lo, hi, dec, val, suffix=''):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(val)
            if suffix:
                s.setSuffix(suffix)
            s.valueChanged.connect(self._update_preview)
            return s

        self.focal_px = spin(1, 1e6, 2, spec.get('focal_px', 1000), ' px')
        self.focal_mm = spin(0.1, 10000, 3, spec.get('focal_mm', 24), ' mm')
        self.sensor_w = spin(0.1, 1000, 3,
                             spec.get('sensor_width_mm', 23.5), ' mm')
        self.pixel_sz = spin(1e-5, 1.0, 6,
                             spec.get('pixel_size_mm', 0.0043), ' mm')
        self.sensor_rx = QtWidgets.QSpinBox()
        self.sensor_rx.setRange(1, 10 ** 6)
        self.sensor_rx.setValue(int(spec.get('sensor_res_x') or
                                    self.image_size[0] or 6000))
        self.sensor_rx.setSuffix(' px')
        self.sensor_rx.valueChanged.connect(self._update_preview)

        form.addRow('Focal length (px):', self.focal_px)
        form.addRow('Focal length (mm):', self.focal_mm)
        form.addRow('Sensor width (mm):', self.sensor_w)
        form.addRow('Pixel size / pitch (mm):', self.pixel_sz)
        form.addRow('Sensor resolution X (px):', self.sensor_rx)

        self.pixel_sz_derived = QtWidgets.QLabel('-')
        form.addRow('Derived pixel size:', self.pixel_sz_derived)
        self.preview = QtWidgets.QLabel('-')
        self.preview.setStyleSheet('font-weight: bold')
        form.addRow('Resulting focal (px):', self.preview)

        self.refine_f = QtWidgets.QCheckBox(
            'Refine focal length in bundle adjustment')
        self.refine_f.setChecked(spec.get('refine_focal', False))
        self.refine_pp = QtWidgets.QCheckBox(
            'Refine principal point in bundle adjustment')
        self.refine_pp.setChecked(spec.get('refine_principal', False))
        form.addRow(self.refine_f)
        form.addRow(self.refine_pp)

        self.lens = QtWidgets.QComboBox()
        self.lens.addItems(list(DIST_MODELS))
        self.lens.setCurrentText(spec.get('model', 'pinhole'))
        form.addRow('Lens distortion model:', self.lens)
        self.dist_edit = QtWidgets.QLineEdit(
            ', '.join(f'{v:g}' for v in spec.get('dist', ())))
        self.dist_edit.setPlaceholderText('initial coefficients, e.g. -0.1, 0.02 (blank = 0)')
        form.addRow('Coefficients:', self.dist_edit)
        self.refine_d = QtWidgets.QCheckBox('Refine distortion in bundle adjustment')
        self.refine_d.setChecked(spec.get('refine_dist', True))
        form.addRow(self.refine_d)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed(self.mode.currentIndex())

    def _mode_changed(self, m):
        vis = {self.focal_px: m == 1,
               self.focal_mm: m in (2, 3, 4),
               self.sensor_w: m in (2, 4),
               self.pixel_sz: m == 3,
               self.sensor_rx: m in (3, 4)}
        lay = self.layout()
        for w, v in vis.items():
            w.setVisible(v)
            lbl = lay.labelForField(w)
            if lbl:
                lbl.setVisible(v)
        self._update_preview()

    def current_spec(self):
        m = self.mode.currentIndex()
        spec = {}
        if m == 1:
            spec['focal_px'] = self.focal_px.value()
        elif m == 2:
            spec['focal_mm'] = self.focal_mm.value()
            spec['sensor_width_mm'] = self.sensor_w.value()
        elif m == 3:
            spec['focal_mm'] = self.focal_mm.value()
            spec['pixel_size_mm'] = self.pixel_sz.value()
            spec['sensor_res_x'] = self.sensor_rx.value()
        elif m == 4:
            spec['focal_mm'] = self.focal_mm.value()
            spec['sensor_width_mm'] = self.sensor_w.value()
            spec['sensor_res_x'] = self.sensor_rx.value()
        spec['refine_focal'] = self.refine_f.isChecked()
        spec['refine_principal'] = self.refine_pp.isChecked()
        spec['model'] = self.lens.currentText()
        spec['refine_dist'] = self.refine_d.isChecked() and spec['model'] != 'pinhole'
        txt = self.dist_edit.text().replace(',', ' ').split()
        if txt:
            spec['dist'] = [float(v) for v in txt]
        return spec

    def camera_mode(self):
        if self.assign is None:
            return None
        return 'per_image' if self.assign.currentIndex() == 1 else 'shared'

    def _update_preview(self):
        spec = self.current_spec()
        w = self.image_size[0] or self.sensor_rx.value()
        if spec.get('sensor_width_mm') and spec.get('sensor_res_x'):
            ps = spec['sensor_width_mm'] / spec['sensor_res_x']
            self.pixel_sz_derived.setText(f'{ps:.6f} mm')
        elif spec.get('pixel_size_mm'):
            self.pixel_sz_derived.setText(f"{spec['pixel_size_mm']:.6f} mm "
                                          '(given)')
        else:
            self.pixel_sz_derived.setText('-')
        f = focal_px_from_spec(w, **{k: spec.get(k) for k in
                                     ('focal_px', 'focal_mm',
                                      'sensor_width_mm', 'pixel_size_mm',
                                      'sensor_res_x')})
        if f:
            self.preview.setText(f'{f:.1f} px  (image width {w} px)')
        else:
            self.preview.setText('auto (EXIF if available, else '
                                 f'{1.2 * max(self.image_size or [0]):.0f} px)')

class Worker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    snapshot = QtCore.pyqtSignal(object)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, folder, camera_mode, spec, specs, options):
        super().__init__()
        self.folder = folder
        self.camera_mode = camera_mode
        self.spec = spec
        self.specs = specs
        self.options = options

    def run(self):
        try:
            def progress(recon):
                self.snapshot.emit(self._snap(recon))

            recon, names, colors_img = pipeline.run_pipeline(
                self.folder, out_dir=None, camera_mode=self.camera_mode,
                spec=self.spec, specs=self.specs, log=self.log.emit,
                progress_fn=progress, **self.options)
            self.snapshot.emit(self._snap(recon))
            self.finished_ok.emit((recon, names, colors_img))
        except Exception as e:
            import traceback
            self.failed.emit(f'{e}\n{traceback.format_exc()}')

    @staticmethod
    def _snap(recon):
        pts = np.array(list(recon.points.values())) if recon.points \
            else np.zeros((0, 3))
        cams = [(img, R.copy(), t.copy())
                for img, (R, t) in recon.poses.items()]
        return pts, cams

class Viewer3D(QtWidgets.QWidget):
    """Point cloud + camera frusta. pyqtgraph OpenGL, matplotlib fallback."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.gl_ok = False
        self.cams = []
        self.cam_items = {}
        self.selected = None
        self.text_item = None
        if HAVE_GL:
            try:
                self.view = PickableGLView()
                self.view.clicked.connect(self.on_click)
                self.view.setCameraPosition(distance=12)
                g = gl.GLGridItem()
                g.scale(1, 1, 1)
                self.view.addItem(g)
                lay.addWidget(self.view)
                self.items = []
                self.gl_ok = True
            except Exception:
                pass
        if not self.gl_ok:
            self.fig = Figure()
            self.canvas = FigureCanvasQTAgg(self.fig)
            lay.addWidget(self.canvas)

    CAM_COLOR = (1, 0.3, 0.2, 1)
    SEL_COLOR = (1, 1, 0.2, 1)

    def update_scene(self, pts, cams, colors=None, names=None):
        self.cams = [(img, R, t,
                      names[img] if names else f'image {img}')
                     for img, R, t in cams]
        self.selected = None
        self.text_item = None
        self.cam_items = {}
        if self.gl_ok:
            for it in self.items:
                self.view.removeItem(it)
            self.items = []
            if len(pts):
                lo, hi = np.percentile(pts, [5, 95], axis=0)
                ctr = (lo + hi) / 2
                from PyQt5.QtGui import QVector3D
                self.view.opts['center'] = QVector3D(*ctr.astype(float))
                self.view.setCameraPosition(
                    distance=max(1.5 * np.linalg.norm(hi - lo), 2.0))
            if len(pts):
                c = colors if colors is not None else \
                    np.tile([0.8, 0.8, 0.8, 1.0], (len(pts), 1))
                sp = gl.GLScatterPlotItem(pos=pts.astype(np.float32),
                                          color=c.astype(np.float32),
                                          size=2.5)
                self.view.addItem(sp)
                self.items.append(sp)
            for img, R, t, name in self.cams:
                C = (-R.T @ t)
                s = 0.3
                corners = np.array([[-s, -s, 2 * s], [s, -s, 2 * s],
                                    [s, s, 2 * s], [-s, s, 2 * s]])
                cw = (R.T @ (corners.T - t[:, None])).T
                segs = []
                for k in range(4):
                    segs += [C, cw[k], cw[k], cw[(k + 1) % 4]]
                ln = gl.GLLinePlotItem(pos=np.array(segs, np.float32),
                                       color=self.CAM_COLOR, width=1.5,
                                       mode='lines')
                self.view.addItem(ln)
                self.items.append(ln)
                self.cam_items[img] = ln
        else:
            self.fig.clear()
            ax = self.fig.add_subplot(111, projection='3d')
            if len(pts):
                lo, hi = np.percentile(pts, [2, 98], axis=0)
                m = np.all((pts > lo - (hi - lo)) & (pts < hi + (hi - lo)), 1)
                col = colors[m, :3] if colors is not None else None
                ax.scatter(*pts[m].T, s=1, c=col)
            for img, R, t, name in self.cams:
                C = -R.T @ t
                ax.scatter(*C, color='red', s=25)
                ax.text(*C, name, fontsize=7)
            self.canvas.draw_idle()

    def screen_pos(self, world):
        """Project a 3D world point to widget pixel coordinates."""
        from PyQt5.QtGui import QVector4D
        try:
            m = self.view.projectionMatrix() * self.view.viewMatrix()
        except TypeError:
            vp = self.view.getViewport()
            m = self.view.projectionMatrix(vp, vp) * self.view.viewMatrix()
        v = m.map(QVector4D(*[float(x) for x in world], 1.0))
        if abs(v.w()) < 1e-12 or v.z() < -abs(v.w()):
            return None
        ndc_x, ndc_y = v.x() / v.w(), v.y() / v.w()
        return ((ndc_x + 1) / 2 * self.view.width(),
                (1 - ndc_y) / 2 * self.view.height())

    def on_click(self, x, y):
        if not self.cams:
            return
        best, best_d = None, 20.0
        for img, R, t, name in self.cams:
            sp = self.screen_pos(-R.T @ t)
            if sp is None:
                continue
            d = np.hypot(sp[0] - x, sp[1] - y)
            if d < best_d:
                best_d, best = d, (img, R, t, name)
        if best is None:
            return
        img, R, t, name = best
        if self.text_item is not None:
            self.view.removeItem(self.text_item)
            self.text_item = None
            if self.selected in self.cam_items:
                self.cam_items[self.selected].setData(color=self.CAM_COLOR)
            if self.selected == img:
                self.selected = None
                return
        C = -R.T @ t
        self.cam_items[img].setData(color=self.SEL_COLOR)
        self.text_item = gl.GLTextItem(
            pos=C, text=f'{name}  C=({C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f})',
            color=(255, 255, 100, 255))
        self.view.addItem(self.text_item)
        self.selected = img

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SfM Toolbox')
        self.resize(1280, 800)
        self.folder = None
        self.paths = []
        self.sizes = []
        self.camera_mode = 'shared'
        self.default_spec = {}
        self.per_image_specs = None
        self.result = None
        self.worker = None

        tb = self.addToolBar('Main')
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.act_open = tb.addAction('Open images…', self.open_folder)
        self.act_cam = tb.addAction('Camera parameters…', self.camera_dialog)
        self.act_cam_sel = tb.addAction('Camera for selection…',
                                        self.camera_for_selection)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(' Downsample: '))
        self.downsample = QtWidgets.QDoubleSpinBox()
        self.downsample.setRange(1.0, 16.0)
        self.downsample.setSingleStep(0.5)
        self.downsample.setValue(1.0)
        self.downsample.setSuffix('x')
        self.downsample.setToolTip('Linear downsample factor applied to the '
                                   'images before processing (1 = full '
                                   'resolution). Reported intrinsics refer '
                                   'to the downsampled images.')
        tb.addWidget(self.downsample)
        tb.addWidget(QtWidgets.QLabel(' Features: '))
        self.feature = QtWidgets.QComboBox()
        model = self.feature.model()
        for name in available_extractors():
            cls = _EXTRACTORS[name]
            self.feature.addItem(name)
            if not cls.available():
                it = model.item(self.feature.count() - 1)
                it.setEnabled(False)
                it.setText(f'{name} (not installed)')
                it.setToolTip(f'needs: {cls.requires or "opencv-contrib-python-headless"}')
        self.feature.setCurrentIndex(self.feature.findText('sift'))
        tb.addWidget(self.feature)
        tb.addWidget(QtWidgets.QLabel(' Pairs: '))
        self.pairs = QtWidgets.QComboBox()
        for i, m in enumerate(PAIR_METHODS):
            self.pairs.addItem(m)
            self.pairs.setItemData(i, PAIR_HELP[m][0], Qt.ToolTipRole)
        tb.addWidget(self.pairs)
        self.pairs_k = QtWidgets.QSpinBox()
        self.pairs_k.setRange(1, 200)
        self.pairs_k.setValue(10)
        self.pairs_k.setPrefix('k=')
        tb.addWidget(self.pairs_k)
        self.pairs.currentTextChanged.connect(self._pairs_changed)
        self._pairs_changed(self.pairs.currentText())
        self.expand = QtWidgets.QCheckBox('expand')
        self.expand.setToolTip(
            'Transitive expansion (one round): after geometric verification, also match '
            'pairs (i, k) where (i, j) and (j, k) both verified but (i, k) was not a '
            'candidate.\nRecovers overlaps the selection method missed, at the cost of '
            'extra matching; most useful with small k.')
        tb.addWidget(self.expand)
        tb.addSeparator()
        self.act_run = tb.addAction('▶ Run reconstruction', self.run_recon)
        self.act_export = tb.addAction('Export…', self.export)
        self.act_cam.setEnabled(False)
        self.act_cam_sel.setEnabled(False)
        self.act_run.setEnabled(False)
        self.act_export.setEnabled(False)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['Image', 'Size', 'Camera', 'f (px)'])
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        dock = QtWidgets.QDockWidget('Images', self)
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.log_widget = QtWidgets.QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setStyleSheet('font-family: monospace')
        self.viewer = Viewer3D()
        self.fig = Figure()
        self.plot_canvas = FigureCanvasQTAgg(self.fig)
        self.cam_table = QtWidgets.QTreeWidget()
        self.cam_table.setHeaderLabels(
            ['Camera', 'Images', 'fx', 'fy', 'cx', 'cy', 'source', 'refined', 'lens',
             'coefficients'])
        self.tabs.addTab(self.log_widget, 'Log')
        self.tabs.addTab(self.viewer, '3D view')
        self.tabs.addTab(self.plot_canvas, 'Reprojection errors')
        self.tabs.addTab(self.cam_table, 'Cameras')
        self.analytics_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.analytics_text.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.FixedFont))
        self.tabs.addTab(self.analytics_text, 'Analytics')
        self.statusBar().showMessage('Open an image folder to begin')

    def open_folder(self, folder=None):
        if not folder:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, 'Image folder')
        if not folder:
            return
        self.folder = folder
        self.paths = pipeline.list_images(folder)
        import cv2
        self.sizes = []
        for p in self.paths:
            img = cv2.imread(str(p))
            self.sizes.append((img.shape[1], img.shape[0])
                              if img is not None else (0, 0))
        self.per_image_specs = None
        self.refresh_tree()
        ok = len(self.paths) >= 2
        for a in (self.act_cam, self.act_cam_sel, self.act_run):
            a.setEnabled(ok)
        self.statusBar().showMessage(
            f'{len(self.paths)} images loaded from {folder}')

    def refresh_tree(self):
        self.tree.clear()
        for i, (p, (w, h)) in enumerate(zip(self.paths, self.sizes)):
            spec = (self.per_image_specs[i] if self.per_image_specs else
                    self.default_spec) or {}
            f = focal_px_from_spec(w, **{k: spec.get(k) for k in
                                         ('focal_px', 'focal_mm',
                                          'sensor_width_mm', 'pixel_size_mm',
                                          'sensor_res_x')})
            cam_lbl = (f'cam {i}' if self.camera_mode == 'per_image'
                       else 'shared')
            if self.per_image_specs and self.per_image_specs[i] and \
                    self.per_image_specs[i].get('_group'):
                cam_lbl = self.per_image_specs[i]['_group']
            it = QtWidgets.QTreeWidgetItem(
                [p.name, f'{w}x{h}', cam_lbl,
                 f'{f:.1f}' if f else 'auto'])
            it.setData(0, Qt.UserRole, i)
            self.tree.addTopLevelItem(it)
        for c in range(4):
            self.tree.resizeColumnToContents(c)

    def camera_dialog(self):
        dlg = CameraDialog(self, image_size=self.sizes[0],
                           spec=self.default_spec,
                           camera_mode=self.camera_mode)
        if dlg.exec_():
            self.default_spec = dlg.current_spec()
            self.camera_mode = dlg.camera_mode()
            self.refresh_tree()

    def camera_for_selection(self):
        rows = [it.data(0, Qt.UserRole)
                for it in self.tree.selectedItems()]
        if not rows:
            QtWidgets.QMessageBox.information(
                self, 'Camera for selection',
                'Select one or more images in the list first.')
            return
        dlg = CameraDialog(self, image_size=self.sizes[rows[0]],
                           show_assignment=False)
        if dlg.exec_():
            if self.per_image_specs is None:
                self.per_image_specs = [dict(self.default_spec)
                                        for _ in self.paths]
            spec = dlg.current_spec()
            gid = f'group@{min(rows)}'
            spec['_group'] = gid
            for r in rows:
                self.per_image_specs[r] = dict(spec)
            self.camera_mode = 'per_image'
            self.refresh_tree()
            self.log(f'[gui] assigned custom camera to images {sorted(rows)}')

    def log(self, msg):
        self.log_widget.appendPlainText(str(msg))

    def _pairs_changed(self, m):
        desc, k_help, k_on = PAIR_HELP[m]
        self.pairs.setToolTip(desc)
        self.pairs_k.setToolTip(k_help)
        self.pairs_k.setEnabled(k_on)

    def run_recon(self):
        if self.worker and self.worker.isRunning():
            return
        self.act_run.setEnabled(False)
        self.log_widget.clear()
        specs = None
        if self.per_image_specs:
            specs = [{k: v for k, v in s.items() if not k.startswith('_')}
                     for s in self.per_image_specs]
        self.worker = Worker(self.folder, self.camera_mode,
                             self.default_spec, specs,
                             options=dict(downsample=self.downsample.value(),
                                          feature=self.feature.currentText().split()[0],
                                          pairs=self.pairs.currentText(),
                                          pairs_k=self.pairs_k.value(),
                                          seq_window=self.pairs_k.value(),
                                          expand_transitive_iters=int(self.expand.isChecked())))
        self.worker.log.connect(self.log)
        self.worker.snapshot.connect(self.on_snapshot)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()
        self.statusBar().showMessage('Reconstruction running…')

    def on_snapshot(self, snap):
        pts, cams = snap
        self.viewer.update_scene(pts, cams,
                                 names=[p.name for p in self.paths])

    def on_failed(self, msg):
        self.log(f'\nERROR: {msg}')
        self.act_run.setEnabled(True)
        self.statusBar().showMessage('Reconstruction failed')

    def on_finished(self, result):
        self.result = result
        recon, names, colors_img = result
        cols = point_colors(recon, colors_img)
        tids = sorted(recon.points)
        pts = np.array([recon.points[t] for t in tids])
        c = np.array([[*cols[t], 255] for t in tids], float) / 255.0
        cams = [(img, R, t) for img, (R, t) in recon.poses.items()]
        self.viewer.update_scene(pts, cams, colors=c, names=names)
        self.draw_plots(recon, names)
        self.fill_cam_table(recon, names)
        if getattr(recon, 'analytics', None):
            self.analytics_text.setPlainText(format_summary(recon.analytics))
        self.act_run.setEnabled(True)
        self.act_export.setEnabled(True)
        self.statusBar().showMessage(
            f'Done: {len(recon.poses)}/{len(names)} cameras, '
            f'{len(recon.points)} points, mean reproj '
            f'{recon.mean_reprojection_error():.3f}px')
        self.tabs.setCurrentWidget(self.viewer)

    def draw_plots(self, recon, names):
        self.fig.clear()
        errs = recon.reprojection_errors()
        if not errs:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No triangulated points survived; see Log / Analytics.',
                    ha='center', va='center')
            ax.set_axis_off()
            self.plot_canvas.draw_idle()
            return
        all_e = np.concatenate(list(errs.values()))
        ax1, ax2, ax3 = self.fig.subplots(1, 3)
        ax1.hist(np.clip(all_e, 0, 10), bins=60, color='steelblue')
        ax1.axvline(all_e.mean(), color='r', ls='--',
                    label=f'mean {all_e.mean():.2f}px')
        ax1.set(title='Error histogram', xlabel='px')
        ax1.legend(fontsize=7)
        imgs = sorted(errs)
        ax2.bar(range(len(imgs)), [errs[i].mean() for i in imgs],
                color='steelblue')
        ax2.set_xticks(range(len(imgs)))
        ax2.set_xticklabels([names[i] for i in imgs], rotation=70,
                            fontsize=6, ha='right')
        ax2.set(title='Mean error / image', ylabel='px')
        labels, vals = zip(*recon.error_history) if recon.error_history else ((), ())
        ax3.plot(vals, 'o-', color='steelblue')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=70, fontsize=6, ha='right')
        ax3.set(title='Error vs BA stage', ylabel='px')
        ax3.grid(alpha=0.3)
        self.fig.tight_layout()
        self.plot_canvas.draw_idle()

    def fill_cam_table(self, recon, names):
        self.cam_table.clear()
        n_by_cam = {}
        for img in range(len(names)):
            n_by_cam.setdefault(recon.cam_of_image[img], []).append(img)
        for c in recon.cameras:
            imgs = n_by_cam.get(c.cam_id, [])
            it = QtWidgets.QTreeWidgetItem([
                f'{c.cam_id} {c.label}', str(len(imgs)),
                f'{c.fx:.2f}', f'{c.fy:.2f}', f'{c.cx:.2f}', f'{c.cy:.2f}',
                c.source,
                'yes' if (c.refine_focal or c.refine_principal or c.refine_dist) else 'no',
                c.model, ' '.join(f'{v:.4g}' for v in c.dist)])
            self.cam_table.addTopLevelItem(it)
        for k in range(10):
            self.cam_table.resizeColumnToContents(k)

    def export(self):
        if not self.result:
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Output folder')
        if not out:
            return
        recon, names, colors_img = self.result
        pipeline.write_outputs(recon, names, colors_img, out, log=self.log)
        self.statusBar().showMessage(f'Exported to {out}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images')
    args, qt_args = ap.parse_known_args()
    app = QtWidgets.QApplication(sys.argv[:1] + qt_args)
    win = MainWindow()
    win.show()
    if args.images:
        win.open_folder(args.images)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
