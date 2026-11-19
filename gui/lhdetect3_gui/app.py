"""LHDetect3 desktop application (PySide6)."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QPointF, QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QIcon, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGraphicsItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from . import __version__
from .batch import done_images, existing_unit, run_batch, unit_name, write_csv
from .engine import DEFAULT_CONFIDENCE, Engine, ImageResult, Params, list_images, load_rgb, resource_dir
from .viz import palette

APP = "LHDetect3"
REPO_URL = "https://github.com/genaev/lhdetect3"
PRESETS = [("Custom", None), ("Wheat dataset (1.2 µm/px)", 1.2),
           ("Soybean protocol 2 (3.0 µm/px)", 3.0), ("No scale: report pixels", 0.0)]


# =========================================================================== workers

class EngineLoader(QObject):
    loaded = Signal(object)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    @Slot()
    def run(self):
        try:
            engine = Engine(device=self.device, log=self.message.emit)
            self.loaded.emit(engine)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class BatchWorker(QObject):
    result = Signal(int, int, object)
    message = Signal(str)
    finished = Signal(bool)

    def __init__(self, engine, files, root, params, scale, csv_path, append, annotated_dir):
        super().__init__()
        self.args = (engine, files, root, params, scale, csv_path, append, annotated_dir)
        self._stop = False

    def stop(self):
        self._stop = True

    @Slot()
    def run(self):
        engine, files, root, params, scale, csv_path, append, annotated_dir = self.args
        ok = True
        try:
            run_batch(engine, files, root, params, scale, csv_path, append=append,
                      annotated_dir=annotated_dir, on_result=self.result.emit,
                      should_stop=lambda: self._stop, log=self.message.emit)
        except Exception as exc:  # noqa: BLE001
            ok = False
            self.message.emit(f"Error: {exc}\n{traceback.format_exc()}")
        self.finished.emit(ok and not self._stop)


class SingleWorker(QObject):
    done = Signal(object)

    def __init__(self, engine, path, root, params):
        super().__init__()
        self.args = (path, root, params)
        self.engine = engine

    @Slot()
    def run(self):
        path, root, params = self.args
        self.done.emit(self.engine.run_file(path, root, params))


# =========================================================================== viewer

class ImageView(QGraphicsView):
    """Zoomable image with the predicted trichomes drawn on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.pix: Optional[QGraphicsPixmapItem] = None
        self.overlay: List[QGraphicsItem] = []
        self.show_contours = True
        self.show_centerlines = True
        self.show_numbers = True
        self._fit = True
        self._placeholder("Select an image in the list")

    def _placeholder(self, text: str):
        self.scene().clear()
        self.pix = None
        self.overlay = []
        item = self.scene().addSimpleText(text)
        item.setBrush(QColor(200, 200, 200))
        self.setSceneRect(item.boundingRect())

    def set_image(self, rgb: Optional[np.ndarray], message: str = ""):
        if rgb is None:
            self._placeholder(message or "Cannot read the image")
            return
        h, w = rgb.shape[:2]
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.scene().clear()
        self.overlay = []
        self.pix = self.scene().addPixmap(QPixmap.fromImage(qimg))
        self.setSceneRect(0, 0, w, h)
        if self._fit:
            self.fit()

    def set_result(self, res: Optional[ImageResult], scale: float):
        for it in self.overlay:
            self.scene().removeItem(it)
        self.overlay = []
        if res is None or self.pix is None:
            return
        unit = "µm" if scale > 0 else "px"
        k = scale if scale > 0 else 1.0
        for i, (t, c) in enumerate(zip(res.trichomes, palette(len(res.trichomes))), 1):
            color = QColor(*c)
            if t.length_px is not None:
                tip = f"Trichome {i}: length {t.length_px * k:.1f} {unit}, score {t.score:.2f}"
            else:
                tip = f"Trichome {i}: length not measured (mask too small), score {t.score:.2f}"
            if self.show_contours and t.contour is None:
                x1, y1, x2, y2 = t.box
                item = self.scene().addRect(x1, y1, x2 - x1, y2 - y1)
                pen = QPen(color, 1.5, Qt.DashLine)
                pen.setCosmetic(True)
                item.setPen(pen)
                item.setToolTip(tip)
                self.overlay.append(item)
            elif self.show_contours:
                poly = QPolygonF([QPointF(float(x), float(y)) for x, y in t.contour])
                item = QGraphicsPolygonItem(poly)
                pen = QPen(color, 2.5)
                pen.setCosmetic(True)
                item.setPen(pen)
                fill = QColor(color)
                fill.setAlpha(45)
                item.setBrush(QBrush(fill))
                item.setToolTip(tip)
                self.scene().addItem(item)
                self.overlay.append(item)
            if self.show_centerlines and len(t.path) > 1:
                path = QPainterPath(QPointF(*t.path[0]))
                for x, y in t.path[1:]:
                    path.lineTo(x, y)
                for width, col in ((3.5, QColor(255, 255, 255, 200)), (1.6, color)):
                    item = QGraphicsPathItem(path)
                    pen = QPen(col, width)
                    pen.setCosmetic(True)
                    item.setPen(pen)
                    item.setToolTip(tip)
                    self.scene().addItem(item)
                    self.overlay.append(item)
            if self.show_numbers:
                if t.path:
                    x, y = t.path[len(t.path) // 2]
                elif t.contour is not None:
                    x, y = tuple(t.contour.mean(0))
                else:
                    x, y = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
                label = QGraphicsSimpleTextItem(str(i))
                f = label.font()
                f.setPointSize(11)
                f.setBold(True)
                label.setFont(f)
                label.setBrush(color)
                label.setPen(QPen(QColor(0, 0, 0), 0.6))
                label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                label.setPos(x + 4, y - 8)
                label.setToolTip(tip)
                self.scene().addItem(label)
                self.overlay.append(label)

    def fit(self):
        if self.pix is not None:
            self.fitInView(self.pix, Qt.KeepAspectRatio)
            self._fit = True

    def actual_size(self):
        self.resetTransform()
        self._fit = False

    def wheelEvent(self, event):
        if self.pix is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._fit = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit()


# =========================================================================== main window

class MainWindow(QMainWindow):
    COLS = ["Image", "Count", "Mean length", "Status"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP} {__version__} — trichome counting and length measurement")
        icon = resource_dir() / "resources" / "icon.png"
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        self.settings = QSettings(APP, "LHDetect3GUI")
        self.engine: Optional[Engine] = None
        self.files: List[Path] = []
        self.root: Optional[Path] = None
        self.rows: Dict[str, int] = {}
        self.results: Dict[str, ImageResult] = {}
        self.thread: Optional[QThread] = None
        self.worker = None
        self.single_thread: Optional[QThread] = None
        self.pending_preview: Optional[str] = None
        self.t_start = 0.0
        self.n_total = 0
        self._build()
        self._restore()
        self._load_engine()

    # ---------------------------------------------------------------- layout
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        box = QGroupBox("Input and output")
        form = QFormLayout(box)
        self.ed_folder = QLineEdit()
        self.ed_folder.setPlaceholderText("Folder with images (PNG, JPEG, TIFF, BMP)")
        bt = self.bt_browse = QPushButton("Browse…")
        bt.clicked.connect(self.choose_folder)
        self.cb_recursive = QCheckBox("Include subfolders")
        self.cb_recursive.setChecked(True)
        self.cb_recursive.toggled.connect(lambda _: self.scan_folder())
        row = QHBoxLayout()
        row.addWidget(self.ed_folder, 1)
        row.addWidget(bt)
        row.addWidget(self.cb_recursive)
        form.addRow("Image folder:", row)
        self.ed_folder.editingFinished.connect(self.scan_folder)

        self.ed_csv = QLineEdit()
        self.ed_csv.setPlaceholderText("Results table (CSV)")
        bt2 = self.bt_saveas = QPushButton("Save as…")
        bt2.clicked.connect(self.choose_csv)
        row = QHBoxLayout()
        row.addWidget(self.ed_csv, 1)
        row.addWidget(bt2)
        form.addRow("Results file:", row)

        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setDecimals(4)
        self.sp_scale.setRange(0.0, 10000.0)
        self.sp_scale.setSingleStep(0.1)
        self.sp_scale.setSpecialValueText("not set (pixels)")
        self.sp_scale.setSuffix(" µm per pixel")
        self.sp_scale.setMinimumWidth(190)
        self.sp_scale.setToolTip("Size of one image pixel in micrometers. Lengths in the results are "
                                 "multiplied by this value. Leave 0 to report lengths in pixels.")
        self.cb_preset = QComboBox()
        for name, _ in PRESETS:
            self.cb_preset.addItem(name)
        self.cb_preset.activated.connect(self._preset_chosen)
        self.sp_scale.valueChanged.connect(self._scale_changed)
        row = QHBoxLayout()
        row.addWidget(self.sp_scale)
        row.addWidget(QLabel("Preset:"))
        row.addWidget(self.cb_preset)
        row.addStretch(1)
        form.addRow("Scale:", row)

        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setDecimals(2)
        self.sp_conf.setRange(0.05, 0.95)
        self.sp_conf.setSingleStep(0.01)
        self.sp_conf.setValue(DEFAULT_CONFIDENCE)
        self.sp_conf.setToolTip(f"Minimal detection score. {DEFAULT_CONFIDENCE} was selected on the "
                                "validation set and is recommended.")
        self.cb_annot = QCheckBox("Save images with drawn contours")
        self.cb_annot.setToolTip("JPEG images are written to the folder <results file name>_annotated")
        self.cb_device = QComboBox()
        self.cb_device.addItems(["Automatic (GPU if available)", "CPU only"])
        self.cb_device.activated.connect(lambda _: self._load_engine())
        self.lb_device = QLabel("Device: loading the model…")
        row = QHBoxLayout()
        row.addWidget(self.sp_conf)
        row.addSpacing(16)
        row.addWidget(self.cb_annot)
        row.addSpacing(16)
        row.addWidget(QLabel("Device:"))
        row.addWidget(self.cb_device)
        row.addWidget(self.lb_device)
        row.addStretch(1)
        form.addRow("Confidence threshold:", row)
        outer.addWidget(box)

        row = QHBoxLayout()
        self.bt_run = QPushButton("▶  Run")
        self.bt_run.setMinimumHeight(34)
        self.bt_run.setStyleSheet("QPushButton{font-weight:bold;padding:0 18px}")
        self.bt_run.clicked.connect(self.start)
        self.bt_stop = QPushButton("■  Stop")
        self.bt_stop.setMinimumHeight(34)
        self.bt_stop.setEnabled(False)
        self.bt_stop.clicked.connect(self.stop)
        self.progress = QProgressBar()
        self.progress.setMinimumHeight(26)
        self.progress.setFormat("%v / %m images")
        self.progress.setValue(0)
        self.progress.setMaximum(1)
        self.lb_eta = QLabel("")
        row.addWidget(self.bt_run)
        row.addWidget(self.bt_stop)
        row.addWidget(self.progress, 1)
        row.addWidget(self.lb_eta)
        outer.addLayout(row)

        split = QSplitter(Qt.Horizontal)
        tabs = QTabWidget()
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(self.COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.preview_selected)
        tabs.addTab(self.table, "Images")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        tabs.addTab(self.log_box, "Log")
        split.addWidget(tabs)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.lb_info = QLabel("")
        self.lb_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bt_fit = QToolButton()
        bt_fit.setText("Fit")
        bt_fit.clicked.connect(lambda: self.view.fit())
        bt_100 = QToolButton()
        bt_100.setText("100%")
        bt_100.clicked.connect(lambda: self.view.actual_size())
        self.cb_show = QCheckBox("Contours")
        self.cb_cl = QCheckBox("Centerlines")
        self.cb_num = QCheckBox("Numbers")
        for cb in (self.cb_show, self.cb_cl, self.cb_num):
            cb.setChecked(True)
            cb.toggled.connect(self._overlay_options)
        bar.addWidget(self.lb_info, 1)
        for w in (self.cb_show, self.cb_cl, self.cb_num, bt_fit, bt_100):
            bar.addWidget(w)
        rl.addLayout(bar)
        self.view = ImageView()
        rl.addWidget(self.view, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setSizes([430, 900])
        outer.addWidget(split, 1)

        menu = self.menuBar().addMenu("&File")
        act = QAction("Open image folder…", self)
        act.triggered.connect(self.choose_folder)
        menu.addAction(act)
        act = QAction("Export results table with the current scale…", self)
        act.triggered.connect(self.export_csv)
        menu.addAction(act)
        menu.addSeparator()
        act = QAction("Quit", self)
        act.triggered.connect(self.close)
        menu.addAction(act)
        hm = self.menuBar().addMenu("&Help")
        act = QAction("How to use", self)
        act.triggered.connect(self.show_help)
        hm.addAction(act)
        act = QAction("Project page (GitHub)", self)
        act.triggered.connect(lambda: QDesktopServices.openUrl(REPO_URL))
        hm.addAction(act)
        act = QAction("About", self)
        act.triggered.connect(self.show_about)
        hm.addAction(act)
        self.statusBar().showMessage("Choose a folder with images")
        self.resize(1400, 860)

    # ---------------------------------------------------------------- settings
    def _restore(self):
        s = self.settings
        self.ed_folder.setText(s.value("folder", "", str))
        self.ed_csv.setText(s.value("csv", "", str))
        self.sp_scale.setValue(float(s.value("scale", 0.0)))
        self.sp_conf.setValue(float(s.value("conf", DEFAULT_CONFIDENCE)))
        self.cb_recursive.setChecked(s.value("recursive", True, bool))
        self.cb_annot.setChecked(s.value("annot", False, bool))
        self.cb_device.setCurrentIndex(int(s.value("device", 0)))
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self._scale_changed(self.sp_scale.value())
        if self.ed_folder.text():
            QTimer.singleShot(0, self.scan_folder)

    def _save(self):
        s = self.settings
        s.setValue("folder", self.ed_folder.text())
        s.setValue("csv", self.ed_csv.text())
        s.setValue("scale", self.sp_scale.value())
        s.setValue("conf", self.sp_conf.value())
        s.setValue("recursive", self.cb_recursive.isChecked())
        s.setValue("annot", self.cb_annot.isChecked())
        s.setValue("device", self.cb_device.currentIndex())
        s.setValue("geometry", self.saveGeometry())

    # ---------------------------------------------------------------- engine
    def log(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{stamp}] {text}")

    def _load_engine(self):
        if self.thread is not None:
            return
        old = getattr(self, "_loader_thread", None)
        try:
            if old is not None and old.isRunning():
                return
        except RuntimeError:  # already deleted
            pass
        self.engine = None
        self.bt_run.setEnabled(False)
        self.lb_device.setText("loading the model…")
        device = "cpu" if self.cb_device.currentIndex() == 1 else "auto"
        th = QThread(self)
        loader = EngineLoader(device)
        loader.moveToThread(th)
        th.started.connect(loader.run)
        loader.message.connect(self.log)
        loader.loaded.connect(self._engine_ready)
        loader.failed.connect(self._engine_failed)
        loader.loaded.connect(th.quit)
        loader.failed.connect(th.quit)
        th.finished.connect(th.deleteLater)
        self._loader = loader
        self._loader_thread = th
        th.start()

    def _engine_ready(self, engine: Engine):
        self.engine = engine
        gpu = engine.provider != "CPUExecutionProvider"
        self.lb_device.setText(engine.device_label)
        self.lb_device.setStyleSheet("color:#1a7f37;font-weight:bold" if gpu else "font-weight:bold")
        self.lb_device.setToolTip(f"ONNX Runtime {engine.ort.__version__}; model input {engine.resolution} px")
        self.bt_run.setEnabled(True)
        self.statusBar().showMessage(f"Ready. Inference on {engine.device_label}")
        if self.pending_preview:
            self.preview_selected()

    def _engine_failed(self, text: str):
        self.lb_device.setText("model not loaded")
        self.log(text)
        QMessageBox.critical(self, APP, f"The model could not be loaded:\n\n{text.splitlines()[0]}")

    # ---------------------------------------------------------------- inputs
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with images", self.ed_folder.text() or str(Path.home()))
        if d:
            self.ed_folder.setText(d)
            self.ed_csv.setText(str(Path(d) / "lhdetect3_results.csv"))
            self.scan_folder()

    def choose_csv(self):
        start = self.ed_csv.text() or str(Path(self.ed_folder.text() or Path.home()) / "lhdetect3_results.csv")
        f, _ = QFileDialog.getSaveFileName(self, "Results file", start, "CSV files (*.csv)")
        if f:
            if not f.lower().endswith(".csv"):
                f += ".csv"
            self.ed_csv.setText(f)

    def _preset_chosen(self, idx: int):
        value = PRESETS[idx][1]
        if value is not None:
            self.sp_scale.setValue(value)

    def _scale_changed(self, value: float):
        idx = 0
        for i, (_, v) in enumerate(PRESETS):
            if v is not None and abs(v - value) < 1e-9:
                idx = i
        self.cb_preset.setCurrentIndex(idx)
        unit = "µm" if value > 0 else "px"
        self.table.setHorizontalHeaderItem(2, QTableWidgetItem(f"Mean length, {unit}"))
        for rel, res in self.results.items():
            self._update_row(res)
        self._show_info()
        if self.view.pix is not None:
            self.view.set_result(self._current_result(), value)

    def scan_folder(self):
        if self.thread is not None:
            return
        folder = Path(self.ed_folder.text().strip())
        self.table.setRowCount(0)
        self.rows.clear()
        self.results.clear()
        self.files = []
        self.view._placeholder("Select an image in the list")
        if not folder.is_dir():
            self.statusBar().showMessage("Choose a folder with images")
            return
        if not self.ed_csv.text().strip():
            self.ed_csv.setText(str(folder / "lhdetect3_results.csv"))
        self.root = folder
        self.files = list_images(folder, self.cb_recursive.isChecked())
        self.table.setRowCount(len(self.files))
        for i, p in enumerate(self.files):
            rel = p.relative_to(folder).as_posix()
            self.rows[rel] = i
            self.table.setItem(i, 0, QTableWidgetItem(rel))
            for c in (1, 2):
                it = QTableWidgetItem("")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, c, it)
            self.table.setItem(i, 3, QTableWidgetItem("waiting"))
        self.progress.setMaximum(max(1, len(self.files)))
        self.progress.setValue(0)
        self.lb_eta.setText("")
        self.statusBar().showMessage(f"{len(self.files)} images found in {folder}")

    # ---------------------------------------------------------------- run
    def _params(self) -> Params:
        return Params(confidence=self.sp_conf.value())

    def start(self):
        if self.engine is None or self.thread is not None:
            return
        if not self.files:
            self.scan_folder()
        if not self.files:
            QMessageBox.information(self, APP, "No images found. Choose a folder with images first.")
            return
        csv_path = Path(self.ed_csv.text().strip() or (self.root / "lhdetect3_results.csv"))
        scale = self.sp_scale.value()
        files = list(self.files)
        append = False
        if csv_path.is_file() and csv_path.stat().st_size > 0:
            done = done_images(csv_path)
            box = QMessageBox(self)
            box.setWindowTitle(APP)
            box.setIcon(QMessageBox.Question)
            box.setText(f"The results file already exists:\n{csv_path}")
            b_cont = None
            same_unit = existing_unit(csv_path) == unit_name(scale)
            if done and same_unit:
                box.setInformativeText(f"It lists {len(done)} images. Continue (skip these images) or overwrite?")
                b_cont = box.addButton("Continue", QMessageBox.AcceptRole)
            else:
                box.setInformativeText("Overwrite it?")
            b_over = box.addButton("Overwrite", QMessageBox.DestructiveRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_cont:
                files = [f for f in files if f.relative_to(self.root).as_posix() not in done]
                append = True
                for rel in done:
                    if rel in self.rows:
                        self.table.item(self.rows[rel], 3).setText("done earlier")
            elif clicked is not b_over:
                return
        if not files:
            QMessageBox.information(self, APP, "All images are already in the results file.")
            return
        annotated = None
        if self.cb_annot.isChecked():
            annotated = csv_path.with_name(csv_path.stem + "_annotated")
        self._save()
        self.n_total = len(files)
        self.progress.setMaximum(len(files))
        self.progress.setValue(0)
        self.t_start = time.time()
        self.lb_eta.setText("starting…")
        self._set_running(True)
        self.log(f"Run: {len(files)} images, scale {scale or 'not set'}, threshold {self.sp_conf.value():.2f}, "
                 f"device {self.engine.device_label}; results → {csv_path}")
        th = QThread(self)
        w = BatchWorker(self.engine, files, self.root, self._params(), scale, csv_path, append, annotated)
        w.moveToThread(th)
        th.started.connect(w.run)
        w.result.connect(self._on_result)
        w.message.connect(self.log)
        w.finished.connect(self._on_finished)
        w.finished.connect(th.quit)
        th.finished.connect(th.deleteLater)
        self.thread, self.worker = th, w
        self._csv_path = csv_path
        th.start()

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.bt_stop.setEnabled(False)
            self.lb_eta.setText("stopping after the current image…")

    def _set_running(self, running: bool):
        self.bt_run.setEnabled(not running and self.engine is not None)
        self.bt_stop.setEnabled(running)
        for w in (self.ed_folder, self.ed_csv, self.cb_recursive, self.sp_conf, self.cb_annot, self.cb_device,
                  self.bt_browse, self.bt_saveas, self.sp_scale, self.cb_preset):
            w.setEnabled(not running)

    def _on_result(self, i: int, n: int, res: ImageResult):
        self.results[res.image] = res
        self._update_row(res)
        self.progress.setValue(i)
        el = time.time() - self.t_start
        per = el / i
        left = per * (n - i)
        self.lb_eta.setText(f"{per:.1f} s/image · {self._fmt_time(left)} left")
        self.statusBar().showMessage(f"{res.image}: {res.count} trichomes")
        if res.image == self._selected_rel():
            self.preview_selected()

    def _on_finished(self, complete: bool):
        el = time.time() - self.t_start
        self.thread = None
        self.worker = None
        self._set_running(False)
        done = self.progress.value()
        self.lb_eta.setText(f"{'done' if complete else 'stopped'} in {self._fmt_time(el)}")
        msg = f"{'Finished' if complete else 'Stopped'}: {done} images in {self._fmt_time(el)}. Results: {self._csv_path}"
        self.log(msg)
        self.statusBar().showMessage(msg)
        if complete:
            box = QMessageBox(self)
            box.setWindowTitle(APP)
            box.setText(f"Processed {done} images.\n\nResults: {self._csv_path}")
            b_open = box.addButton("Open folder", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Ok)
            box.exec()
            if box.clickedButton() is b_open:
                QDesktopServices.openUrl(self._csv_path.parent.as_uri())

    @staticmethod
    def _fmt_time(sec: float) -> str:
        sec = int(round(sec))
        if sec < 60:
            return f"{sec} s"
        if sec < 3600:
            return f"{sec // 60} min {sec % 60:02d} s"
        return f"{sec // 3600} h {sec % 3600 // 60:02d} min"

    def _update_row(self, res: ImageResult):
        r = self.rows.get(res.image)
        if r is None:
            return
        scale = self.sp_scale.value()
        if res.error:
            self.table.item(r, 1).setText("")
            self.table.item(r, 2).setText("")
            self.table.item(r, 3).setText(res.error)
            return
        lengths = res.lengths(scale)
        self.table.item(r, 1).setText(str(res.count))
        self.table.item(r, 2).setText(f"{np.mean(lengths):.1f}" if lengths else "")
        self.table.item(r, 3).setText(f"done, {res.seconds:.1f} s")

    def export_csv(self):
        if not self.results:
            QMessageBox.information(self, APP, "There are no results in this session yet.")
            return
        start = self.ed_csv.text() or str(Path.home() / "lhdetect3_results.csv")
        f, _ = QFileDialog.getSaveFileName(self, "Export results", start, "CSV files (*.csv)")
        if not f:
            return
        order = sorted(self.results.values(), key=lambda r: self.rows.get(r.image, 0))
        write_csv(Path(f), order, self.sp_scale.value())
        self.log(f"Exported {len(order)} images to {f}")

    # ---------------------------------------------------------------- preview
    def _selected_rel(self) -> Optional[str]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return item.text() if item else None

    def _current_result(self) -> Optional[ImageResult]:
        rel = self._selected_rel()
        return self.results.get(rel) if rel else None

    def _overlay_options(self):
        self.view.show_contours = self.cb_show.isChecked()
        self.view.show_centerlines = self.cb_cl.isChecked()
        self.view.show_numbers = self.cb_num.isChecked()
        self.view.set_result(self._current_result(), self.sp_scale.value())

    def _show_info(self):
        res = self._current_result()
        rel = self._selected_rel()
        if not rel:
            self.lb_info.setText("")
            return
        if res is None:
            self.lb_info.setText(f"<b>{rel}</b>")
            return
        if res.error:
            self.lb_info.setText(f"<b>{rel}</b> — {res.error}")
            return
        scale = self.sp_scale.value()
        unit = "µm" if scale > 0 else "px"
        lengths = res.lengths(scale)
        mean = f", mean length {np.mean(lengths):.1f} {unit}" if lengths else ""
        meas = f" ({res.n_measured} measured)" if res.n_measured != res.count else ""
        self.lb_info.setText(f"<b>{rel}</b> — {res.count} trichomes{meas}{mean}")

    def preview_selected(self):
        rel = self._selected_rel()
        if not rel or self.root is None:
            return
        path = self.root / rel
        rgb = load_rgb(path)
        self.view.set_image(rgb)
        res = self.results.get(rel)
        self._show_info()
        if res is not None:
            self.view.set_result(res, self.sp_scale.value())
            return
        if rgb is None:
            return
        if self.thread is not None:
            self.lb_info.setText(f"<b>{rel}</b> — will be shown when processed")
            return
        if self.engine is None:
            self.pending_preview = rel
            self.lb_info.setText(f"<b>{rel}</b> — waiting for the model")
            return
        if self.single_thread is not None:
            self.pending_preview = rel
            return
        self.pending_preview = None
        self.lb_info.setText(f"<b>{rel}</b> — detecting trichomes…")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        th = QThread(self)
        w = SingleWorker(self.engine, path, self.root, self._params())
        w.moveToThread(th)
        th.started.connect(w.run)
        w.done.connect(self._single_done)
        w.done.connect(th.quit)
        th.finished.connect(th.deleteLater)
        self.single_thread, self._single_worker = th, w
        th.start()

    def _single_done(self, res: ImageResult):
        QApplication.restoreOverrideCursor()
        self.single_thread = None
        if res.error:
            self.log(f"{res.image}: {res.error}")
        self.results[res.image] = res
        self._update_row(res)
        r = self.rows.get(res.image)
        if r is not None and not res.error:
            self.table.item(r, 3).setText(f"preview, {res.seconds:.1f} s")
        if self.pending_preview and self.pending_preview != res.image:
            self.preview_selected()
        elif res.image == self._selected_rel():
            self.view.set_result(res, self.sp_scale.value())
            self._show_info()

    # ---------------------------------------------------------------- dialogs
    def show_help(self):
        QMessageBox.information(self, f"{APP}: how to use", (
            "1. Choose the folder with leaf-fold images (Browse…).\n"
            "2. Enter the image scale in micrometers per pixel, or choose a preset. With the scale set, "
            "all lengths are reported in micrometers; without it, in pixels.\n"
            "3. Press Run. The progress bar shows the processed images and the remaining time. "
            "Rows of the results file are written as soon as each image is ready, so an interrupted run "
            "can be continued.\n"
            "4. Click an image in the list to see the detected trichomes: each contour has its own color, "
            "the thin line is the centerline used to measure the length, and the numbers follow the order "
            "of the measured lengths in the results file. A dashed box marks a trichome that was detected but "
            "whose mask is too small to measure; it is counted but has no length. Hover over a trichome to see "
            "its length.\n\n"
            "Results file columns: image, count (number of detected trichomes), n_measured (trichomes with a "
            "measured length), mean_length, lengths (separated by semicolons).\n\n"
            "The GPU is used automatically when available (NVIDIA, AMD, Intel via DirectML on Windows; "
            "CUDA or ROCm on Linux; Core ML on macOS); otherwise the CPU is used."))

    def show_about(self):
        dev = self.engine.device_label if self.engine else "not loaded"
        QMessageBox.about(self, f"About {APP}", (
            f"<b>{APP} {__version__}</b><br>Counting and length measurement of leaf trichomes in soybean "
            "and wheat leaf-fold images with an RF-DETR-Seg Small model (ONNX Runtime).<br><br>"
            f"Inference device: {dev}<br><br>"
            f"<a href='{REPO_URL}'>{REPO_URL}</a><br><br>"
            "If you use this software, please cite: Genaev M.A. et al. Quantitative Assessment of the "
            "Pubescence of Soybean and Wheat Leaves Based on RGB Images Using Deep Learning Methods. "
            "<i>Plants</i>."))

    def closeEvent(self, event):
        if self.thread is not None:
            if QMessageBox.question(self, APP, "Processing is running. Stop and quit?") != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(30000)
        self._save()
        event.accept()


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setOrganizationName(APP)
    app.setStyle("Fusion")
    icon = resource_dir() / "resources" / "icon.png"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
