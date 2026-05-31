#######################################################################
#  Pandorica - Analytical tools for cryo-electron microscopy          #
#                                                                     #
#  https://github.com/RRobert92                                       #
#                                                                     #
#  Robert Kiewisz                                                     #
#  PolyForm Noncommercial License 1.0.0 - see LICENSE                 #
#######################################################################
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Robert Kiewisz

"""
napari dock widgets for the stitch validator + coarse-GT recorder.

* :class:`StitchValidatorWidget` — load a dataset, run the stitch, overlay raw
  vs. aligned microtubules (and aligned volumes), read the per-interface QC, and
  export the stitched volume + microtubules.
* :class:`CoarseGTWidget` — per-interface manual coarse alignment: rotate (slider),
  translate (spinbox or mouse-drag), and anisotropic scale (Sx, Sy sliders) of
  the moving bottom-face onto the fixed top-face. Saves ``{angle, tx, ty, sx,
  sy, scale}`` per interface to ``coarse_gt.json``. Older files without
  ``sx``/``sy`` load with the legacy isotropic ``scale`` value.

UI only — all loading/stitching/IO lives in :mod:`._io`, :mod:`._stitch`,
:mod:`._geometry`.
"""

import json
import os
import traceback
from os.path import join
from typing import Dict, List, Optional

import numpy as np
from qtpy.QtCore import QObject, QThread, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QApplication,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pandorica.stitch import geometry as geo
from pandorica.stitch import stitch as stitch
from pandorica.napari import _geometry as npg
from pandorica.stitch.dataset import Dataset, load_dataset
from pandorica.stitch.pipeline.stitcher import stitch_sections

GT_FILENAME = "coarse_gt.json"


# =========================================================================== #
#  Validator
# =========================================================================== #
class StitchValidatorWidget(QWidget):
    """Load → run stitch → overlay + QC → export stitched volume & microtubules."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.dataset: Optional[Dataset] = None
        self.poses: Optional[List[dict]] = None
        self.result = None
        self._build_ui()

    # ---- UI -------------------------------------------------------------- #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # dataset
        box = QGroupBox("Dataset")
        f = QFormLayout(box)
        self.folder_lbl = QLabel("(none)")
        self.folder_lbl.setWordWrap(True)
        browse = QPushButton("Browse folder…")
        browse.clicked.connect(self._browse)
        # New: load individual .am files (volumes + spatialGraph) directly —
        # tardis-style file picking when the user doesn't want to drop them
        # into a folder first.
        browse_files = QPushButton("Browse files…")
        browse_files.clicked.connect(self._browse_files)
        self.load_volumes_cb = QCheckBox("Load volumes (large)")
        self.display_ds = QSpinBox()
        self.display_ds.setRange(1, 32)
        self.display_ds.setValue(4)
        # New: render spatial graphs as connected polylines (splines) instead of
        # unstructured points. Default ON — matches the tardis-style display.
        self.splines_cb = QCheckBox("Render spatial graphs as splines")
        self.splines_cb.setChecked(True)
        load_btn = QPushButton("Load dataset")
        load_btn.clicked.connect(self._load)
        f.addRow(browse, self.folder_lbl)
        f.addRow(browse_files)
        f.addRow("Display downscale", self.display_ds)
        f.addRow(self.load_volumes_cb)
        f.addRow(self.splines_cb)
        f.addRow(load_btn)
        layout.addWidget(box)

        # run
        rbox = QGroupBox("Run stitch")
        rl = QVBoxLayout(rbox)
        h = QHBoxLayout()
        run_btn = QPushButton("Run stitch")
        run_btn.clicked.connect(self._run)
        h.addWidget(run_btn)
        rl.addLayout(h)
        self.use_images_cb = QCheckBox("Build face images from volumes (slow)")
        rl.addWidget(self.use_images_cb)
        sh = QHBoxLayout()
        self.warp_omega = QDoubleSpinBox()
        self.warp_omega.setRange(0.05, 2.0)
        self.warp_omega.setSingleStep(0.05)
        self.warp_omega.setDecimals(2)
        self.warp_omega.setValue(0.5)  # gentler than the 1.0 pipeline default
        sh.addWidget(QLabel("Warp vorticity max (lower = smoother)"))
        sh.addWidget(self.warp_omega)
        rl.addLayout(sh)
        self.accept_lbl = QLabel("—")
        self.accept_lbl.setWordWrap(True)
        rl.addWidget(self.accept_lbl)
        layout.addWidget(rbox)

        # QC table
        self.table = QTableWidget(0, 0)
        self.table.setMinimumHeight(160)
        layout.addWidget(self.table)

        # export
        ebox = QGroupBox("Export stitched volume + microtubules")
        ef = QFormLayout(ebox)
        self.export_ds = QSpinBox()
        self.export_ds.setRange(1, 32)
        self.export_ds.setValue(1)
        self.write_vol_cb = QCheckBox("Write volume (else graph only)")
        self.write_vol_cb.setChecked(True)
        self.apply_warp_cb = QCheckBox("Apply TPS warp (fine deformation)")
        self.apply_warp_cb.setChecked(True)
        self.zblend_cb = QCheckBox("Z-blend warp (symmetric)")
        self.zblend_cb.setChecked(True)
        self.image_fill_cb = QCheckBox(
            "Image-fill MT-free regions (slow, loads volumes)"
        )
        self.imgfill_method = QComboBox()
        self.imgfill_method.addItems(["mi", "grad", "ncc"])  # mi: best visual (Robert)
        # performance
        from pandorica.stitch import accel as _accel

        self.gpu_cb = QCheckBox("GPU warp (mps/cuda)")
        self.gpu_cb.setChecked(_accel.gpu_available())
        self.gpu_cb.setEnabled(_accel.gpu_available())
        self.match_workers = QSpinBox()
        self.match_workers.setRange(1, max(1, (os.cpu_count() or 2)))
        self.match_workers.setValue(max(1, (os.cpu_count() or 4) - 2))
        exp_btn = QPushButton("Export → <folder>/stitched_output")
        exp_btn.clicked.connect(self._export)
        ef.addRow("Export downscale", self.export_ds)
        ef.addRow(self.write_vol_cb)
        ef.addRow(self.apply_warp_cb)
        ef.addRow(self.zblend_cb)
        ef.addRow(self.image_fill_cb)
        ef.addRow("Image-fill metric", self.imgfill_method)
        ef.addRow(self.gpu_cb)
        ef.addRow("Match workers", self.match_workers)
        ef.addRow(exp_btn)
        layout.addWidget(ebox)
        layout.addStretch()

    # ---- actions --------------------------------------------------------- #
    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if d:
            self.folder_lbl.setText(d)

    def _browse_files(self) -> None:
        """Pick individual .am files (volumes + spatial graphs); load directly."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select AmiraMesh files (volumes + *_spatialGraph.am)",
            "",
            "AmiraMesh (*.am);;All files (*)",
        )
        if not files:
            return
        try:
            self.dataset = self._dataset_from_files(files)
        except Exception as e:  # noqa: BLE001
            self._error("Load failed", e)
            return
        self.folder_lbl.setText(f"({len(files)} files selected)")
        self._after_load()

    def _load(self) -> None:
        folder = self.folder_lbl.text()
        if not os.path.isdir(folder):
            self._warn("Pick a valid dataset folder first.")
            return
        try:
            self.dataset = load_dataset(folder)
        except Exception as e:  # noqa: BLE001
            self._error("Load failed", e)
            return
        self._after_load()

    def _after_load(self) -> None:
        """Common post-load wiring (used by both folder and file pickers)."""
        self.poses = None
        self.result = None
        self._clear_layers()
        self._draw_sections(self.dataset.coords_list(), prefix="raw", aligned=False)
        if self.load_volumes_cb.isChecked():
            self._draw_volumes(aligned=False)
        names = ", ".join(self.dataset.names)
        self.accept_lbl.setText(
            f"Loaded {len(self.dataset)} sections: {names}\nRun stitch to align."
        )
        self.viewer.reset_view()

    def _dataset_from_files(self, paths) -> Dataset:
        """Build a Dataset from a list of individual .am files.

        Classifies each file by header content (matching :func:`sort_tomogram_files`'s
        rules: ``Lattice`` → image volume; ``VERTEX`` / ``EDGE`` /
        ``HxSpatialGraph`` → spatial graph). Pairs them by filename-stem
        containment — the graph stem must contain the image stem (or vice
        versa) for them to pair. Unpaired graphs become graph-only sections;
        unpaired volumes become volume-only sections.
        """
        from pandorica.io.amira import read_segmented_points
        from pandorica.stitch.dataset import Section, _derive_names, _stem

        image_paths: list = []
        coord_paths: list = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(8192)
            except OSError:
                continue
            if "Lattice" in head:
                image_paths.append(p)
            elif (
                "VERTEX" in head
                or "EDGE" in head
                or "HxSpatialGraph" in head
                or p.endswith("_spatialGraph.am")
            ):
                coord_paths.append(p)
            else:
                # Default: non-lattice .am with no graph markers → assume image
                image_paths.append(p)

        # Pair by stem containment (graph stem contains image stem, or symmetric).
        def stem(p: str) -> str:
            return os.path.splitext(os.path.basename(p))[0]

        used_coords: set = set()
        pairs: list = []
        for img in sorted(image_paths):
            img_stem = stem(img)
            match = None
            for c in coord_paths:
                if c in used_coords:
                    continue
                c_stem = stem(c).replace("_spatialGraph", "")
                if c_stem == img_stem or c_stem in img_stem or img_stem in c_stem:
                    match = c
                    break
            if match:
                used_coords.add(match)
            pairs.append((img, match))
        # Orphan graphs become graph-only sections (no volume).
        for c in sorted(coord_paths):
            if c not in used_coords:
                pairs.append((None, c))

        if not pairs:
            raise ValueError("No .am files classified as volume or spatial graph.")

        names = _derive_names(
            [_stem(coord or img, i) for i, (img, coord) in enumerate(pairs)]
        )
        sections: list = []
        for i, (img, coord) in enumerate(pairs):
            coords = np.empty((0, 4))
            if coord is not None:
                try:
                    c = read_segmented_points(coord)
                    if c is not None and len(c):
                        coords = np.asarray(c, dtype=float)
                except Exception:  # noqa: BLE001
                    pass
            sections.append(
                Section(
                    name=names[i], index=i, image_path=img, coord_path=coord,
                    coords=coords,
                )
            )
        # Use the first file's directory as the dataset folder (informational).
        folder = os.path.dirname(paths[0]) if paths else ""
        return Dataset(folder=folder, sections=sections)

    def _run(self) -> None:
        if self.dataset is None:
            self._warn("Load a dataset first.")
            return
        coords = self.dataset.coords_list()
        if any(len(c) == 0 for c in coords):
            self._warn(
                "Some sections have no microtubule graph — stitching needs graphs."
            )
            return
        omega = self.warp_omega.value()
        try:
            imgs = None
            if self.use_images_cb.isChecked():
                imgs = npg.boundary_face_images(self.dataset, downscale=8)
            self.result = stitch_sections(
                coords, section_images=imgs, warp_omega_max=omega
            )
        except Exception as e:  # noqa: BLE001
            self._error("Stitch failed", e)
            return
        self.poses = stitch.result_poses(self.result)
        self._draw_sections(
            [npg.apply_pose_to_coords(self.poses[i], c) for i, c in enumerate(coords)],
            prefix="aligned",
            aligned=True,
        )
        if self.load_volumes_cb.isChecked():
            self._draw_volumes(aligned=True)
        self._fill_table()
        self.accept_lbl.setText(
            f"Stitch: accepted={self.result.accepted}.  "
            f"Orange = aligned overlay; toggle 'raw:*' layers to compare."
        )
        self.viewer.reset_view()

    def _export(self) -> None:
        if self.dataset is None or self.poses is None:
            self._warn("Load a dataset and run the stitch first.")
            return
        out = join(self.dataset.folder, "stitched_output")
        warps = None
        if self.apply_warp_cb.isChecked() and self.result is not None:
            warps = stitch.result_warps(self.result)
        image_warps = None
        if self.image_fill_cb.isChecked():
            from pandorica.stitch import image_warp as _image_warp

            self.accept_lbl.setText("Image-fill: matching MT-free regions…")
            QApplication.processEvents()
            image_warps = _image_warp.image_residual_warps(
                self.dataset,
                self.poses,
                mt_warps=warps,
                method=self.imgfill_method.currentText(),
                workers=self.match_workers.value(),
                omega_max=self.warp_omega.value(),
                progress=lambda m, fr: (
                    self.accept_lbl.setText(f"Image-fill: {m} ({fr*100:.0f}%)"),
                    QApplication.processEvents(),
                ),
            )
        try:
            written = stitch.export_stitched(
                self.dataset,
                self.poses,
                out,
                downscale=self.export_ds.value(),
                write_volume=self.write_vol_cb.isChecked(),
                warps=warps,
                warp_zblend=self.zblend_cb.isChecked(),
                image_warps=image_warps,
                use_gpu=self.gpu_cb.isChecked(),
                progress=lambda m, fr: self.accept_lbl.setText(
                    f"Export: {m} ({fr*100:.0f}%)"
                ),
            )
        except Exception as e:  # noqa: BLE001
            self._error("Export failed", e)
            return
        msg = "\n".join(f"{k}: {v}" for k, v in written.items())
        self.accept_lbl.setText("Exported:\n" + msg)
        QMessageBox.information(self, "Export complete", msg)

    # ---- drawing --------------------------------------------------------- #
    def _clear_layers(self, prefixes=("raw", "aligned", "vol")) -> None:
        for lyr in list(self.viewer.layers):
            if lyr.name.split(":")[0] in prefixes:
                self.viewer.layers.remove(lyr)

    def _draw_sections(self, coords_list, prefix: str, aligned: bool) -> None:
        """Dispatch on the splines checkbox: render as paths or as points."""
        self._clear_layers(prefixes=(prefix,))
        as_splines = self.splines_cb.isChecked()
        for i, c in enumerate(coords_list):
            color = npg.section_color(i)
            if as_splines:
                paths = npg.coords_to_paths_zyx(c)
                if not paths:
                    continue
                self.viewer.add_shapes(
                    paths,
                    shape_type="path",
                    name=f"{prefix}:{self.dataset.names[i]}",
                    edge_color=color,
                    edge_width=2.0,
                    opacity=0.9 if aligned else 0.55,
                    visible=aligned,
                )
            else:
                pts = npg.coords_to_points_zyx(c)
                if len(pts) == 0:
                    continue
                self.viewer.add_points(
                    pts,
                    name=f"{prefix}:{self.dataset.names[i]}",
                    size=max(np.ptp(pts[:, 1:]) / 200.0, 1.0) if len(pts) else 1.0,
                    face_color=color,
                    border_color=color,
                    opacity=0.9 if aligned else 0.35,
                    visible=aligned,
                )

    def _draw_volumes(self, aligned: bool) -> None:
        self._clear_layers(prefixes=("vol",))
        ds = self.display_ds.value()
        for i, s in enumerate(self.dataset.sections):
            if not s.has_volume():
                continue
            try:
                vol = s.load_volume(downscale=ds)
            except Exception:  # noqa: BLE001
                continue
            px = s.pixel_size
            affine = (
                npg.napari_affine(self.poses[i])
                if (aligned and self.poses)
                else np.eye(4)
            )
            self.viewer.add_image(
                vol,
                name=f"vol:{s.name}",
                scale=(px, px, px),
                affine=affine,
                colormap="gray",
                blending="additive",
                opacity=0.5,
                visible=False,
            )
            s.drop_volume()

    def _fill_table(self) -> None:
        rows = stitch.interface_rows(self.result, self.dataset)
        cols = [
            "interface",
            "coarse_deg",
            "relative_deg",
            "match_frac",
            "incoherence_rho",
            "tangent_deg",
            "warp_ok",
            "qc_ok",
            "hybrid_deg",
            "hybrid_flag",
            "intensity_ok",
            "reasons",
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for cix, key in enumerate(cols):
                v = row.get(key, "")
                if isinstance(v, float):
                    v = f"{v:.3f}"
                self.table.setItem(r, cix, QTableWidgetItem(str(v)))
        self.table.resizeColumnsToContents()

    # ---- misc ------------------------------------------------------------ #
    def _warn(self, msg: str) -> None:
        QMessageBox.warning(self, "Stitch validator", msg)

    def _error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, f"{exc}\n\n{traceback.format_exc()}")


# =========================================================================== #
#  Coarse ground-truth recorder
# =========================================================================== #
class _PrefillWorker(QObject):
    """Background worker: runs the CPD-based hybrid coarse search per interface.

    Emits ``progress(k_done, n_total, raw_angle)`` after each interface so the
    UI can stream estimates in as they land (the operator gets the *current*
    interface's prefill as soon as that one finishes, rather than waiting for
    the whole stack). ``finished(angles)`` carries the final A–P-fused /
    continuity-resolved per-interface angles after the full pass; the per-step
    raw angles are an early approximation and may be revised by the final pass.
    """

    progress = Signal(int, int, float)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, coords_list: List[np.ndarray]):
        super().__init__()
        self._coords = coords_list
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            from pandorica.stitch.coarse.coarse_hybrid import hybrid_coarse

            def _cb(k: int, n: int, angle: float) -> None:
                # Best-effort cancellation: stop after the current interface
                # finishes (CPD itself is not interruptible mid-fit).
                if self._stop:
                    raise StopIteration
                self.progress.emit(k, n, angle)

            res = hybrid_coarse(
                self._coords, search_kwargs={"use_cpd": True}, progress=_cb
            )
            if not self._stop:
                self.finished.emit(list(res.angles))
        except StopIteration:
            self.finished.emit([])  # cancelled
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class CoarseGTWidget(QWidget):
    """Per-interface manual coarse alignment → ``coarse_gt.json``."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.dataset: Optional[Dataset] = None
        self.gt: Dict[str, dict] = {}
        self.k = 0  # current interface index
        self._fixed = None  # fixed top-face endpoints (xy)
        self._moving = None  # moving bottom-face endpoints (xy), un-posed
        self._center = np.zeros(2)
        self._grab = False
        # Prefill state. ``_hybrid_angles`` is the running per-interface estimate;
        # entries are filled in as the worker streams `progress` signals, and the
        # final A–P-fused / continuity-resolved values overwrite them on finish.
        self._hybrid_angles: Optional[List[Optional[float]]] = None
        self._prefill_thread: Optional[QThread] = None
        self._prefill_worker: Optional[_PrefillWorker] = None
        self._proj: Dict[int, tuple] = (
            {}
        )  # section index -> (top_maxproj, bottom_maxproj)
        self._proj_px = 1.0  # Å per projection pixel (pixel_size * proj downscale)
        self._build_ui()
        self.viewer.mouse_drag_callbacks.append(self._on_drag)

    # ---- UI -------------------------------------------------------------- #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        box = QGroupBox("Dataset")
        f = QFormLayout(box)
        self.folder_lbl = QLabel("(none)")
        self.folder_lbl.setWordWrap(True)
        browse = QPushButton("Browse folder…")
        browse.clicked.connect(self._browse)
        load_btn = QPushButton("Load dataset")
        load_btn.clicked.connect(self._load)
        f.addRow(browse, self.folder_lbl)
        f.addRow(load_btn)
        layout.addWidget(box)

        ibox = QGroupBox("Interface")
        il = QVBoxLayout(ibox)
        self.iface_combo = QComboBox()
        self.iface_combo.currentIndexChanged.connect(self._select_interface)
        il.addWidget(self.iface_combo)

        # angle
        ah = QHBoxLayout()
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.valueChanged.connect(self._angle_from_slider)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.valueChanged.connect(self._refresh_from_controls)
        ah.addWidget(QLabel("angle°"))
        ah.addWidget(self.angle_slider)
        ah.addWidget(self.angle_spin)
        il.addLayout(ah)

        # translation
        th = QHBoxLayout()
        self.tx_spin = QDoubleSpinBox()
        self.ty_spin = QDoubleSpinBox()
        for sp in (self.tx_spin, self.ty_spin):
            sp.setRange(-1e7, 1e7)
            sp.setDecimals(1)
            sp.valueChanged.connect(self._refresh_from_controls)
        th.addWidget(QLabel("tx"))
        th.addWidget(self.tx_spin)
        th.addWidget(QLabel("ty"))
        th.addWidget(self.ty_spin)
        il.addLayout(th)

        # Anisotropic scale: Sx and Sy independently, with a "Lock aspect" toggle
        # that mirrors one to the other (the common isotropic case stays a one-knob
        # experience). Real motivation is knife compression in plastic sections —
        # typically ~5-15% along the cutting axis, so Sx ≠ Sy is biologically
        # meaningful, not just a UI nicety. Sliders work in integer percent
        # (50..200 = 0.5..2.0); spinboxes carry the precise float.
        self.scale_lock_cb = QCheckBox("Lock aspect (isotropic scale)")
        self.scale_lock_cb.setChecked(True)
        # No explicit handler: the per-axis _on_sx/sy_changed slots read the
        # checkbox each time and mirror when checked.
        il.addWidget(self.scale_lock_cb)

        sxh = QHBoxLayout()
        self.sx_slider = QSlider(Qt.Horizontal)
        self.sx_slider.setRange(50, 200)
        self.sx_slider.setValue(100)
        self.sx_slider.valueChanged.connect(self._sx_from_slider)
        self.sx_spin = QDoubleSpinBox()
        self.sx_spin.setRange(0.5, 2.0)
        self.sx_spin.setSingleStep(0.01)
        self.sx_spin.setDecimals(3)
        self.sx_spin.setValue(1.0)
        self.sx_spin.valueChanged.connect(self._on_sx_changed)
        sxh.addWidget(QLabel("sx"))
        sxh.addWidget(self.sx_slider)
        sxh.addWidget(self.sx_spin)
        il.addLayout(sxh)

        syh = QHBoxLayout()
        self.sy_slider = QSlider(Qt.Horizontal)
        self.sy_slider.setRange(50, 200)
        self.sy_slider.setValue(100)
        self.sy_slider.valueChanged.connect(self._sy_from_slider)
        self.sy_spin = QDoubleSpinBox()
        self.sy_spin.setRange(0.5, 2.0)
        self.sy_spin.setSingleStep(0.01)
        self.sy_spin.setDecimals(3)
        self.sy_spin.setValue(1.0)
        self.sy_spin.valueChanged.connect(self._on_sy_changed)
        syh.addWidget(QLabel("sy"))
        syh.addWidget(self.sy_slider)
        syh.addWidget(self.sy_spin)
        il.addLayout(syh)

        self.grab_cb = QCheckBox("Grab: drag in canvas to translate moving face")
        self.grab_cb.toggled.connect(self._set_grab)
        il.addWidget(self.grab_cb)

        hh = QHBoxLayout()
        self.prefill_btn = QPushButton("Prefill from auto coarse")
        self.prefill_btn.clicked.connect(self._prefill_hybrid)
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset_controls)
        save = QPushButton("Save GT for interface")
        save.clicked.connect(self._save_interface)
        hh.addWidget(self.prefill_btn)
        hh.addWidget(reset)
        hh.addWidget(save)
        il.addLayout(hh)

        # Prefill progress (hidden until a CPD run is in flight). The Prefill
        # button stays usable so the operator can re-trigger the current
        # interface's spinbox once its angle has streamed in.
        self.prefill_progress = QProgressBar()
        self.prefill_progress.setRange(0, 1)
        self.prefill_progress.setValue(0)
        self.prefill_progress.setFormat("auto coarse: %v / %m  (%p%)")
        self.prefill_progress.setVisible(False)
        il.addWidget(self.prefill_progress)
        layout.addWidget(ibox)

        # boundary-face image overlay (Z-max projections of the touching faces)
        pbox = QGroupBox("Boundary-face images (orange=moving / blue=fixed)")
        pf = QFormLayout(pbox)
        self.nslice_spin = QSpinBox()
        self.nslice_spin.setRange(1, 200)
        self.nslice_spin.setValue(10)
        self.projds_spin = QSpinBox()
        self.projds_spin.setRange(1, 16)
        self.projds_spin.setValue(2)
        self.invert_z_cb = QCheckBox("Invert Z (swap top/bottom)")
        load_proj = QPushButton("Load face projections (slow)")
        load_proj.clicked.connect(self._load_projections)
        pf.addRow("# boundary slices", self.nslice_spin)
        pf.addRow("Projection XY downscale", self.projds_spin)
        pf.addRow(self.invert_z_cb)
        pf.addRow(load_proj)
        layout.addWidget(pbox)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(120)
        layout.addWidget(self.status)
        layout.addStretch()

    # ---- load ------------------------------------------------------------ #
    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if d:
            self.folder_lbl.setText(d)

    def _load(self) -> None:
        folder = self.folder_lbl.text()
        if not os.path.isdir(folder):
            self._log("Pick a valid dataset folder first.")
            return
        try:
            self.dataset = load_dataset(folder)
        except Exception as e:  # noqa: BLE001
            self._log(f"Load failed: {e}")
            return
        # New dataset → discard any in-flight or cached auto-coarse from the
        # previous one (interface counts and section identities won't match).
        self._cancel_prefill()
        self._hybrid_angles = None
        self.gt = self._read_gt()
        self.iface_combo.blockSignals(True)
        self.iface_combo.clear()
        for k in range(len(self.dataset) - 1):
            self.iface_combo.addItem(self.dataset.interface_label(k))
        self.iface_combo.blockSignals(False)
        if len(self.dataset) >= 2:
            self.iface_combo.setCurrentIndex(0)
            self._select_interface(0)
        self._log(
            f"Loaded {len(self.dataset)} sections from {folder}\n"
            f"Existing GT entries: {len(self.gt)}"
        )

    # ---- interface selection --------------------------------------------- #
    def _select_interface(self, k: int) -> None:
        if self.dataset is None or k < 0 or k + 1 >= len(self.dataset):
            return
        self.k = k
        coords = self.dataset.coords_list()
        # Convention (core): top(n)=high-Z fixed; bottom(n+1)=low-Z moving.
        fixed_ep = geo.face_endpoints(coords[k], "top")
        moving_ep = geo.face_endpoints(coords[k + 1], "bottom")
        self._fixed = geo.endpoints_xy(fixed_ep)
        self._moving = geo.endpoints_xy(moving_ep)
        self._center = self._moving.mean(0) if len(self._moving) else np.zeros(2)
        self._clear_layers()
        self._add_face_images(k)  # below the endpoints, so points stay visible
        if len(self._fixed):
            self.viewer.add_points(
                self._fixed[:, [1, 0]],  # (x,y) -> (y,x) for 2-D view
                name="gt:fixed",
                size=self._pt_size(),
                face_color="#7fb3ff",
                border_color="#1f6fff",
            )
        # restore stored / zeroed controls, then draw moving.
        # Backward-compat: old GT JSONs only have "scale" (isotropic); read sx/sy
        # when present, else fall back to the legacy single scale, else 1.0.
        rec = self.gt.get(self.dataset.interface_label(k), {})
        legacy = rec.get("scale", 1.0)
        self._set_controls(
            rec.get("angle", 0.0),
            rec.get("tx", 0.0),
            rec.get("ty", 0.0),
            rec.get("sx", legacy),
            rec.get("sy", legacy),
        )
        self.viewer.dims.ndisplay = 2
        self.viewer.reset_view()
        self._log_interface()

    def _posed_moving(self) -> np.ndarray:
        if self._moving is None or len(self._moving) == 0:
            return np.empty((0, 2))
        return npg.apply_anisotropic_xy(
            self._moving,
            self.angle_spin.value(),
            self.tx_spin.value(),
            self.ty_spin.value(),
            self.sx_spin.value(),
            self.sy_spin.value(),
            self._center,
        )

    def _current_affine_2d(self) -> np.ndarray:
        """Napari 3×3 affine (y, x order) reflecting the current control values."""
        return npg.napari_affine_2d_anisotropic(
            self.angle_spin.value(),
            self.tx_spin.value(),
            self.ty_spin.value(),
            self.sx_spin.value(),
            self.sy_spin.value(),
            self._center,
        )

    def _refresh_moving(self) -> None:
        xy = self._posed_moving()
        name = "gt:moving"
        data = xy[:, [1, 0]] if len(xy) else np.empty((0, 2))  # (x,y)->(y,x)
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_points(
                data,
                name=name,
                size=self._pt_size(),
                face_color="#f28e2b",
                border_color="#d4691a",
                symbol="x",
            )
        # drive the moving boundary-face image with the same transform (no resampling)
        if "gt:img-moving" in self.viewer.layers:
            self.viewer.layers["gt:img-moving"].affine = self._current_affine_2d()

    # ---- boundary-face image overlay ------------------------------------- #
    def _load_projections(self) -> None:
        """
        Load each volume once, Z-max-project its top + bottom boundary slabs.

        Heavy: reads the full (multi-GB) volume per section, but only keeps the two
        small 2-D projections (top + bottom), then drops the volume. Refreshes the
        current interface so the overlay appears.
        """
        if self.dataset is None:
            self._log("Load a dataset first.")
            return
        n_slices = self.nslice_spin.value()
        ds = self.projds_spin.value()
        inv = self.invert_z_cb.isChecked()
        self._proj = {}
        px = None
        for i, s in enumerate(self.dataset.sections):
            if not s.has_volume():
                continue
            self._log(f"Loading volume {s.name} ({i + 1}/{len(self.dataset)})…")
            QApplication.processEvents()
            try:
                vol = s.load_volume(downscale=1)
            except Exception as e:  # noqa: BLE001
                self._log(f"  {s.name}: load failed ({e})")
                continue
            top = np.asarray(geo.zmax_face(vol, "top", n_slices, inv))[::ds, ::ds]
            bot = np.asarray(geo.zmax_face(vol, "bottom", n_slices, inv))[::ds, ::ds]
            self._proj[i] = (top, bot)
            px = s.pixel_size
            s.drop_volume()
        self._proj_px = (px or 1.0) * ds
        if not self._proj:
            self._log("No section volumes found to project.")
            return
        self._select_interface(self.k)
        self._log(
            f"Loaded face projections for {len(self._proj)} sections "
            f"({n_slices}-slice Z-max, XY/{ds}). "
            f"Blue=fixed top(n); orange=moving bottom(n+1)."
        )

    def _add_face_images(self, k: int) -> None:
        """Add the fixed (top of n) + moving (bottom of n+1) face projections, if loaded."""
        if not self._proj:
            return
        px = self._proj_px
        top = self._proj.get(k, (None, None))[0]
        bot = self._proj.get(k + 1, (None, None))[1]
        if top is not None:
            self.viewer.add_image(
                top,
                name="gt:img-fixed",
                scale=(px, px),
                colormap="bop blue",
                blending="additive",
                opacity=0.9,
            )
        if bot is not None:
            self.viewer.add_image(
                bot,
                name="gt:img-moving",
                scale=(px, px),
                colormap="bop orange",
                blending="additive",
                opacity=0.9,
                affine=self._current_affine_2d(),
            )

    # ---- control wiring -------------------------------------------------- #
    def _angle_from_slider(self, v: int) -> None:
        if abs(self.angle_spin.value() - v) > 0.5:
            self.angle_spin.setValue(float(v))  # triggers refresh

    def _sx_from_slider(self, v: int) -> None:
        """Slider holds 100·sx (integer percent); push to spinbox if changed enough."""
        s = v / 100.0
        if abs(self.sx_spin.value() - s) > 0.005:
            self.sx_spin.setValue(s)  # triggers _on_sx_changed

    def _sy_from_slider(self, v: int) -> None:
        s = v / 100.0
        if abs(self.sy_spin.value() - s) > 0.005:
            self.sy_spin.setValue(s)

    def _on_sx_changed(self, *_):
        """Mirror to sy when 'Lock aspect' is on, then trigger a redraw."""
        if self.scale_lock_cb.isChecked():
            v = self.sx_spin.value()
            self.sy_spin.blockSignals(True)
            self.sy_slider.blockSignals(True)
            self.sy_spin.setValue(v)
            self.sy_slider.setValue(int(round(v * 100)))
            self.sy_spin.blockSignals(False)
            self.sy_slider.blockSignals(False)
        self._refresh_from_controls()

    def _on_sy_changed(self, *_):
        if self.scale_lock_cb.isChecked():
            v = self.sy_spin.value()
            self.sx_spin.blockSignals(True)
            self.sx_slider.blockSignals(True)
            self.sx_spin.setValue(v)
            self.sx_slider.setValue(int(round(v * 100)))
            self.sx_spin.blockSignals(False)
            self.sx_slider.blockSignals(False)
        self._refresh_from_controls()

    def _refresh_from_controls(self, *_):
        # Sync the integer sliders to their float spinboxes (one-way: spin → slider).
        a = self.angle_spin.value()
        if self.angle_slider.value() != int(round(a)):
            self.angle_slider.blockSignals(True)
            self.angle_slider.setValue(int(round(a)))
            self.angle_slider.blockSignals(False)
        for spin, slider in (
            (self.sx_spin, self.sx_slider),
            (self.sy_spin, self.sy_slider),
        ):
            s_pct = int(round(spin.value() * 100))
            if slider.value() != s_pct:
                slider.blockSignals(True)
                slider.setValue(s_pct)
                slider.blockSignals(False)
        self._refresh_moving()
        self._log_interface()

    def _set_controls(
        self,
        angle: float,
        tx: float,
        ty: float,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        controls = (
            self.angle_spin,
            self.tx_spin,
            self.ty_spin,
            self.angle_slider,
            self.sx_spin,
            self.sx_slider,
            self.sy_spin,
            self.sy_slider,
        )
        for w in controls:
            w.blockSignals(True)
        self.angle_spin.setValue(float(angle))
        self.angle_slider.setValue(int(round(angle)))
        self.tx_spin.setValue(float(tx))
        self.ty_spin.setValue(float(ty))
        self.sx_spin.setValue(float(sx))
        self.sx_slider.setValue(int(round(sx * 100)))
        self.sy_spin.setValue(float(sy))
        self.sy_slider.setValue(int(round(sy * 100)))
        for w in controls:
            w.blockSignals(False)
        self._refresh_moving()

    def _reset_controls(self) -> None:
        self._set_controls(0.0, 0.0, 0.0, 1.0, 1.0)
        self._log_interface()

    def _set_grab(self, on: bool) -> None:
        """Toggle grab mode; disable camera panning so a drag moves the face, not the view."""
        self._grab = on
        try:
            self.viewer.camera.mouse_pan = not on
        except Exception:  # noqa: BLE001
            pass  # older napari without camera.mouse_pan — drag still works, view may pan

    def _on_drag(self, viewer, event):
        if not self._grab or self._moving is None:
            return
        last = np.array(event.position[-2:])  # (y, x) world coords of the 2-D view
        yield
        while event.type == "mouse_move":
            cur = np.array(event.position[-2:])
            dy, dx = cur - last
            self.tx_spin.blockSignals(True)
            self.ty_spin.blockSignals(True)
            self.tx_spin.setValue(self.tx_spin.value() + float(dx))
            self.ty_spin.setValue(self.ty_spin.value() + float(dy))
            self.tx_spin.blockSignals(False)
            self.ty_spin.blockSignals(False)
            last = cur
            self._refresh_moving()
            yield
        self._log_interface()

    # ---- auto-coarse prefill --------------------------------------------- #
    def _prefill_hybrid(self) -> None:
        """Prefill the current interface's angle from the auto coarse search.

        On large stacks the CPD-based search takes minutes, so it runs in a
        background ``QThread`` (:class:`_PrefillWorker`). The UI stays
        responsive: the operator can keep manually aligning while estimates
        stream in. The current interface's spinbox auto-updates the moment its
        estimate arrives, and any later interface gets the same treatment when
        the operator navigates to it.
        """
        if self.dataset is None:
            return
        # Already-computed angles → instant fill, no recompute.
        if (
            self._hybrid_angles is not None
            and self.k < len(self._hybrid_angles)
            and self._hybrid_angles[self.k] is not None
        ):
            self._apply_prefill(self.k, float(self._hybrid_angles[self.k]))
            return
        # A computation is already in flight; just wait for it.
        if self._prefill_thread is not None and self._prefill_thread.isRunning():
            self._log(
                "Auto coarse already running — the current interface will fill "
                "the moment its estimate arrives. You can keep aligning others."
            )
            return
        # Cold start: spin up a worker.
        n_iface = max(0, len(self.dataset) - 1)
        if n_iface == 0:
            return
        self._hybrid_angles = [None] * n_iface
        self.prefill_progress.setRange(0, n_iface)
        self.prefill_progress.setValue(0)
        self.prefill_progress.setVisible(True)

        thread = QThread(self)
        worker = _PrefillWorker(self.dataset.coords_list())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_prefill_progress)
        worker.finished.connect(self._on_prefill_finished)
        worker.failed.connect(self._on_prefill_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._prefill_thread = thread
        self._prefill_worker = worker
        self._log("Auto coarse: starting CPD search in background…")
        thread.start()

    def _on_prefill_progress(self, k: int, n: int, angle: float) -> None:
        """Streamed: interface ``k`` of ``n`` just finished its raw estimate."""
        if self._hybrid_angles is None or k >= len(self._hybrid_angles):
            return
        self._hybrid_angles[k] = float(angle)
        self.prefill_progress.setValue(k + 1)
        if k == self.k:
            # Operator is sitting on this interface — fill its spinbox now.
            self._apply_prefill(k, float(angle))
        else:
            self._log(
                f"Auto coarse: interface {k + 1}/{n} ready "
                f"(raw angle {angle:+.2f}°). Move to it to fill its spinbox."
            )

    def _on_prefill_finished(self, angles) -> None:
        """Full pass complete — overwrite raw estimates with A–P-fused / continuity-resolved angles."""
        self.prefill_progress.setVisible(False)
        if not angles:
            self._log("Auto coarse: cancelled.")
            return
        self._hybrid_angles = [float(a) for a in angles]
        # If the operator's still on an interface we've already filled, re-apply
        # the final (resolved) value in case the continuity pass changed it.
        if self.k < len(self._hybrid_angles):
            self._apply_prefill(self.k, float(self._hybrid_angles[self.k]))
        self._log(f"Auto coarse: done ({len(angles)} interface(s)).")

    def _on_prefill_failed(self, msg: str) -> None:
        self.prefill_progress.setVisible(False)
        self._hybrid_angles = None
        self._log(f"Auto coarse failed:\n{msg}")

    def _apply_prefill(self, k: int, angle: float) -> None:
        """Push a computed angle into the spinbox without touching tx/ty/sx/sy."""
        if k != self.k:
            return
        self._set_controls(
            angle,
            self.tx_spin.value(),
            self.ty_spin.value(),
            self.sx_spin.value(),
            self.sy_spin.value(),
        )
        self._log_interface()

    def _cancel_prefill(self) -> None:
        """Stop any in-flight prefill worker. Best-effort: CPD is not interruptible
        mid-fit, so we stop at the next interface boundary."""
        if self._prefill_worker is not None:
            self._prefill_worker.stop()
        if self._prefill_thread is not None and self._prefill_thread.isRunning():
            self._prefill_thread.quit()
            self._prefill_thread.wait(2000)
        self._prefill_thread = None
        self._prefill_worker = None
        self.prefill_progress.setVisible(False)

    # ---- save ------------------------------------------------------------ #
    def _gt_path(self) -> str:
        return join(self.dataset.folder, GT_FILENAME)

    def _read_gt(self) -> Dict[str, dict]:
        p = self._gt_path()
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _save_interface(self) -> None:
        if self.dataset is None:
            return
        label = self.dataset.interface_label(self.k)
        sx = self.sx_spin.value()
        sy = self.sy_spin.value()
        # "scale" stays in the JSON as the isotropic mean for callers that
        # haven't been updated to read sx/sy; sx/sy are the source of truth.
        self.gt[label] = {
            "angle": round(self.angle_spin.value(), 4),
            "tx": round(self.tx_spin.value(), 3),
            "ty": round(self.ty_spin.value(), 3),
            "sx": round(sx, 4),
            "sy": round(sy, 4),
            "scale": round((sx + sy) / 2.0, 4),
            "n_fixed": int(len(self._fixed)) if self._fixed is not None else 0,
            "n_moving": int(len(self._moving)) if self._moving is not None else 0,
        }
        with open(self._gt_path(), "w") as f:
            json.dump(self.gt, f, indent=2)
        self._log(f"Saved GT[{label}] = {self.gt[label]}\n→ {self._gt_path()}")

    # ---- helpers --------------------------------------------------------- #
    def _pt_size(self) -> float:
        pts = np.vstack(
            [p for p in (self._fixed, self._moving) if p is not None and len(p)]
        )
        return max(float(np.ptp(pts)) / 80.0, 1.0) if len(pts) else 5.0

    def _clear_layers(self) -> None:
        for lyr in list(self.viewer.layers):
            if lyr.name.startswith("gt:"):
                self.viewer.layers.remove(lyr)

    def _log_interface(self) -> None:
        if self.dataset is None:
            return
        label = self.dataset.interface_label(self.k)
        est = ""
        if (
            self._hybrid_angles is not None
            and self.k < len(self._hybrid_angles)
            and self._hybrid_angles[self.k] is not None
        ):
            est = f"  | C coarse: {self._hybrid_angles[self.k]:+.1f}°"
        stored = self.gt.get(label)
        self._log(
            f"Interface {label}{est}\n"
            f"  fixed pts={0 if self._fixed is None else len(self._fixed)}  "
            f"moving pts={0 if self._moving is None else len(self._moving)}\n"
            f"  current: angle={self.angle_spin.value():+.2f}°  "
            f"tx={self.tx_spin.value():.1f}  ty={self.ty_spin.value():.1f}  "
            f"sx={self.sx_spin.value():.3f}  sy={self.sy_spin.value():.3f}\n"
            f"  saved:   {stored if stored else '(none)'}"
        )

    def _log(self, msg: str) -> None:
        self.status.setPlainText(msg)
