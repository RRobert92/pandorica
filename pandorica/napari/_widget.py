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
* :class:`CoarseGTWidget` — per-interface manual coarse alignment: rotate (slider)
  and translate (spinbox or mouse-drag) the moving bottom-face onto the fixed
  top-face and save ``{angle, tx, ty}`` per interface to ``coarse_gt.json``.

UI only — all loading/stitching/IO lives in :mod:`._io`, :mod:`._stitch`,
:mod:`._geometry`.
"""

import json
import os
import traceback
from os.path import join
from typing import Dict, List, Optional

import numpy as np
from qtpy.QtCore import Qt
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
        self.load_volumes_cb = QCheckBox("Load volumes (large)")
        self.display_ds = QSpinBox()
        self.display_ds.setRange(1, 32)
        self.display_ds.setValue(4)
        load_btn = QPushButton("Load dataset")
        load_btn.clicked.connect(self._load)
        f.addRow(browse, self.folder_lbl)
        f.addRow("Display downscale", self.display_ds)
        f.addRow(self.load_volumes_cb)
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
        self._clear_layers(prefixes=(prefix,))
        for i, c in enumerate(coords_list):
            pts = npg.coords_to_points_zyx(c)
            if len(pts) == 0:
                continue
            color = npg.section_color(i)
            self.viewer.add_points(
                pts,
                name=f"{prefix}:{self.dataset.names[i]}",
                size=max(np.ptp(pts[:, 1:]) / 200.0, 1.0) if len(pts) else 1.0,
                face_color=color,
                border_color=color,
                opacity=0.9 if aligned else 0.35,
                visible=aligned,  # show aligned by default, hide raw until toggled
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
        self._hybrid_angles = None
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

        self.grab_cb = QCheckBox("Grab: drag in canvas to translate moving face")
        self.grab_cb.toggled.connect(self._set_grab)
        il.addWidget(self.grab_cb)

        hh = QHBoxLayout()
        prefill = QPushButton("Prefill from auto coarse")
        prefill.clicked.connect(self._prefill_hybrid)
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset_controls)
        save = QPushButton("Save GT for interface")
        save.clicked.connect(self._save_interface)
        hh.addWidget(prefill)
        hh.addWidget(reset)
        hh.addWidget(save)
        il.addLayout(hh)
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
        # restore stored / zeroed controls, then draw moving
        rec = self.gt.get(self.dataset.interface_label(k), {})
        self._set_controls(
            rec.get("angle", 0.0), rec.get("tx", 0.0), rec.get("ty", 0.0)
        )
        self.viewer.dims.ndisplay = 2
        self.viewer.reset_view()
        self._log_interface()

    def _posed_moving(self) -> np.ndarray:
        if self._moving is None or len(self._moving) == 0:
            return np.empty((0, 2))
        pose = geo.centroid_pose(
            self.angle_spin.value(),
            self.tx_spin.value(),
            self.ty_spin.value(),
            self._center,
        )
        from pandorica.stitch.transform.solver import apply_pose

        return apply_pose(pose, self._moving)

    def _current_pose(self):
        return geo.centroid_pose(
            self.angle_spin.value(),
            self.tx_spin.value(),
            self.ty_spin.value(),
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
        # drive the moving boundary-face image with the same pose (no resampling)
        if "gt:img-moving" in self.viewer.layers:
            self.viewer.layers["gt:img-moving"].affine = npg.napari_affine_2d(
                self._current_pose()
            )

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
                affine=npg.napari_affine_2d(self._current_pose()),
            )

    # ---- control wiring -------------------------------------------------- #
    def _angle_from_slider(self, v: int) -> None:
        if abs(self.angle_spin.value() - v) > 0.5:
            self.angle_spin.setValue(float(v))  # triggers refresh

    def _refresh_from_controls(self, *_):
        a = self.angle_spin.value()
        if self.angle_slider.value() != int(round(a)):
            self.angle_slider.blockSignals(True)
            self.angle_slider.setValue(int(round(a)))
            self.angle_slider.blockSignals(False)
        self._refresh_moving()
        self._log_interface()

    def _set_controls(self, angle: float, tx: float, ty: float) -> None:
        for w in (self.angle_spin, self.tx_spin, self.ty_spin, self.angle_slider):
            w.blockSignals(True)
        self.angle_spin.setValue(float(angle))
        self.angle_slider.setValue(int(round(angle)))
        self.tx_spin.setValue(float(tx))
        self.ty_spin.setValue(float(ty))
        for w in (self.angle_spin, self.tx_spin, self.ty_spin, self.angle_slider):
            w.blockSignals(False)
        self._refresh_moving()

    def _reset_controls(self) -> None:
        self._set_controls(0.0, 0.0, 0.0)
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
        if self.dataset is None:
            return
        if self._hybrid_angles is None:
            # Use the CPD multi-seed search — it is the pipeline's default
            # (cpd_coarse=True) AND ~20× faster than the gated sweep here.
            self._log("Computing auto coarse (CPD, one-time)…")
            QApplication.processEvents()
            try:
                from pandorica.stitch.coarse.coarse_hybrid import (
                    hybrid_coarse,
                )

                self._hybrid_angles = hybrid_coarse(
                    self.dataset.coords_list(),
                    search_kwargs={"use_cpd": True},
                ).angles
            except Exception as e:  # noqa: BLE001
                self._log(f"Hybrid coarse failed: {e}")
                return
        if self.k < len(self._hybrid_angles):
            self._set_controls(
                float(self._hybrid_angles[self.k]),
                self.tx_spin.value(),
                self.ty_spin.value(),
            )
            self._log_interface()

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
        self.gt[label] = {
            "angle": round(self.angle_spin.value(), 4),
            "tx": round(self.tx_spin.value(), 3),
            "ty": round(self.ty_spin.value(), 3),
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
        if self._hybrid_angles is not None and self.k < len(self._hybrid_angles):
            est = f"  | C coarse: {self._hybrid_angles[self.k]:+.1f}°"
        stored = self.gt.get(label)
        self._log(
            f"Interface {label}{est}\n"
            f"  fixed pts={0 if self._fixed is None else len(self._fixed)}  "
            f"moving pts={0 if self._moving is None else len(self._moving)}\n"
            f"  current: angle={self.angle_spin.value():+.2f}°  "
            f"tx={self.tx_spin.value():.1f}  ty={self.ty_spin.value():.1f}\n"
            f"  saved:   {stored if stored else '(none)'}"
        )

    def _log(self, msg: str) -> None:
        self.status.setPlainText(msg)
