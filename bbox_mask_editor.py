import os
import copy
import pickle
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R


def load_ply_xyz(ply_path: str) -> np.ndarray:
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1
    ).astype(np.float32)
    return xyz


@dataclass
class EditableObject:
    name: str
    indices: np.ndarray
    color: np.ndarray
    bbox_min_world: np.ndarray
    bbox_max_world: np.ndarray

    def copy_mask(self) -> np.ndarray:
        return self.indices.copy()

    @property
    def center_world(self) -> np.ndarray:
        return 0.5 * (self.bbox_min_world + self.bbox_max_world)

    @property
    def extent_world(self) -> np.ndarray:
        return np.maximum(self.bbox_max_world - self.bbox_min_world, 1e-6)

    def translate_bbox(self, delta: np.ndarray) -> None:
        delta = delta.astype(np.float32)
        self.bbox_min_world += delta
        self.bbox_max_world += delta

    def scale_bbox_axis(self, axis: int, delta: float) -> None:
        c = self.center_world.copy()
        e = self.extent_world.copy()
        e[axis] = max(1e-4, e[axis] + delta)
        half = 0.5 * e
        self.bbox_min_world = (c - half).astype(np.float32)
        self.bbox_max_world = (c + half).astype(np.float32)

    def set_bbox_from_points(self, pts_world: np.ndarray, pad: float = 0.0) -> None:
        if pts_world.shape[0] == 0:
            return
        mn = pts_world.min(axis=0).astype(np.float32)
        mx = pts_world.max(axis=0).astype(np.float32)
        pad_vec = np.array([pad, pad, pad], dtype=np.float32)
        self.bbox_min_world = mn - pad_vec
        self.bbox_max_world = mx + pad_vec


class InteractiveGaussianGUI:
    ALIGNMENT_KEY = "_editor_scene_alignment"

    def __init__(
        self,
        xyz_all_orig: np.ndarray,
        editing_modifier: dict,
        modifier_path: str,
        point_size: int = 3,
        move_step: float = 0.05,
        scale_step: float = 0.05,
        rot_deg: float = 8.0,
        fine_rot_deg: float = 1.0,
        bbox_pad: float = 0.0,
    ):
        self.xyz_all_orig = xyz_all_orig.copy()
        self.editing_modifier = editing_modifier
        self.loaded_modifier_original = copy.deepcopy(editing_modifier)
        self.modifier_path = modifier_path

        self.point_size = point_size
        self.move_step = move_step
        self.scale_step = scale_step
        self.rot_deg = rot_deg
        self.fine_rot_deg = fine_rot_deg
        self.default_bbox_pad = bbox_pad

        self.base_env_color = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        self.object_palette = [
            np.array([1.0, 0.2, 0.2], dtype=np.float32),
            np.array([0.2, 1.0, 0.2], dtype=np.float32),
            np.array([0.2, 0.45, 1.0], dtype=np.float32),
            np.array([1.0, 0.8, 0.2], dtype=np.float32),
            np.array([0.9, 0.2, 0.9], dtype=np.float32),
            np.array([0.2, 0.9, 0.9], dtype=np.float32),
            np.array([1.0, 0.5, 0.2], dtype=np.float32),
            np.array([0.6, 0.35, 1.0], dtype=np.float32),
        ]

        self.scene_R = np.eye(3, dtype=np.float32)
        self.scene_t = np.zeros(3, dtype=np.float32)
        self._load_persisted_alignment()

        self.objects: List[EditableObject] = []
        self.original_masks: List[np.ndarray] = []
        self.selected_idx = 0
        self.show_boxes = True
        self.show_only_selected = False
        self.bbox_select_mode = "replace"

        self._build_objects_from_current_modifier()

        self.window = gui.Application.instance.create_window(
            "Interactive Gaussian BBox / Mask Editor", 1720, 1060
        )
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.05, 0.05, 0.05, 1.0])

        self.panel = gui.Vert(0, gui.Margins(8, 8, 8, 8))
        self.info_label = gui.Label("")
        self.panel.add_child(self.info_label)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = float(self.point_size)

        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 2.0

        self._build_controls()

        self._added_geometry_names = set()
        self._rebuild_scene()

        pcd = self._build_point_cloud()
        bbox = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

        self._update_info(print_console=True)

        self.window.set_on_close(self._on_close)
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.window.set_needs_layout()
        )

    # ------------------------------------------------------------------
    # Alignment persistence
    # ------------------------------------------------------------------

    def _load_persisted_alignment(self) -> None:
        block = self.editing_modifier.get(self.ALIGNMENT_KEY, None)
        if not isinstance(block, dict):
            return

        try:
            scene_R = np.asarray(block.get("scene_R", np.eye(3)), dtype=np.float32)
            scene_t = np.asarray(block.get("scene_t", np.zeros(3)), dtype=np.float32)

            if scene_R.shape != (3, 3):
                raise ValueError("scene_R must be 3x3")
            if scene_t.shape != (3,):
                raise ValueError("scene_t must be length 3")

            self.scene_R = scene_R.copy()
            self.scene_t = scene_t.copy()
            print("Loaded saved coordinate alignment from editing_modifier.pkl")
            print("scene_R:\n", self.scene_R)
            print("scene_t:", self.scene_t)
        except Exception as e:
            print(f"Failed to load saved alignment, using identity. Error: {e}")
            self.scene_R = np.eye(3, dtype=np.float32)
            self.scene_t = np.zeros(3, dtype=np.float32)

    def _write_alignment_to_modifier_dict(self, modifier: Dict[str, Any]) -> None:
        modifier[self.ALIGNMENT_KEY] = {
            "scene_R": self.scene_R.astype(np.float32),
            "scene_t": self.scene_t.astype(np.float32),
        }

    def _save_alignment_only_to_file(self) -> None:
        with open(self.modifier_path, "rb") as f:
            modifier = pickle.load(f)

        self._write_alignment_to_modifier_dict(modifier)

        with open(self.modifier_path, "wb") as f:
            pickle.dump(modifier, f)

        self.editing_modifier = modifier
        print(f"Saved alignment only to: {self.modifier_path}")
        print("scene_R:\n", self.scene_R)
        print("scene_t:", self.scene_t)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _apply_scene_transform_points(self, pts_orig: np.ndarray) -> np.ndarray:
        return (self.scene_R @ pts_orig.T).T + self.scene_t[None, :]

    def _invert_scene_transform_points(self, pts_aligned: np.ndarray) -> np.ndarray:
        return (self.scene_R.T @ (pts_aligned - self.scene_t[None, :]).T).T

    # ------------------------------------------------------------------
    # Scene / object setup
    # ------------------------------------------------------------------

    def _build_objects_from_current_modifier(self) -> None:
        self.objects.clear()
        self.original_masks.clear()

        obj_list = self.editing_modifier.get("objects", [])
        if len(obj_list) == 0:
            raise ValueError("No objects found in editing_modifier.pkl")

        total_n = self.xyz_all_orig.shape[0]

        for i, obj in enumerate(obj_list):
            mask = np.asarray(obj["affected_gaussian_idx"]).astype(bool)
            if mask.shape[0] != total_n:
                raise ValueError(
                    f"Object {i} mask length {mask.shape[0]} does not match {total_n}"
                )

            obj_xyz_orig = self.xyz_all_orig[mask].copy()

            saved_bbox = obj.get("bbox_editor", None)
            use_saved_bbox = (
                isinstance(saved_bbox, dict)
                and "bbox_min_world" in saved_bbox
                and "bbox_max_world" in saved_bbox
            )

            if use_saved_bbox:
                bbox_min = np.asarray(saved_bbox["bbox_min_world"], dtype=np.float32)
                bbox_max = np.asarray(saved_bbox["bbox_max_world"], dtype=np.float32)
            else:
                if obj_xyz_orig.shape[0] > 0:
                    obj_xyz_aligned = self._apply_scene_transform_points(obj_xyz_orig)
                    bbox_min = obj_xyz_aligned.min(axis=0).astype(np.float32)
                    bbox_max = obj_xyz_aligned.max(axis=0).astype(np.float32)
                else:
                    aligned_all = self._apply_scene_transform_points(self.xyz_all_orig)
                    center = aligned_all.mean(axis=0).astype(np.float32)
                    bbox_min = center - 0.05
                    bbox_max = center + 0.05

            editable = EditableObject(
                name=obj.get("name", f"object_{i}"),
                indices=mask.copy(),
                color=self.object_palette[i % len(self.object_palette)].copy(),
                bbox_min_world=bbox_min.copy(),
                bbox_max_world=bbox_max.copy(),
            )
            self.objects.append(editable)
            self.original_masks.append(mask.copy())

        if len(self.objects) == 0:
            raise ValueError("All object masks were empty.")

    def _recompute_all_bboxes_from_masks(self) -> None:
        xyz_all_aligned = self._build_current_aligned_xyz()
        for obj in self.objects:
            pts = xyz_all_aligned[obj.indices]
            if pts.shape[0] > 0:
                obj.set_bbox_from_points(pts, pad=0.0)

    # ------------------------------------------------------------------
    # GUI controls
    # ------------------------------------------------------------------

    def _build_controls(self):
        self.panel.add_child(gui.Label("Object Selection"))

        row = gui.Horiz(4)
        btn_prev = gui.Button("Prev")
        btn_next = gui.Button("Next")
        btn_prev.set_on_clicked(self._on_prev)
        btn_next.set_on_clicked(self._on_next)
        row.add_child(btn_prev)
        row.add_child(btn_next)
        self.panel.add_child(row)

        btn_only = gui.Button("Toggle Only Selected")
        btn_only.set_on_clicked(self._on_toggle_only_selected)
        self.panel.add_child(btn_only)

        self.panel.add_child(gui.Label("BBox Translation"))

        btn_tx_p = gui.Button("+X")
        btn_tx_p.set_on_clicked(lambda: self._on_translate_bbox(np.array([ self.move_step, 0.0, 0.0], dtype=np.float32)))
        self.panel.add_child(btn_tx_p)

        btn_tx_n = gui.Button("-X")
        btn_tx_n.set_on_clicked(lambda: self._on_translate_bbox(np.array([-self.move_step, 0.0, 0.0], dtype=np.float32)))
        self.panel.add_child(btn_tx_n)

        btn_ty_p = gui.Button("+Y")
        btn_ty_p.set_on_clicked(lambda: self._on_translate_bbox(np.array([0.0,  self.move_step, 0.0], dtype=np.float32)))
        self.panel.add_child(btn_ty_p)

        btn_ty_n = gui.Button("-Y")
        btn_ty_n.set_on_clicked(lambda: self._on_translate_bbox(np.array([0.0, -self.move_step, 0.0], dtype=np.float32)))
        self.panel.add_child(btn_ty_n)

        btn_tz_p = gui.Button("+Z")
        btn_tz_p.set_on_clicked(lambda: self._on_translate_bbox(np.array([0.0, 0.0,  self.move_step], dtype=np.float32)))
        self.panel.add_child(btn_tz_p)

        btn_tz_n = gui.Button("-Z")
        btn_tz_n.set_on_clicked(lambda: self._on_translate_bbox(np.array([0.0, 0.0, -self.move_step], dtype=np.float32)))
        self.panel.add_child(btn_tz_n)

        self.panel.add_child(gui.Label("BBox Scaling"))

        btn_sx_p = gui.Button("Wider X")
        btn_sx_p.set_on_clicked(lambda: self._on_scale_bbox(0, +1.0))
        self.panel.add_child(btn_sx_p)

        btn_sx_n = gui.Button("Narrower X")
        btn_sx_n.set_on_clicked(lambda: self._on_scale_bbox(0, -1.0))
        self.panel.add_child(btn_sx_n)

        btn_sy_p = gui.Button("Wider Y")
        btn_sy_p.set_on_clicked(lambda: self._on_scale_bbox(1, +1.0))
        self.panel.add_child(btn_sy_p)

        btn_sy_n = gui.Button("Narrower Y")
        btn_sy_n.set_on_clicked(lambda: self._on_scale_bbox(1, -1.0))
        self.panel.add_child(btn_sy_n)

        btn_sz_p = gui.Button("Wider Z")
        btn_sz_p.set_on_clicked(lambda: self._on_scale_bbox(2, +1.0))
        self.panel.add_child(btn_sz_p)

        btn_sz_n = gui.Button("Narrower Z")
        btn_sz_n.set_on_clicked(lambda: self._on_scale_bbox(2, -1.0))
        self.panel.add_child(btn_sz_n)

        self.panel.add_child(gui.Label("BBox -> Mask"))

        btn_mode_replace = gui.Button("Mode: Replace")
        btn_mode_replace.set_on_clicked(lambda: self._set_bbox_mode("replace"))
        self.panel.add_child(btn_mode_replace)

        btn_mode_add = gui.Button("Mode: Add")
        btn_mode_add.set_on_clicked(lambda: self._set_bbox_mode("add"))
        self.panel.add_child(btn_mode_add)

        btn_mode_remove = gui.Button("Mode: Remove")
        btn_mode_remove.set_on_clicked(lambda: self._set_bbox_mode("remove"))
        self.panel.add_child(btn_mode_remove)

        btn_apply_bbox = gui.Button("Apply BBox To Mask")
        btn_apply_bbox.set_on_clicked(self._on_apply_bbox_to_mask)
        self.panel.add_child(btn_apply_bbox)

        self.panel.add_child(gui.Label("BBox from Current Mask"))
        self.bbox_pad_edit = gui.TextEdit()
        self.bbox_pad_edit.text_value = str(self.default_bbox_pad)
        self.panel.add_child(gui.Label("bbox pad"))
        self.panel.add_child(self.bbox_pad_edit)

        btn_fit_bbox = gui.Button("Fit BBox To Mask")
        btn_fit_bbox.set_on_clicked(self._on_fit_bbox_to_mask)
        self.panel.add_child(btn_fit_bbox)

        btn_restore_mask = gui.Button("Restore Original Mask")
        btn_restore_mask.set_on_clicked(self._on_restore_original_mask)
        self.panel.add_child(btn_restore_mask)

        btn_clear_mask = gui.Button("Clear Selected Mask")
        btn_clear_mask.set_on_clicked(self._on_clear_selected_mask)
        self.panel.add_child(btn_clear_mask)

        self.panel.add_child(gui.Label("Overall Coordinate Alignment"))

        btn_ax_p = gui.Button("+Align X")
        btn_ax_p.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("x", +1.0, False))
        self.panel.add_child(btn_ax_p)

        btn_ax_n = gui.Button("-Align X")
        btn_ax_n.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("x", -1.0, False))
        self.panel.add_child(btn_ax_n)

        btn_ay_p = gui.Button("+Align Y")
        btn_ay_p.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("y", +1.0, False))
        self.panel.add_child(btn_ay_p)

        btn_ay_n = gui.Button("-Align Y")
        btn_ay_n.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("y", -1.0, False))
        self.panel.add_child(btn_ay_n)

        btn_az_p = gui.Button("+Align Z")
        btn_az_p.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("z", +1.0, False))
        self.panel.add_child(btn_az_p)

        btn_az_n = gui.Button("-Align Z")
        btn_az_n.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("z", -1.0, False))
        self.panel.add_child(btn_az_n)

        self.panel.add_child(gui.Label("Fine Alignment"))

        btn_fax_p = gui.Button("+Fine X")
        btn_fax_p.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("x", +1.0, True))
        self.panel.add_child(btn_fax_p)

        btn_fax_n = gui.Button("-Fine X")
        btn_fax_n.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("x", -1.0, True))
        self.panel.add_child(btn_fax_n)

        btn_fay_p = gui.Button("+Fine Y")
        btn_fay_p.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("y", +1.0, True))
        self.panel.add_child(btn_fay_p)

        btn_fay_n = gui.Button("-Fine Y")
        btn_fay_n.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("y", -1.0, True))
        self.panel.add_child(btn_fay_n)

        btn_faz_p = gui.Button("+Fine Z")
        btn_faz_p.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("z", +1.0, True))
        self.panel.add_child(btn_faz_p)

        btn_faz_n = gui.Button("-Fine Z")
        btn_faz_n.set_on_clicked(lambda: self._on_rotate_scene_alignment_custom("z", -1.0, True))
        self.panel.add_child(btn_faz_n)

        btn_align_reset = gui.Button("Reset Overall Alignment")
        btn_align_reset.set_on_clicked(self._on_reset_alignment)
        self.panel.add_child(btn_align_reset)

        self.panel.add_child(gui.Label("Display"))

        btn_toggle = gui.Button("Toggle Boxes")
        btn_toggle.set_on_clicked(self._on_toggle_boxes)
        self.panel.add_child(btn_toggle)

        btn_reset = gui.Button("Reset Masks/BBoxes")
        btn_reset.set_on_clicked(self._on_reset)
        self.panel.add_child(btn_reset)

        btn_save_align = gui.Button("Save Alignment Only")
        btn_save_align.set_on_clicked(self._on_save_alignment_only)
        self.panel.add_child(btn_save_align)

        btn_save = gui.Button("Save Masks + Alignment")
        btn_save.set_on_clicked(self._on_save_modifier)
        self.panel.add_child(btn_save)

        btn_fit = gui.Button("Fit View")
        btn_fit.set_on_clicked(self._on_fit_view)
        self.panel.add_child(btn_fit)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 330
        self.panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x + panel_width, r.y, r.width - panel_width, r.height)

    # ------------------------------------------------------------------
    # Geometry building
    # ------------------------------------------------------------------

    def _clear_scene(self):
        for name in list(self._added_geometry_names):
            try:
                self.scene_widget.scene.remove_geometry(name)
            except Exception:
                pass
        self._added_geometry_names.clear()

    def _build_current_aligned_xyz(self) -> np.ndarray:
        return self._apply_scene_transform_points(self.xyz_all_orig)

    def _build_point_cloud(self) -> o3d.geometry.PointCloud:
        xyz_all_current = self._build_current_aligned_xyz()

        colors = np.tile(self.base_env_color[None, :], (xyz_all_current.shape[0], 1))
        for i, obj in enumerate(self.objects):
            color = self.object_palette[i % len(self.object_palette)].copy()
            if self.show_only_selected and i != self.selected_idx:
                continue
            if i == self.selected_idx:
                color = np.clip(color * 1.25, 0.0, 1.0)
            colors[obj.indices] = color[None, :]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_all_current)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _build_boxes(self):
        boxes = []
        for i, obj in enumerate(self.objects):
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=obj.bbox_min_world.astype(np.float64),
                max_bound=obj.bbox_max_world.astype(np.float64),
            )
            color = self.object_palette[i % len(self.object_palette)].copy()
            if i == self.selected_idx:
                color = np.clip(color * 1.25, 0.0, 1.0)
            bbox.color = color.tolist()
            boxes.append((f"box_{i}", bbox))
        return boxes

    def _rebuild_scene(self):
        self._clear_scene()

        pcd = self._build_point_cloud()
        self.scene_widget.scene.add_geometry("points", pcd, self.material)
        self._added_geometry_names.add("points")

        if self.show_boxes:
            for name, box in self._build_boxes():
                self.scene_widget.scene.add_geometry(name, box, self.line_material)
                self._added_geometry_names.add(name)

        self.scene_widget.force_redraw()

    # ------------------------------------------------------------------
    # Info / parsing
    # ------------------------------------------------------------------

    def _update_info(self, print_console: bool = False):
        obj = self.objects[self.selected_idx]
        inside_count = int(self._bbox_inside_mask(obj).sum())
        self.info_label.text = (
            f"Selected: {self.selected_idx} | {obj.name}\n"
            f"Objects: {len(self.objects)}\n"
            f"Mask count: {int(obj.indices.sum())}\n"
            f"BBox-inside count: {inside_count}\n"
            f"BBox center: {np.round(obj.center_world, 4)}\n"
            f"BBox size: {np.round(obj.extent_world, 4)}\n"
            f"Mode: {self.bbox_select_mode}"
        )

        if print_console:
            print("=" * 60)
            print(f"Selected object: {self.selected_idx} | {obj.name}")
            print(f"Total objects: {len(self.objects)}")
            print(f"Mask count: {int(obj.indices.sum())}")
            print(f"BBox-inside count: {inside_count}")
            print(f"BBox min: {obj.bbox_min_world}")
            print(f"BBox max: {obj.bbox_max_world}")
            print(f"BBox center: {obj.center_world}")
            print(f"BBox extent: {obj.extent_world}")
            print(f"scene_R:\n{self.scene_R}")
            print(f"scene_t: {self.scene_t}")
            print(f"bbox mode: {self.bbox_select_mode}")

    def _parse_float(self, edit: gui.TextEdit, default: float) -> float:
        try:
            return float(edit.text_value.strip())
        except Exception:
            return float(default)

    # ------------------------------------------------------------------
    # Mask / bbox editing
    # ------------------------------------------------------------------

    def _bbox_inside_mask(self, obj: EditableObject) -> np.ndarray:
        xyz_all_aligned = self._build_current_aligned_xyz()
        inside = np.all(
            (xyz_all_aligned >= obj.bbox_min_world[None, :]) &
            (xyz_all_aligned <= obj.bbox_max_world[None, :]),
            axis=1,
        )
        return inside

    def _write_masks_back_to_modifier(self) -> Dict[str, Any]:
        modifier = copy.deepcopy(self.editing_modifier)
        self._write_alignment_to_modifier_dict(modifier)

        for i, obj in enumerate(self.objects):
            modifier["objects"][i]["affected_gaussian_idx"] = obj.indices.astype(bool)
            modifier["objects"][i]["bbox_editor"] = {
                "bbox_min_world": obj.bbox_min_world.astype(np.float32),
                "bbox_max_world": obj.bbox_max_world.astype(np.float32),
                "mask_count": int(obj.indices.sum()),
            }
        return modifier

    # ------------------------------------------------------------------
    # Scene/global alignment
    # ------------------------------------------------------------------

    def _current_scene_center(self) -> np.ndarray:
        aligned_all = self._apply_scene_transform_points(self.xyz_all_orig)
        return aligned_all.mean(axis=0).astype(np.float32)

    def _apply_global_scene_rotation(self, rot_mat: np.ndarray):
        pivot = self._current_scene_center()

        self.scene_R = rot_mat @ self.scene_R
        self.scene_t = rot_mat @ self.scene_t + pivot - rot_mat @ pivot

        self._recompute_all_bboxes_from_masks()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _set_bbox_mode(self, mode: str):
        self.bbox_select_mode = mode
        self._update_info(print_console=True)

    def _on_translate_bbox(self, delta):
        obj = self.objects[self.selected_idx]
        obj.translate_bbox(delta)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_scale_bbox(self, axis: int, sign: float):
        obj = self.objects[self.selected_idx]
        obj.scale_bbox_axis(axis, sign * self.scale_step)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_apply_bbox_to_mask(self):
        obj = self.objects[self.selected_idx]
        inside = self._bbox_inside_mask(obj)
        before = int(obj.indices.sum())

        if self.bbox_select_mode == "replace":
            obj.indices = inside
        elif self.bbox_select_mode == "add":
            obj.indices = np.logical_or(obj.indices, inside)
        elif self.bbox_select_mode == "remove":
            obj.indices = np.logical_and(obj.indices, np.logical_not(inside))
        else:
            raise ValueError(f"Unknown mode: {self.bbox_select_mode}")

        after = int(obj.indices.sum())
        print(f"Applied bbox to '{obj.name}' with mode={self.bbox_select_mode}: {before} -> {after}")
        self._recompute_all_bboxes_from_masks()
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_fit_bbox_to_mask(self):
        obj = self.objects[self.selected_idx]
        pad = self._parse_float(self.bbox_pad_edit, self.default_bbox_pad)
        self.default_bbox_pad = pad

        xyz_all_aligned = self._build_current_aligned_xyz()
        pts = xyz_all_aligned[obj.indices]
        if pts.shape[0] == 0:
            print(f"Object '{obj.name}' has empty mask, cannot fit bbox.")
            return

        obj.set_bbox_from_points(pts, pad=pad)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_restore_original_mask(self):
        obj = self.objects[self.selected_idx]
        obj.indices = self.original_masks[self.selected_idx].copy()

        xyz_all_aligned = self._build_current_aligned_xyz()
        pts = xyz_all_aligned[obj.indices]
        if pts.shape[0] > 0:
            obj.set_bbox_from_points(pts, pad=0.0)

        print(f"Restored original mask for '{obj.name}'")
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_clear_selected_mask(self):
        obj = self.objects[self.selected_idx]
        obj.indices[:] = False
        print(f"Cleared mask for '{obj.name}'")
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_prev(self):
        self.selected_idx = (self.selected_idx - 1) % len(self.objects)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_next(self):
        self.selected_idx = (self.selected_idx + 1) % len(self.objects)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_toggle_only_selected(self):
        self.show_only_selected = not self.show_only_selected
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_rotate_scene_alignment_custom(self, axis, sign, use_fine: bool):
        angle = sign * (self.fine_rot_deg if use_fine else self.rot_deg)
        rot = R.from_euler(axis, angle, degrees=True).as_matrix().astype(np.float32)
        self._apply_global_scene_rotation(rot)
        print(f"Rotated overall coordinate around {axis.upper()} by {angle} deg")
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_reset_alignment(self):
        self.scene_R = np.eye(3, dtype=np.float32)
        self.scene_t = np.zeros(3, dtype=np.float32)
        self._recompute_all_bboxes_from_masks()

        print("Reset overall alignment to original world coordinate")
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_reset(self):
        self._build_objects_from_current_modifier()
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_save_alignment_only(self):
        self._save_alignment_only_to_file()
        self._update_info(print_console=True)

    def _on_save_modifier(self):
        modifier = self._write_masks_back_to_modifier()
        with open(self.modifier_path, "wb") as f:
            pickle.dump(modifier, f)
        self.editing_modifier = modifier
        print(f"Saved updated masks and alignment back to: {self.modifier_path}")
        self._update_info(print_console=True)

    def _on_fit_view(self):
        pcd = self._build_point_cloud()
        bbox = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

    def _on_close(self):
        self.window = None
        return True


def main():
    parser = argparse.ArgumentParser("Open3D GUI Gaussian bbox/mask editor")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--point_size", type=int, default=3)
    parser.add_argument("--move_step", type=float, default=0.05)
    parser.add_argument("--scale_step", type=float, default=0.05)
    parser.add_argument("--rot_deg", type=float, default=8.0)
    parser.add_argument("--fine_rot_deg", type=float, default=1.0)
    parser.add_argument("--bbox_pad", type=float, default=0.0)
    args = parser.parse_args()

    point_cloud_root = os.path.join(args.model_path, "point_cloud")
    if not os.path.isdir(point_cloud_root):
        raise FileNotFoundError(f"Could not find point_cloud directory: {point_cloud_root}")

    if args.iteration < 0:
        iter_dirs = [d for d in os.listdir(point_cloud_root) if d.startswith("iteration_")]
        if len(iter_dirs) == 0:
            raise FileNotFoundError(f"No iteration_* directories found in {point_cloud_root}")
        iter_ids = sorted(int(d.split("_")[-1]) for d in iter_dirs)
        iteration = iter_ids[-1]
    else:
        iteration = args.iteration

    iter_dir = os.path.join(point_cloud_root, f"iteration_{iteration}")
    ply_path = os.path.join(iter_dir, "point_cloud.ply")
    modifier_path = os.path.join(iter_dir, "editing_modifier.pkl")

    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Missing PLY file: {ply_path}")
    if not os.path.isfile(modifier_path):
        raise FileNotFoundError(f"Missing editing_modifier.pkl: {modifier_path}")

    xyz = load_ply_xyz(ply_path)
    with open(modifier_path, "rb") as f:
        editing_modifier = pickle.load(f)

    gui.Application.instance.initialize()
    InteractiveGaussianGUI(
        xyz_all_orig=xyz,
        editing_modifier=editing_modifier,
        modifier_path=modifier_path,
        point_size=args.point_size,
        move_step=args.move_step,
        scale_step=args.scale_step,
        rot_deg=args.rot_deg,
        fine_rot_deg=args.fine_rot_deg,
        bbox_pad=args.bbox_pad,
    )
    gui.Application.instance.run()


if __name__ == "__main__":
    main()