import os
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
class RigidBodyState:
    name: str
    indices: np.ndarray
    rest_xyz_local: np.ndarray
    current_xyz_world: np.ndarray
    center_world: np.ndarray
    rotation_matrix: np.ndarray
    translation: np.ndarray

    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    mass: float
    restitution: float
    friction: float
    linear_damping: float
    angular_damping: float
    is_static: bool
    frozen: bool

    half_extents_local: np.ndarray
    rest_center_local: np.ndarray

    def reset(self, initial_center_world: np.ndarray, initial_rotation: np.ndarray) -> None:
        self.center_world = initial_center_world.astype(np.float32).copy()
        self.rotation_matrix = initial_rotation.astype(np.float32).copy()
        self.translation = np.zeros(3, dtype=np.float32)
        self.linear_velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.frozen = False
        self._update_world_points()

    def _update_world_points(self) -> None:
        rotated = (self.rotation_matrix @ self.rest_xyz_local.T).T
        self.current_xyz_world = rotated + self.center_world[None, :]

    def apply_translation(self, delta: np.ndarray) -> None:
        delta = delta.astype(np.float32)
        self.center_world += delta
        self.translation += delta
        self._update_world_points()

    def apply_rotation_about_center(self, rot_mat: np.ndarray) -> None:
        rot_mat = rot_mat.astype(np.float32)
        self.rotation_matrix = rot_mat @ self.rotation_matrix
        self._update_world_points()

    def integrate(self, dt: float, gravity: np.ndarray) -> None:
        if self.is_static or self.frozen:
            return

        self.linear_velocity += gravity.astype(np.float32) * dt
        self.linear_velocity *= max(0.0, 1.0 - self.linear_damping * dt)
        self.angular_velocity *= max(0.0, 1.0 - self.angular_damping * dt)

        self.center_world += self.linear_velocity * dt

        omega_norm = np.linalg.norm(self.angular_velocity)
        if omega_norm > 1e-8:
            axis = self.angular_velocity / omega_norm
            angle = omega_norm * dt
            rot_step = R.from_rotvec(axis * angle).as_matrix().astype(np.float32)
            self.rotation_matrix = rot_step @ self.rotation_matrix

        self._update_world_points()

    def get_bottom_z(self) -> float:
        return float(self.current_xyz_world[:, 2].min())

    def get_aabb_world(self):
        mn = self.current_xyz_world.min(axis=0)
        mx = self.current_xyz_world.max(axis=0)
        return mn, mx

    def to_serializable_dict(self) -> dict:
        return {
            "name": self.name,
            "center_world": self.center_world.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation": self.translation.tolist(),
            "linear_velocity": self.linear_velocity.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "mass": float(self.mass),
            "restitution": float(self.restitution),
            "friction": float(self.friction),
            "is_static": bool(self.is_static),
            "frozen": bool(self.frozen),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_xyz_world": self.current_xyz_world.copy(),
            "center_world": self.center_world.copy(),
            "rotation_matrix": self.rotation_matrix.copy(),
            "translation": self.translation.copy(),
            "linear_velocity": self.linear_velocity.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            "frozen": self.frozen,
        }

    def restore_snapshot(self, snap: Dict[str, Any]) -> None:
        self.current_xyz_world = snap["current_xyz_world"].copy()
        self.center_world = snap["center_world"].copy()
        self.rotation_matrix = snap["rotation_matrix"].copy()
        self.translation = snap["translation"].copy()
        self.linear_velocity = snap["linear_velocity"].copy()
        self.angular_velocity = snap["angular_velocity"].copy()
        self.frozen = bool(snap["frozen"])


class SimplePhysicsWorld:
    """
    Physics runs in the SAME aligned coordinate shown in the viewer.
    Z axis is the gravity axis.
    Ground plane is z = ground_z.
    """

    def __init__(self, bodies: List[RigidBodyState], ground_z: float = 0.0):
        self.bodies = bodies
        self.ground_z = float(ground_z)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

    def step(self, dt: float) -> None:
        for body in self.bodies:
            body.integrate(dt, self.gravity)
            self._resolve_ground_collision(body)

        self._resolve_pairwise_aabb_collisions()

    def _resolve_ground_collision(self, body: RigidBodyState) -> None:
        if body.is_static or body.frozen:
            return

        bottom_z = body.get_bottom_z()
        if bottom_z < self.ground_z:
            penetration = self.ground_z - bottom_z
            body.center_world[2] += penetration
            body._update_world_points()

            if body.linear_velocity[2] < 0.0:
                body.linear_velocity[2] = -body.restitution * body.linear_velocity[2]

            body.linear_velocity[0] *= max(0.0, 1.0 - body.friction)
            body.linear_velocity[1] *= max(0.0, 1.0 - body.friction)

            if abs(body.linear_velocity[2]) < 0.05:
                body.linear_velocity[2] = 0.0

    def _resolve_pairwise_aabb_collisions(self) -> None:
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a = self.bodies[i]
                b = self.bodies[j]

                a_locked = a.is_static or a.frozen
                b_locked = b.is_static or b.frozen

                if a_locked and b_locked:
                    continue

                a_min, a_max = a.get_aabb_world()
                b_min, b_max = b.get_aabb_world()

                overlap = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
                if np.any(overlap <= 0):
                    continue

                axis = int(np.argmin(overlap))
                penetration = float(overlap[axis])
                delta = np.zeros(3, dtype=np.float32)

                if a.center_world[axis] < b.center_world[axis]:
                    delta[axis] = -penetration
                else:
                    delta[axis] = penetration

                if not a_locked and not b_locked:
                    a.center_world += 0.5 * delta
                    b.center_world -= 0.5 * delta
                elif not a_locked:
                    a.center_world += delta
                elif not b_locked:
                    b.center_world -= delta

                a._update_world_points()
                b._update_world_points()

                if axis == 2:
                    if not a_locked and a.linear_velocity[2] > 0:
                        a.linear_velocity[2] *= -a.restitution
                    if not b_locked and b.linear_velocity[2] < 0:
                        b.linear_velocity[2] *= -b.restitution
                else:
                    if not a_locked:
                        a.linear_velocity[axis] *= -0.3
                    if not b_locked:
                        b.linear_velocity[axis] *= -0.3


class InteractiveGaussianGUI:
    ALIGNMENT_KEY = "_editor_scene_alignment"

    def __init__(
        self,
        xyz_all_orig: np.ndarray,
        editing_modifier: dict,
        modifier_path: str,
        save_path: str,
        point_size: int = 3,
        move_step: float = 0.05,
        rot_deg: float = 8.0,
        fine_rot_deg: float = 1.0,
        sim_dt: float = 1.0 / 60.0,
    ):
        self.xyz_all_orig = xyz_all_orig.copy()
        self.editing_modifier = editing_modifier
        self.modifier_path = modifier_path
        self.save_path = save_path
        self.point_size = point_size
        self.move_step = move_step
        self.rot_deg = rot_deg
        self.fine_rot_deg = fine_rot_deg
        self.sim_dt = sim_dt

        # aligned_xyz = scene_R @ orig_xyz + scene_t
        self.scene_R = np.eye(3, dtype=np.float32)
        self.scene_t = np.zeros(3, dtype=np.float32)
        self._load_alignment_from_modifier()

        self.base_env_color = np.array([0.22, 0.22, 0.22], dtype=np.float32)
        self.object_palette = [
            np.array([1.0, 0.2, 0.2], dtype=np.float32),
            np.array([0.2, 1.0, 0.2], dtype=np.float32),
            np.array([0.2, 0.45, 1.0], dtype=np.float32),
            np.array([1.0, 0.8, 0.2], dtype=np.float32),
            np.array([0.9, 0.2, 0.9], dtype=np.float32),
            np.array([0.2, 0.9, 0.9], dtype=np.float32),
        ]

        self.objects: List[RigidBodyState] = []
        self.selected_idx = 0
        self.show_boxes = True
        self.sim_running = False

        self.initial_centers = []
        self.initial_rotations = []

        self._build_objects_from_original()
        self.physics = SimplePhysicsWorld(self.objects, ground_z=self._estimate_ground_z())

        self.window = gui.Application.instance.create_window(
            "Interactive Gaussian Physics Viewer", 1720, 1060
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

        self.frame_material = rendering.MaterialRecord()
        self.frame_material.shader = "defaultUnlit"

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

        self._tick()

    # ------------------------------------------------------------------
    # Alignment persistence
    # ------------------------------------------------------------------

    def _load_alignment_from_modifier(self):
        align = self.editing_modifier.get(self.ALIGNMENT_KEY, None)
        if align is None:
            print("No saved alignment found. Using identity.")
            return

        try:
            scene_R = np.asarray(align["scene_R"], dtype=np.float32)
            scene_t = np.asarray(align["scene_t"], dtype=np.float32)

            if scene_R.shape != (3, 3):
                raise ValueError(f"scene_R must be shape (3,3), got {scene_R.shape}")
            if scene_t.shape != (3,):
                raise ValueError(f"scene_t must be shape (3,), got {scene_t.shape}")

            # Optional sanity check for rotation-like matrix
            det = float(np.linalg.det(scene_R))
            if not np.isfinite(det) or abs(det) < 1e-8:
                raise ValueError("scene_R is singular or invalid")

            self.scene_R = scene_R.copy()
            self.scene_t = scene_t.copy()

            print("Loaded saved alignment:")
            print("scene_R:\n", self.scene_R)
            print("scene_t:", self.scene_t)
        except Exception as e:
            print(f"Failed to load saved alignment. Using identity. Error: {e}")
            self.scene_R = np.eye(3, dtype=np.float32)
            self.scene_t = np.zeros(3, dtype=np.float32)

    def _write_alignment_to_modifier_dict(self, modifier: dict):
        modifier[self.ALIGNMENT_KEY] = {
            "scene_R": self.scene_R.astype(np.float32),
            "scene_t": self.scene_t.astype(np.float32),
        }

    def _save_alignment_only_to_file(self):
        with open(self.modifier_path, "rb") as f:
            modifier = pickle.load(f)

        self._write_alignment_to_modifier_dict(modifier)

        with open(self.modifier_path, "wb") as f:
            pickle.dump(modifier, f)

        self.editing_modifier = modifier
        print(f"Saved alignment ONLY to: {self.modifier_path}")
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

    def _clear_objects(self):
        self.objects.clear()
        self.initial_centers.clear()
        self.initial_rotations.clear()

    def _build_objects_from_original(self) -> None:
        self._clear_objects()

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
            if obj_xyz_orig.shape[0] == 0:
                continue

            obj_xyz_aligned = self._apply_scene_transform_points(obj_xyz_orig)
            center_world = obj_xyz_aligned.mean(axis=0).astype(np.float32)
            rest_xyz_local = obj_xyz_aligned - center_world[None, :]
            half_extents = 0.5 * (obj_xyz_aligned.max(axis=0) - obj_xyz_aligned.min(axis=0))
            half_extents = np.maximum(half_extents.astype(np.float32), 1e-4)

            body = RigidBodyState(
                name=obj.get("name", f"object_{i}"),
                indices=mask,
                rest_xyz_local=rest_xyz_local.astype(np.float32),
                current_xyz_world=obj_xyz_aligned.astype(np.float32).copy(),
                center_world=center_world.copy(),
                rotation_matrix=np.eye(3, dtype=np.float32),
                translation=np.zeros(3, dtype=np.float32),
                linear_velocity=np.zeros(3, dtype=np.float32),
                angular_velocity=np.zeros(3, dtype=np.float32),
                mass=1.0,
                restitution=0.15,
                friction=0.08,
                linear_damping=0.15,
                angular_damping=0.2,
                is_static=False,
                frozen=False,
                half_extents_local=half_extents,
                rest_center_local=np.zeros(3, dtype=np.float32),
            )

            self.objects.append(body)
            self.initial_centers.append(center_world.copy())
            self.initial_rotations.append(np.eye(3, dtype=np.float32))

        if len(self.objects) == 0:
            raise ValueError("All object masks were empty.")

    def _estimate_ground_z(self) -> float:
        aligned_all = self._apply_scene_transform_points(self.xyz_all_orig)
        return float(np.percentile(aligned_all[:, 2], 2.0))

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

        btn_freeze = gui.Button("Freeze / Unfreeze Selected")
        btn_freeze.set_on_clicked(self._on_toggle_freeze)
        self.panel.add_child(btn_freeze)

        self.panel.add_child(gui.Label("Translate Object"))

        btn_tx_p = gui.Button("+X")
        btn_tx_p.set_on_clicked(lambda: self._on_translate_button(np.array([ self.move_step, 0.0, 0.0], dtype=np.float32)))
        self.panel.add_child(btn_tx_p)

        btn_tx_n = gui.Button("-X")
        btn_tx_n.set_on_clicked(lambda: self._on_translate_button(np.array([-self.move_step, 0.0, 0.0], dtype=np.float32)))
        self.panel.add_child(btn_tx_n)

        btn_ty_p = gui.Button("+Y")
        btn_ty_p.set_on_clicked(lambda: self._on_translate_button(np.array([0.0,  self.move_step, 0.0], dtype=np.float32)))
        self.panel.add_child(btn_ty_p)

        btn_ty_n = gui.Button("-Y")
        btn_ty_n.set_on_clicked(lambda: self._on_translate_button(np.array([0.0, -self.move_step, 0.0], dtype=np.float32)))
        self.panel.add_child(btn_ty_n)

        btn_tz_p = gui.Button("+Z")
        btn_tz_p.set_on_clicked(lambda: self._on_translate_button(np.array([0.0, 0.0,  self.move_step], dtype=np.float32)))
        self.panel.add_child(btn_tz_p)

        btn_tz_n = gui.Button("-Z")
        btn_tz_n.set_on_clicked(lambda: self._on_translate_button(np.array([0.0, 0.0, -self.move_step], dtype=np.float32)))
        self.panel.add_child(btn_tz_n)

        self.panel.add_child(gui.Label("Rotate Object Geometry"))

        btn_rx_p = gui.Button("+Rot X")
        btn_rx_p.set_on_clicked(lambda: self._on_rotate_button("x", +1.0))
        self.panel.add_child(btn_rx_p)

        btn_rx_n = gui.Button("-Rot X")
        btn_rx_n.set_on_clicked(lambda: self._on_rotate_button("x", -1.0))
        self.panel.add_child(btn_rx_n)

        btn_ry_p = gui.Button("+Rot Y")
        btn_ry_p.set_on_clicked(lambda: self._on_rotate_button("y", +1.0))
        self.panel.add_child(btn_ry_p)

        btn_ry_n = gui.Button("-Rot Y")
        btn_ry_n.set_on_clicked(lambda: self._on_rotate_button("y", -1.0))
        self.panel.add_child(btn_ry_n)

        btn_rz_p = gui.Button("+Rot Z")
        btn_rz_p.set_on_clicked(lambda: self._on_rotate_button("z", +1.0))
        self.panel.add_child(btn_rz_p)

        btn_rz_n = gui.Button("-Rot Z")
        btn_rz_n.set_on_clicked(lambda: self._on_rotate_button("z", -1.0))
        self.panel.add_child(btn_rz_n)

        self.panel.add_child(gui.Label("Overall Coordinate / Gravity Alignment"))
        self.panel.add_child(gui.Label("Rotates scene + env + objects + physics"))

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

        btn_save_align = gui.Button("Save Alignment Only")
        btn_save_align.set_on_clicked(self._on_save_alignment_only)
        self.panel.add_child(btn_save_align)

        self.panel.add_child(gui.Label("Initial Horizontal Velocity"))
        self.panel.add_child(gui.Label("(horizontal = X/Y, vertical = Z)"))

        self.vx_edit = gui.TextEdit()
        self.vx_edit.text_value = "0.0"
        self.panel.add_child(gui.Label("vx"))
        self.panel.add_child(self.vx_edit)

        self.vy_edit = gui.TextEdit()
        self.vy_edit.text_value = "0.0"
        self.panel.add_child(gui.Label("vy"))
        self.panel.add_child(self.vy_edit)

        btn_apply_vel = gui.Button("Apply Vx/Vy to Selected")
        btn_apply_vel.set_on_clicked(self._on_apply_velocity)
        self.panel.add_child(btn_apply_vel)

        btn_zero_vel = gui.Button("Zero Selected Velocity")
        btn_zero_vel.set_on_clicked(self._on_zero_velocity)
        self.panel.add_child(btn_zero_vel)

        self.panel.add_child(gui.Label("Recording"))

        self.record_frames_edit = gui.TextEdit()
        self.record_frames_edit.text_value = "500"
        self.panel.add_child(gui.Label("record frames"))
        self.panel.add_child(self.record_frames_edit)

        self.record_substeps_edit = gui.TextEdit()
        self.record_substeps_edit.text_value = "4"
        self.panel.add_child(gui.Label("record substeps"))
        self.panel.add_child(self.record_substeps_edit)

        btn_record = gui.Button("Record Trajectory")
        btn_record.set_on_clicked(self._on_record_trajectory)
        self.panel.add_child(btn_record)

        self.panel.add_child(gui.Label("Physics"))

        btn_play = gui.Button("Play / Pause")
        btn_play.set_on_clicked(self._on_toggle_sim)
        self.panel.add_child(btn_play)

        btn_step = gui.Button("Step Once")
        btn_step.set_on_clicked(self._on_step_once)
        self.panel.add_child(btn_step)

        self.panel.add_child(gui.Label("Display"))

        btn_toggle = gui.Button("Toggle Boxes")
        btn_toggle.set_on_clicked(self._on_toggle_boxes)
        self.panel.add_child(btn_toggle)

        btn_reset = gui.Button("Reset")
        btn_reset.set_on_clicked(self._on_reset)
        self.panel.add_child(btn_reset)

        btn_save = gui.Button("Save State")
        btn_save.set_on_clicked(self._on_save)
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

    def _sync_global_xyz_from_objects(self):
        self.xyz_all_current = self._apply_scene_transform_points(self.xyz_all_orig)
        for obj in self.objects:
            self.xyz_all_current[obj.indices] = obj.current_xyz_world

    def _build_point_cloud(self) -> o3d.geometry.PointCloud:
        self._sync_global_xyz_from_objects()

        colors = np.tile(self.base_env_color[None, :], (self.xyz_all_current.shape[0], 1))
        for i, obj in enumerate(self.objects):
            color = self.object_palette[i % len(self.object_palette)].copy()
            if i == self.selected_idx:
                color = np.clip(color * 1.25, 0.0, 1.0)
            if obj.frozen:
                color = np.clip(
                    0.5 * color + 0.5 * np.array([0.8, 0.8, 0.8], dtype=np.float32),
                    0.0,
                    1.0,
                )
            colors[obj.indices] = color[None, :]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz_all_current)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _build_boxes(self):
        boxes = []
        for i, obj in enumerate(self.objects):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj.current_xyz_world)
            bbox = pcd.get_axis_aligned_bounding_box()
            color = self.object_palette[i % len(self.object_palette)].copy()
            if obj.frozen:
                color = np.clip(
                    0.5 * color + 0.5 * np.array([0.8, 0.8, 0.8], dtype=np.float32),
                    0.0,
                    1.0,
                )
            bbox.color = color.tolist()
            boxes.append((f"box_{i}", bbox))
        return boxes

    def _build_ground_plane_mesh(self):
        size = 4.0
        z = self.physics.ground_z
        verts = np.array([
            [-size, -size, z],
            [ size, -size, z],
            [ size,  size, z],
            [-size,  size, z],
        ], dtype=np.float64)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.35, 0.35, 0.35])
        return mesh

    def _rebuild_scene(self):
        self._clear_scene()

        pcd = self._build_point_cloud()
        self.scene_widget.scene.add_geometry("points", pcd, self.material)
        self._added_geometry_names.add("points")

        ground_mesh = self._build_ground_plane_mesh()
        self.scene_widget.scene.add_geometry("ground_plane", ground_mesh, self.frame_material)
        self._added_geometry_names.add("ground_plane")

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
        self.info_label.text = (
            f"Selected: {self.selected_idx} | {obj.name}\n"
            f"Objects: {len(self.objects)}\n"
            f"Center: {np.round(obj.center_world, 4)}\n"
            f"Vel: {np.round(obj.linear_velocity, 4)}\n"
            f"Frozen: {obj.frozen}\n"
            f"Sim: {'RUNNING' if self.sim_running else 'PAUSED'}\n"
            f"Ground Z: {self.physics.ground_z:.4f}\n"
            f"Gravity: [0, 0, {self.physics.gravity[2]:.2f}]"
        )

        if print_console:
            print("=" * 60)
            print(f"Selected object: {self.selected_idx} | {obj.name}")
            print(f"Total objects: {len(self.objects)}")
            print(f"Points: {obj.current_xyz_world.shape[0]}")
            print(f"Center: {obj.center_world}")
            print(f"Translation: {obj.translation}")
            print(f"Velocity: {obj.linear_velocity}")
            print(f"Frozen: {obj.frozen}")
            print(f"Rotation:\n{obj.rotation_matrix}")
            print(f"scene_R:\n{self.scene_R}")
            print(f"scene_t: {self.scene_t}")
            print(f"ground_z: {self.physics.ground_z}")

    def _parse_float(self, edit: gui.TextEdit, default: float) -> float:
        try:
            return float(edit.text_value.strip())
        except Exception:
            return float(default)

    def _parse_int(self, edit: gui.TextEdit, default: int) -> int:
        try:
            return int(edit.text_value.strip())
        except Exception:
            return int(default)

    # ------------------------------------------------------------------
    # Editing modifier IO
    # ------------------------------------------------------------------

    def _find_or_create_physics_action(self, obj_entry: Dict[str, Any]) -> Dict[str, Any]:
        actions = obj_entry.setdefault("actions", [])
        for act in actions:
            if act.get("action", "") == "physics":
                return act

        new_action = {
            "action": "physics",
            "particle_type": "rigid",
        }
        actions.append(new_action)
        return new_action

    def _snapshot_all_objects(self) -> List[Dict[str, Any]]:
        return [obj.snapshot() for obj in self.objects]

    def _restore_all_objects(self, snapshots: List[Dict[str, Any]]) -> None:
        for obj, snap in zip(self.objects, snapshots):
            obj.restore_snapshot(snap)

    def _write_recorded_trajectories_to_modifier(self, trajectories_aligned: List[np.ndarray], num_frames: int) -> None:
        with open(self.modifier_path, "rb") as f:
            modifier = pickle.load(f)

        self._write_alignment_to_modifier_dict(modifier)

        for i, traj_aligned in enumerate(trajectories_aligned):
            if i >= len(modifier["objects"]):
                continue

            T, N, _ = traj_aligned.shape
            traj_orig = self._invert_scene_transform_points(
                traj_aligned.reshape(-1, 3)
            ).reshape(T, N, 3).astype(np.float32)

            obj_entry = modifier["objects"][i]
            action = self._find_or_create_physics_action(obj_entry)

            action["particles_trajectory_tn3"] = traj_orig
            action["used_num_frames"] = int(num_frames)
            action["used_gravity_axis"] = "z"
            action["used_gravity_value"] = float(self.physics.gravity[2])
            action["used_initial_velocity_aligned"] = self.objects[i].linear_velocity.astype(np.float32).copy()
            action["used_scene_R"] = self.scene_R.astype(np.float32).copy()
            action["used_scene_t"] = self.scene_t.astype(np.float32).copy()

        with open(self.modifier_path, "wb") as f:
            pickle.dump(modifier, f)

        self.editing_modifier = modifier
        print(f"Updated physics trajectories in: {self.modifier_path}")

    def _record_current_trajectories(self, num_frames: int, substeps: int) -> None:
        print(f"Recording trajectories for {len(self.objects)} object(s), frames={num_frames}, substeps={substeps}")

        snapshots = self._snapshot_all_objects()
        scene_R_before = self.scene_R.copy()
        scene_t_before = self.scene_t.copy()
        ground_z_before = self.physics.ground_z
        sim_running_before = self.sim_running
        self.sim_running = False

        trajectories = [[] for _ in self.objects]

        substeps = max(1, int(substeps))
        dt_sub = self.sim_dt / substeps

        for _ in range(num_frames):
            for _sub in range(substeps):
                self.physics.step(dt_sub)
                for i, obj in enumerate(self.objects):
                    trajectories[i].append(obj.current_xyz_world.copy())

        traj_arrays = [np.stack(t, axis=0).astype(np.float32) for t in trajectories]
        self._write_recorded_trajectories_to_modifier(traj_arrays, traj_arrays[0].shape[0])

        self._restore_all_objects(snapshots)
        self.scene_R = scene_R_before
        self.scene_t = scene_t_before
        self.physics.ground_z = ground_z_before
        self.sim_running = sim_running_before
        self._rebuild_scene()
        self._update_info(print_console=True)

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

        for obj in self.objects:
            obj.center_world = rot_mat @ (obj.center_world - pivot) + pivot
            obj.rotation_matrix = rot_mat @ obj.rotation_matrix
            obj.linear_velocity = rot_mat @ obj.linear_velocity
            obj.angular_velocity = rot_mat @ obj.angular_velocity
            obj._update_world_points()

        self.physics.ground_z = self._estimate_ground_z()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_translate_button(self, delta):
        obj = self.objects[self.selected_idx]
        obj.apply_translation(delta)
        obj.linear_velocity[:] = 0.0
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_rotate_button(self, axis, sign):
        angle = sign * self.rot_deg
        rot = R.from_euler(axis, angle, degrees=True).as_matrix().astype(np.float32)
        obj = self.objects[self.selected_idx]
        obj.apply_rotation_about_center(rot)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_toggle_freeze(self):
        obj = self.objects[self.selected_idx]
        obj.frozen = not obj.frozen
        if obj.frozen:
            obj.linear_velocity[:] = 0.0
            obj.angular_velocity[:] = 0.0
        print(f"{'Froze' if obj.frozen else 'Unfroze'} object '{obj.name}'")
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

        self._build_objects_from_original()
        self.physics = SimplePhysicsWorld(self.objects, ground_z=self._estimate_ground_z())
        self.sim_running = False

        print("Reset overall alignment to original world coordinate")
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_save_alignment_only(self):
        self._save_alignment_only_to_file()
        self._update_info(print_console=True)

    def _on_apply_velocity(self):
        vx = self._parse_float(self.vx_edit, 0.0)
        vy = self._parse_float(self.vy_edit, 0.0)
        obj = self.objects[self.selected_idx]
        if obj.frozen:
            print(f"Object '{obj.name}' is frozen. Unfreeze it first to apply velocity.")
            return
        obj.linear_velocity[0] = vx
        obj.linear_velocity[1] = vy
        print(f"Applied horizontal velocity to '{obj.name}': vx={vx}, vy={vy}")
        self._update_info(print_console=True)

    def _on_zero_velocity(self):
        obj = self.objects[self.selected_idx]
        obj.linear_velocity[:] = 0.0
        obj.angular_velocity[:] = 0.0
        print(f"Zeroed velocity for '{obj.name}'")
        self._update_info(print_console=True)

    def _on_record_trajectory(self):
        num_frames = self._parse_int(self.record_frames_edit, 500)
        substeps = self._parse_int(self.record_substeps_edit, 4)

        if num_frames <= 0:
            print("record frames must be > 0")
            return
        if substeps <= 0:
            print("record substeps must be > 0")
            return

        self._record_current_trajectories(num_frames, substeps)

    def _on_prev(self):
        self.selected_idx = (self.selected_idx - 1) % len(self.objects)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_next(self):
        self.selected_idx = (self.selected_idx + 1) % len(self.objects)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_toggle_sim(self):
        self.sim_running = not self.sim_running
        self._update_info(print_console=True)

    def _on_step_once(self):
        self.physics.step(self.sim_dt)
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_reset(self):
        self.sim_running = False
        self._build_objects_from_original()
        self.physics = SimplePhysicsWorld(self.objects, ground_z=self._estimate_ground_z())
        self._rebuild_scene()
        self._update_info(print_console=True)

    def _on_save(self):
        payload = {
            "scene": {
                "scene_R": self.scene_R.tolist(),
                "scene_t": self.scene_t.tolist(),
                "ground_z_runtime": float(self.physics.ground_z),
                "gravity_axis": "z",
                "gravity_value": float(self.physics.gravity[2]),
            },
            "objects": [obj.to_serializable_dict() for obj in self.objects],
        }
        with open(self.save_path, "wb") as f:
            pickle.dump(payload, f)

        with open(self.modifier_path, "rb") as f:
            modifier = pickle.load(f)

        self._write_alignment_to_modifier_dict(modifier)

        with open(self.modifier_path, "wb") as f:
            pickle.dump(modifier, f)

        self.editing_modifier = modifier
        print(f"Saved state to {self.save_path}")
        print(f"Saved alignment to {self.modifier_path}")

    def _on_fit_view(self):
        pcd = self._build_point_cloud()
        bbox = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _tick(self):
        if self.window is None:
            return

        if self.sim_running:
            self.physics.step(self.sim_dt)
            self._rebuild_scene()
            self._update_info(print_console=False)

        gui.Application.instance.post_to_main_thread(self.window, self._tick)

    def _on_close(self):
        self.window = None
        return True


def main():
    parser = argparse.ArgumentParser("Open3D GUI Gaussian physics viewer")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="interactive_object_states.pkl")
    parser.add_argument("--point_size", type=int, default=3)
    parser.add_argument("--move_step", type=float, default=0.05)
    parser.add_argument("--rot_deg", type=float, default=8.0)
    parser.add_argument("--fine_rot_deg", type=float, default=1.0)
    parser.add_argument("--sim_dt", type=float, default=1.0 / 60.0)
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
        save_path=args.save_path,
        point_size=args.point_size,
        move_step=args.move_step,
        rot_deg=args.rot_deg,
        fine_rot_deg=args.fine_rot_deg,
        sim_dt=args.sim_dt,
    )
    gui.Application.instance.run()


if __name__ == "__main__":
    main()