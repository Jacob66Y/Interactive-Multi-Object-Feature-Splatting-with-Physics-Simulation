import os
import cv2
import time
import pickle
import numpy as np
import torch
import open3d as o3d

from argparse import ArgumentParser
from typing import Dict, List, Tuple, Any

from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene, skip_feat_decoder
import featsplat_editor
from PIL import Image


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_camera_hw(cam) -> Tuple[int, int]:
    if hasattr(cam, "image_height") and hasattr(cam, "image_width"):
        return int(cam.image_height), int(cam.image_width)
    if hasattr(cam, "height") and hasattr(cam, "width"):
        return int(cam.height), int(cam.width)
    if hasattr(cam, "original_image"):
        h, w = cam.original_image.shape[-2:]
        return int(h), int(w)
    raise AttributeError("Cannot infer camera image size.")


def get_camera_stem(cam) -> str:
    name = str(cam.image_name)
    return os.path.splitext(os.path.basename(name))[0]


def build_point_cloud(points: np.ndarray, colors: np.ndarray = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_distinct_colors(n: int) -> np.ndarray:
    palette = np.array([
        [0.90, 0.10, 0.10],
        [0.10, 0.75, 0.20],
        [0.10, 0.35, 0.90],
        [0.95, 0.75, 0.15],
        [0.80, 0.20, 0.80],
        [0.10, 0.80, 0.80],
        [0.95, 0.45, 0.10],
        [0.55, 0.35, 0.90],
        [0.60, 0.60, 0.60],
        [0.25, 0.80, 0.45],
        [0.75, 0.15, 0.45],
        [0.15, 0.55, 0.55],
    ], dtype=np.float64)
    colors = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        colors[i] = palette[i % len(palette)]
    return colors


def prompt_str(msg: str, default: str = "") -> str:
    raw = input(f"{msg} [{default}]: ").strip()
    return raw if raw != "" else default


def prompt_float(msg: str, default: float) -> float:
    raw = input(f"{msg} [{default}]: ").strip()
    if raw == "":
        return float(default)
    return float(raw)


def prompt_int(msg: str, default: int) -> int:
    raw = input(f"{msg} [{default}]: ").strip()
    if raw == "":
        return int(default)
    return int(raw)


def prompt_yes_no(msg: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{msg} ({suffix}): ").strip().lower()
    if raw == "":
        return default
    return raw in ["y", "yes", "1", "true"]


def parse_id_list(text: str) -> List[int]:
    text = text.strip()
    if text == "":
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def parse_multiline_view_masks(block: str) -> Dict[str, List[int]]:
    view_masks: Dict[str, List[int]] = {}

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line == "":
            continue

        parts = line.split()
        if len(parts) != 2:
            print(f"{bcolors.WARNING}Skip bad line: {raw_line}{bcolors.ENDC}")
            continue

        stem = parts[0]
        ids = parse_id_list(parts[1])
        if len(ids) == 0:
            print(f"{bcolors.WARNING}Skip line with no ids: {raw_line}{bcolors.ENDC}")
            continue

        view_masks[stem] = ids

    return view_masks


def read_view_masks_line_by_line() -> Dict[str, List[int]]:
    print("Enter one or more view selections for this object.")
    print("Format: <frame_stem> <mask_ids_comma_separated>")
    print("Example: frame_00361 2,4,5")
    print("Blank line to finish this object.\n")

    view_masks: Dict[str, List[int]] = {}
    while True:
        line = input("view selection: ").strip()
        if line == "":
            break

        parts = line.split()
        if len(parts) != 2:
            print("Bad format. Use: frame_00361 2,4,5")
            continue

        stem = parts[0]
        ids = parse_id_list(parts[1])
        if len(ids) == 0:
            print("No mask ids given.")
            continue

        view_masks[stem] = ids

    return view_masks


def read_view_masks_from_paste() -> Dict[str, List[int]]:
    print("Paste all view selections for this object at once.")
    print("Format per line: <frame_stem> <mask_ids_comma_separated>")
    print("Example:")
    print("frame_00361 2,4,5")
    print("frame_00362 1,3")
    print("Finish by entering a single line: END\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    block = "\n".join(lines)
    return parse_multiline_view_masks(block)


def read_view_masks_from_file(txt_path: str) -> Dict[str, List[int]]:
    if not os.path.exists(txt_path):
        print(f"{bcolors.FAIL}File not found: {txt_path}{bcolors.ENDC}")
        return {}

    with open(txt_path, "r", encoding="utf-8") as f:
        block = f.read()

    return parse_multiline_view_masks(block)


def ensure_labeled_preview(image_path: str, mask_id_path: str, save_path: str) -> None:
    if os.path.exists(save_path):
        return

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_ids = np.load(mask_id_path)
    mask_h, mask_w = mask_ids.shape[:2]

    if image.shape[0] != mask_h or image.shape[1] != mask_w:
        image = cv2.resize(image, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)

    overlay = image.copy()

    unique_ids = [x for x in np.unique(mask_ids) if x > 0]
    colors = get_distinct_colors(len(unique_ids))
    id_to_color = {mid: (colors[k] * 255).astype(np.uint8) for k, mid in enumerate(unique_ids)}

    alpha = 0.45
    for mid in unique_ids:
        mask = (mask_ids == mid)
        color = id_to_color[mid]
        overlay[mask] = (alpha * color + (1 - alpha) * overlay[mask]).astype(np.uint8)

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())

        cv2.putText(
            overlay,
            str(mid),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            overlay,
            str(mid),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    Image.fromarray(overlay).save(save_path)


def project_points_to_camera(xyz: np.ndarray, cam) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = cam.full_proj_transform.device
    pts = torch.from_numpy(xyz).float().to(device)
    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=device)
    pts_h = torch.cat([pts, ones], dim=1)

    clip = pts_h @ cam.full_proj_transform
    w = clip[:, 3:4]
    ndc = clip[:, :3] / (w + 1e-8)

    H, W = get_camera_hw(cam)
    x = ((ndc[:, 0] + 1.0) * 0.5) * (W - 1)
    y = ((1.0 - ndc[:, 1]) * 0.5) * (H - 1)
    z = ndc[:, 2]

    valid = (
        (w[:, 0] > 0) &
        (x >= 0) & (x <= (W - 1)) &
        (y >= 0) & (y <= (H - 1)) &
        (z >= -1.0) & (z <= 1.0)
    )

    return x.detach().cpu().numpy(), y.detach().cpu().numpy(), z.detach().cpu().numpy(), valid.detach().cpu().numpy()


def compute_visibility_mask(
    px: np.ndarray,
    py: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    H: int,
    W: int,
    depth_tol: float = 1e-2,
) -> np.ndarray:
    vis = np.zeros_like(valid, dtype=bool)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return vis

    xi = np.round(px[valid_idx]).astype(np.int32)
    yi = np.round(py[valid_idx]).astype(np.int32)
    xi = np.clip(xi, 0, W - 1)
    yi = np.clip(yi, 0, H - 1)

    lin = yi * W + xi
    min_depth = np.full(H * W, np.inf, dtype=np.float32)

    d = depth[valid_idx].astype(np.float32)
    for k in range(len(valid_idx)):
        if d[k] < min_depth[lin[k]]:
            min_depth[lin[k]] = d[k]

    vis_valid = np.abs(d - min_depth[lin]) <= depth_tol
    vis[valid_idx] = vis_valid
    return vis


def vote_gaussians_from_manual_masks(
    xyz: np.ndarray,
    cameras_by_stem: Dict[str, Any],
    object_view_masks: Dict[str, List[int]],
    sam_feature_dir: str,
    min_votes: int,
    min_vote_ratio: float,
    depth_tol: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    scores = np.zeros(xyz.shape[0], dtype=np.float32)
    used_views = 0

    for stem, mask_ids in object_view_masks.items():
        if stem not in cameras_by_stem:
            print(f"{bcolors.WARNING}Skip '{stem}': no matching camera found.{bcolors.ENDC}")
            continue

        mask_id_path = os.path.join(sam_feature_dir, stem + "_mask_ids.npy")
        if not os.path.exists(mask_id_path):
            print(f"{bcolors.WARNING}Skip '{stem}': missing {mask_id_path}.{bcolors.ENDC}")
            continue

        mask_map = np.load(mask_id_path)
        H, W = mask_map.shape

        cam = cameras_by_stem[stem]
        cam_H, cam_W = get_camera_hw(cam)

        px, py, depth, valid = project_points_to_camera(xyz, cam)

        if cam_W != W or cam_H != H:
            scale_x = float(W) / float(cam_W)
            scale_y = float(H) / float(cam_H)
            px = px * scale_x
            py = py * scale_y

        visible = compute_visibility_mask(px, py, depth, valid, H, W, depth_tol=depth_tol)

        valid_idx = np.where(valid & visible)[0]
        if len(valid_idx) == 0:
            continue

        xi = np.round(px[valid_idx]).astype(np.int32)
        yi = np.round(py[valid_idx]).astype(np.int32)

        xi = np.clip(xi, 0, W - 1)
        yi = np.clip(yi, 0, H - 1)

        hit_ids = mask_map[yi, xi]
        hit_flag = np.isin(hit_ids, np.array(mask_ids, dtype=np.int32))
        hit_idx = valid_idx[hit_flag]

        scores[hit_idx] += 1.0
        used_views += 1

    if used_views == 0:
        return np.zeros(xyz.shape[0], dtype=bool), scores, 0

    needed_votes = max(min_votes, int(np.ceil(min_vote_ratio * used_views)))

    print(f"used_views = {used_views}")
    print(f"max score = {scores.max()}")
    print(f"needed_votes = {needed_votes}")
    print(f"num gaussians with >=1 vote = {(scores >= 1).sum()}")
    print(f"num gaussians with >=2 votes = {(scores >= 2).sum()}")
    print(f"num gaussians with >=4 votes = {(scores >= 4).sum()}")
    print(f"num gaussians with >={needed_votes} votes = {(scores >= needed_votes).sum()}")

    selected = scores >= needed_votes
    return selected, scores, used_views


def densify_surface_points(gaussians, obj_flag: np.ndarray, voxel_resolution: int = 256) -> np.ndarray:
    binarized_voxel, center_xyz, scale_xyz = gaussians.extract_fields(
        mask=obj_flag,
        resolution=voxel_resolution,
        binarize_threshold=0.1
    )

    voxel_res = binarized_voxel.shape[0]
    pts_on_disk_n3 = np.mgrid[0:voxel_res, 0:voxel_res, 0:voxel_res].reshape(3, -1).T
    pts_on_disk_n3 = pts_on_disk_n3[binarized_voxel.flatten() == 1]
    pts_on_disk_n3 = pts_on_disk_n3 / (voxel_res // 2) - 1
    pts_on_disk_n3 = (pts_on_disk_n3 / scale_xyz) + center_xyz
    return pts_on_disk_n3


def get_manual_object_definition() -> Dict[str, Any]:
    obj_name = input("Object name (blank to finish): ").strip()
    if obj_name == "":
        return {}

    print("\nChoose input mode for this object:")
    print("  1 = line-by-line input")
    print("  2 = paste a whole block")
    print("  3 = load from a text file")
    mode = prompt_str("mode", "1").strip()

    view_masks: Dict[str, List[int]] = {}

    if mode == "1":
        view_masks = read_view_masks_line_by_line()
    elif mode == "2":
        view_masks = read_view_masks_from_paste()
    elif mode == "3":
        txt_path = prompt_str("path to txt file", "")
        view_masks = read_view_masks_from_file(txt_path)
    else:
        print("Unknown mode. Skipping object.")
        return {}

    if len(view_masks) == 0:
        print("No valid views provided for this object. Skipping.")
        return {}

    particle_type = prompt_str("particle_type", "elastic")
    youngs_modulus_scale = prompt_float("youngs_modulus_scale", 2.5)
    poisson_ratio = prompt_float("poisson_ratio", 0.24)
    gravity_y = prompt_float("gravity_y", -4.5)
    rigid_speed = prompt_float("rigid_speed", 0.0)
    num_frames = prompt_int("num_frames", 500)
    use_rigidity = prompt_yes_no("use rigidity for this object", False)

    return {
        "name": obj_name,
        "view_masks": view_masks,
        "particle_type": particle_type,
        "youngs_modulus_scale": youngs_modulus_scale,
        "poisson_ratio": poisson_ratio,
        "gravity_y": gravity_y,
        "rigid_speed": rigid_speed,
        "num_frames": num_frames,
        "use_rigidity": use_rigidity,
    }


@torch.no_grad()
def select_gs_for_phys(
    dataset: ModelParams,
    iteration: int,
    ground_plane_name: str,
    threshold: float,
    interactive_viz: bool,
    voxel_resolution: int,
    min_votes: int,
    min_vote_ratio: float,
    depth_tol: float,
):
    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    ply_editing_dir = os.path.join(
        scene.model_path,
        "point_cloud",
        "iteration_" + str(scene.loaded_iter)
    )

    sam_feature_dir = os.path.join(dataset.source_path, "sam_clip_features")
    image_dir = os.path.join(dataset.source_path, "images")
    if not os.path.exists(image_dir):
        image_dir = os.path.join(dataset.source_path, "color")

    if not os.path.isdir(sam_feature_dir):
        raise RuntimeError(f"Missing {sam_feature_dir}. Run compute_obj_feature.py first.")

    part_level_flag = (dataset.feature_type == "clip_part")
    my_feat_decoder = skip_feat_decoder(dataset.distill_feature_dim, part_level=part_level_flag).cuda()
    decoder_weight_path = os.path.join(dataset.model_path, "feat_decoder.pth")
    if not os.path.exists(decoder_weight_path):
        raise RuntimeError(f"Missing decoder weights: {decoder_weight_path}")
    decoder_weight_dict = torch.load(decoder_weight_path)
    my_feat_decoder.load_state_dict(decoder_weight_dict, strict=True)
    my_feat_decoder.eval()

    clip_segmeter = featsplat_editor.clip_segmenter(gaussians, my_feat_decoder, clip_device="cuda")
    ground_estimator = featsplat_editor.ground_estimator(rotation_flip=True)

    print(f"{bcolors.OKBLUE}Estimating ground plane...{bcolors.ENDC}")
    ground_similarity = clip_segmeter.compute_similarity_one(ground_plane_name)
    ground_idx = ground_similarity > threshold
    ground_R, ground_T, _ = ground_estimator.estimate(gaussians.get_xyz.cpu().numpy()[ground_idx])

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    train_cameras = scene.getTrainCameras()
    cameras_by_stem = {get_camera_stem(cam): cam for cam in train_cameras}

    print(f"{bcolors.OKBLUE}Preparing labeled mask previews...{bcolors.ENDC}")
    available_stems = []
    for stem in sorted(cameras_by_stem.keys()):
        image_path_png = os.path.join(image_dir, stem + ".png")
        image_path_jpg = os.path.join(image_dir, stem + ".jpg")
        image_path_jpeg = os.path.join(image_dir, stem + ".jpeg")
        image_path = None
        for candidate in [image_path_png, image_path_jpg, image_path_jpeg]:
            if os.path.exists(candidate):
                image_path = candidate
                break

        mask_id_path = os.path.join(sam_feature_dir, stem + "_mask_ids.npy")
        save_path = os.path.join(sam_feature_dir, stem + "_mask_labeled.png")

        if image_path is not None and os.path.exists(mask_id_path):
            ensure_labeled_preview(image_path, mask_id_path, save_path)
            available_stems.append(stem)

    print(f"{bcolors.OKGREEN}Labeled previews saved under {sam_feature_dir}{bcolors.ENDC}")
    print("Open the *_mask_labeled.png images and use the printed mask IDs.\n")

    manual_objects = []
    while True:
        obj_def = get_manual_object_definition()
        if len(obj_def) == 0:
            break
        manual_objects.append(obj_def)

    if len(manual_objects) == 0:
        raise RuntimeError("No manual objects defined.")

    print(f"{bcolors.OKBLUE}Voting 3D Gaussian membership from 2D SAM masks...{bcolors.ENDC}")

    object_scores = []
    object_pre_flags = []
    for obj_def in manual_objects:
        pre_flag, score, used_views = vote_gaussians_from_manual_masks(
            xyz=xyz,
            cameras_by_stem=cameras_by_stem,
            object_view_masks=obj_def["view_masks"],
            sam_feature_dir=sam_feature_dir,
            min_votes=min_votes,
            min_vote_ratio=min_vote_ratio,
            depth_tol=depth_tol,
        )
        print(f"Object '{obj_def['name']}': used_views={used_views}, selected_gaussians={int(pre_flag.sum())}")
        object_scores.append(score)
        object_pre_flags.append(pre_flag)

    score_mat = np.stack(object_scores, axis=0)
    pre_flag_mat = np.stack(object_pre_flags, axis=0)

    masked_scores = np.where(pre_flag_mat, score_mat, -1.0)
    best_obj_idx = masked_scores.argmax(axis=0)
    best_scores = masked_scores.max(axis=0)

    final_object_flags = []
    for obj_i in range(len(manual_objects)):
        flag = (best_obj_idx == obj_i) & (best_scores > 0)
        final_object_flags.append(flag)

    if interactive_viz:
        colors = np.tile(np.array([[0.45, 0.45, 0.45]]), (xyz.shape[0], 1))
        obj_colors = get_distinct_colors(len(final_object_flags))
        for i, flag in enumerate(final_object_flags):
            colors[flag] = obj_colors[i]
        pcd = build_point_cloud(xyz, colors)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        mesh_frame = mesh_frame.rotate(ground_R.T, center=(0, 0, 0))
        mesh_frame = mesh_frame.translate(-(ground_R.T @ ground_T))

        print(f"{bcolors.WARNING}Preview final multi-object assignment{bcolors.ENDC}")
        o3d.visualization.draw_geometries([pcd, mesh_frame])

    print(f"{bcolors.OKBLUE}Building editing_modifier.pkl...{bcolors.ENDC}")

    object_entries = []
    for obj_def, obj_flag in zip(manual_objects, final_object_flags):
        if obj_flag.sum() == 0:
            print(f"{bcolors.WARNING}Object '{obj_def['name']}' has no selected Gaussians. Skipping.{bcolors.ENDC}")
            continue

        pts_on_disk_n3 = densify_surface_points(
            gaussians=gaussians,
            obj_flag=obj_flag,
            voxel_resolution=voxel_resolution,
        )

        action = {
            "action": "physics",
            "particle_type": obj_def["particle_type"],
            "infilling_surface_pts": pts_on_disk_n3,
            "static_idx": None,
            "youngs_modulus_scale": obj_def["youngs_modulus_scale"],
            "poisson_ratio": obj_def["poisson_ratio"],
            "gravity_y": obj_def["gravity_y"],
            "rigid_speed": obj_def["rigid_speed"],
            "num_frames": obj_def["num_frames"],
            "use_rigidity": obj_def["use_rigidity"],
            "source_views": obj_def["view_masks"],
        }

        object_entries.append({
            "name": obj_def["name"],
            "affected_gaussian_idx": obj_flag,
            "actions": [action],
        })

    if len(object_entries) == 0:
        raise RuntimeError("No valid objects remained after voting.")

    editing_modifier_dict = {
        "scene": {
            "ground_R": ground_R,
            "ground_T": ground_T,
        },
        "objects": object_entries,
    }

    print(len(editing_modifier_dict["objects"]))
    for i, obj in enumerate(editing_modifier_dict["objects"]):
        print(i, obj["name"], int(np.asarray(obj["affected_gaussian_idx"]).sum()))

    editing_modifier_save_path = os.path.join(ply_editing_dir, "editing_modifier.pkl")
    with open(editing_modifier_save_path, "wb") as f:
        pickle.dump(editing_modifier_dict, f)

    print(f"{bcolors.OKGREEN}Saved: {editing_modifier_save_path}{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Objects saved: {len(object_entries)}{bcolors.ENDC}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Manual mask-driven Gaussian selection for physics")
    model = ModelParams(parser, sentinel=True)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ground_plane_name", default="ground,floor,tabletop", type=str)
    parser.add_argument("--threshold", default=0.6, type=float)
    parser.add_argument("--interactive_viz", action="store_true")

    parser.add_argument("--voxel_resolution", default=256, type=int)
    parser.add_argument("--min_votes", default=1, type=int)
    parser.add_argument("--min_vote_ratio", default=0.4, type=float)
    parser.add_argument("--depth_tol", default=0.01, type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    select_gs_for_phys(
        dataset=model.extract(args),
        iteration=args.iteration,
        ground_plane_name=args.ground_plane_name,
        threshold=args.threshold,
        interactive_viz=args.interactive_viz,
        voxel_resolution=args.voxel_resolution,
        min_votes=args.min_votes,
        min_vote_ratio=args.min_vote_ratio,
        depth_tol=args.depth_tol,
    )