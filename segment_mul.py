import numpy as np
import torch
import pickle
from scene import Scene
import os
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import skip_feat_decoder
from scipy.spatial.transform import Rotation as R
import featsplat_editor
from einops import einsum
from typing import List, Dict, Any

import open3d as o3d
import time


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


def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [0, 0, 1]
    ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    return Rz, Ry


def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):
    assert not np.all(end == origin)
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=size / 17.5 * scale,
        cone_height=size * 0.2 * scale,
        cylinder_radius=size / 30 * scale,
        cylinder_height=size * (1 - 0.2 * scale)
    )
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return mesh


def densify_surface_points(gaussians, obj_flag: np.ndarray, resolution: int = 256):
    print(bcolors.WARNING + f"Densifying surface Gaussians for object, res={resolution}" + bcolors.ENDC)
    start_cp = time.time()

    binarized_voxel, center_xyz, scale_xyz = gaussians.extract_fields(
        mask=obj_flag,
        resolution=resolution,
        binarize_threshold=0.1
    )
    print("Voxelization time: ", time.time() - start_cp)

    voxel_res = binarized_voxel.shape[0]
    assert voxel_res % 2 == 0

    pts_on_disk_n3 = np.mgrid[0:voxel_res, 0:voxel_res, 0:voxel_res].reshape(3, -1).T
    pts_on_disk_n3 = pts_on_disk_n3[binarized_voxel.flatten() == 1]
    pts_on_disk_n3 = pts_on_disk_n3 / (voxel_res // 2) - 1
    pts_on_disk_n3 = (pts_on_disk_n3 / scale_xyz) + center_xyz

    print("Total infilling points: ", pts_on_disk_n3.shape[0])
    return pts_on_disk_n3


def segment_one_object(
    gaussians,
    clip_segmeter,
    ground_R,
    ground_T,
    fg_obj_list: List[str],
    bg_obj_list: List[str],
    threshold: float,
    object_select_eps: float,
    inward_selection_eps: float,
    final_noise_filtering: bool,
    interactive_viz: bool,
    rigid_object_name: str,
) -> Dict[str, Any]:
    start_cp = time.time()

    fg_obj_bbox = clip_segmeter.fast_compute_rough_bbox(fg_obj_list)

    if interactive_viz:
        print(bcolors.WARNING + f"Check bbox for object: {','.join(fg_obj_list)}" + bcolors.ENDC)
        input("Press enter to continue")
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(gaussians.get_xyz.cpu().numpy())

        center = (fg_obj_bbox[0] + fg_obj_bbox[1]) / 2
        size = fg_obj_bbox[1] - fg_obj_bbox[0]
        bbox = o3d.geometry.OrientedBoundingBox(center=center, R=np.eye(3), extent=size)
        o3d.visualization.draw_geometries([scene_pcd, bbox])

    bounded_xyz = gaussians.get_xyz
    within_bbox = (
        (bounded_xyz[:, 0] > fg_obj_bbox[0, 0]) & (bounded_xyz[:, 0] < fg_obj_bbox[1, 0]) &
        (bounded_xyz[:, 1] > fg_obj_bbox[0, 1]) & (bounded_xyz[:, 1] < fg_obj_bbox[1, 1]) &
        (bounded_xyz[:, 2] > fg_obj_bbox[0, 2]) & (bounded_xyz[:, 2] < fg_obj_bbox[1, 2])
    )

    bounded_xyz = bounded_xyz[within_bbox]
    bounded_xyz_np = bounded_xyz.cpu().numpy()
    current_idx = torch.arange(gaussians.get_xyz.shape[0])[within_bbox.cpu()]
    bounded_features = gaussians.get_distill_features[within_bbox]

    fg_obj_similarity = clip_segmeter.compute_similarity_one(fg_obj_list, feature=bounded_features)
    fg_obj_idx = fg_obj_similarity > threshold

    if interactive_viz:
        print(bcolors.WARNING + f"Check raw selection for object: {','.join(fg_obj_list)}" + bcolors.ENDC)
        input("Press enter to continue")
        selected_pcd = o3d.geometry.PointCloud()
        selected_pcd.points = o3d.utility.Vector3dVector(bounded_xyz_np)
        selected_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * fg_obj_idx[:, None])
        o3d.visualization.draw_geometries([selected_pcd])

    selected_obj_idx = fg_obj_idx

    if interactive_viz:
        print(bcolors.WARNING + f"Check clustered selection for object: {','.join(fg_obj_list)}" + bcolors.ENDC)
        input("Press enter to continue")
        while True:
            selected_obj_idx = clip_segmeter.cluster_instance(
                bounded_xyz_np,
                fg_obj_idx,
                eps=object_select_eps
            )

            clustered_pcd = o3d.geometry.PointCloud()
            clustered_pcd.points = o3d.utility.Vector3dVector(bounded_xyz_np)
            clustered_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * selected_obj_idx[:, None])
            o3d.visualization.draw_geometries([clustered_pcd])

            result = input(
                f"If some particles are missing, increase eps. "
                f"If too many noises, decrease eps. "
                f"Current eps: {object_select_eps:.4f}; New eps: "
            )
            if result == "":
                break
            provided_eps = float(result)
            if np.isclose(provided_eps, object_select_eps):
                break
            object_select_eps = provided_eps
    else:
        selected_obj_idx = clip_segmeter.cluster_instance(
            bounded_xyz_np,
            selected_obj_idx,
            eps=object_select_eps
        )

    selected_obj_idx = clip_segmeter.ground_bbox_filter(
        bounded_xyz_np,
        selected_obj_idx,
        ground_R,
        ground_T,
        boundary=np.array([0, 0, 0])
    )

    bounded_xyz = bounded_xyz[selected_obj_idx]
    bounded_xyz_np = bounded_xyz.cpu().numpy()
    bounded_features = bounded_features[selected_obj_idx]
    current_idx = current_idx[selected_obj_idx]

    word_list = bg_obj_list + fg_obj_list
    text_features_mc = clip_segmeter.get_text_embeddings(word_list)

    chunk_feature_nc = clip_segmeter.decoder_infer(bounded_features, 'object')
    chunk_feature_nc = chunk_feature_nc / (chunk_feature_nc.norm(dim=1, keepdim=True) + 1e-6)
    similarity_nm = einsum(chunk_feature_nc.float(), text_features_mc.float(), 'n c, m c -> n m')

    positive_obj_idx = similarity_nm.argmax(dim=1) >= len(bg_obj_list)
    positive_obj_idx = positive_obj_idx.cpu().numpy()

    if interactive_viz:
        print(bcolors.WARNING + f"Selecting interior for object: {','.join(fg_obj_list)}" + bcolors.ENDC)
        input("Press enter to continue")
        obj_idx = positive_obj_idx
        while True:
            positive_obj_idx = clip_segmeter.ground_bbox_filter(
                bounded_xyz_np,
                obj_idx,
                ground_R,
                ground_T,
                boundary=np.array([inward_selection_eps, inward_selection_eps, inward_selection_eps])
            )
            clustered_pcd = o3d.geometry.PointCloud()
            clustered_pcd.points = o3d.utility.Vector3dVector(bounded_xyz_np)
            clustered_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * positive_obj_idx[:, None])
            o3d.visualization.draw_geometries([clustered_pcd])

            result = input(
                f"Increase inward bbox eps if it contains noises. "
                f"Current eps: {inward_selection_eps:.4f}; New eps: "
            )
            if result == "":
                break
            provided_eps = float(result)
            if np.isclose(provided_eps, inward_selection_eps):
                break
            inward_selection_eps = provided_eps
    else:
        positive_obj_idx = clip_segmeter.ground_bbox_filter(
            bounded_xyz_np,
            positive_obj_idx,
            ground_R,
            ground_T,
            boundary=np.array([inward_selection_eps, inward_selection_eps, inward_selection_eps])
        )

    positive_obj_idx = clip_segmeter.knn_infilling(
        bounded_xyz_np,
        positive_obj_idx,
        dilation_iters=1,
        positive_ratio=0.5,
        k=20
    )

    positive_obj_idx = clip_segmeter.remove_ground(
        bounded_xyz_np,
        positive_obj_idx,
        ground_R,
        ground_T
    )

    non_fg_obj_idx = ~positive_obj_idx
    non_fg_obj_idx = clip_segmeter.knn_infilling(
        bounded_xyz_np,
        non_fg_obj_idx,
        dilation_iters=1,
        positive_ratio=0.5,
        k=20
    )
    positive_obj_idx = ~non_fg_obj_idx

    if final_noise_filtering and bounded_xyz_np.shape[0] > 0:
        guessed_eps = np.mean(bounded_xyz_np.max(axis=0) - bounded_xyz_np.min(axis=0)) / 10
        positive_obj_idx = clip_segmeter.cluster_instance(
            bounded_xyz_np,
            positive_obj_idx,
            eps=guessed_eps
        )

    final_obj_flag = np.zeros(gaussians.get_xyz.shape[0], dtype=bool)
    final_obj_flag[current_idx.cpu().numpy()] = positive_obj_idx

    rigid_obj_similarity = None
    if rigid_object_name:
        rigid_obj_similarity = clip_segmeter.compute_similarity_one(
            rigid_object_name,
            feature=gaussians.get_distill_features[final_obj_flag]
        )
        rigid_obj_similarity = rigid_obj_similarity > 0.8

        non_rigid_obj_similarity = ~rigid_obj_similarity
        non_rigid_obj_idx = clip_segmeter.knn_infilling(
            gaussians.get_xyz.cpu().numpy()[final_obj_flag],
            non_rigid_obj_similarity,
            dilation_iters=1,
            positive_ratio=0.4,
            k=20
        )
        rigid_obj_similarity = ~non_rigid_obj_idx

        rigid_obj_similarity = clip_segmeter.knn_infilling(
            gaussians.get_xyz.cpu().numpy()[final_obj_flag],
            rigid_obj_similarity,
            dilation_iters=3,
            positive_ratio=0.2,
            k=20
        )

    print(f"Segmentation time for {','.join(fg_obj_list)}: ", time.time() - start_cp)

    return {
        "name": ",".join(fg_obj_list),
        "final_obj_flag": final_obj_flag,
        "rigid_obj_similarity": rigid_obj_similarity,
    }


@torch.no_grad()
def select_gs_for_phys_multi(
    dataset: ModelParams,
    iteration: int,
    object_specs: List[Dict[str, Any]],
    ground_plane_name: str,
    threshold: float,
    object_select_eps: float,
    inward_selection_eps: float,
    final_noise_filtering: bool,
    interactive_viz: bool,
    voxel_resolution: int,
):
    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    if interactive_viz:
        print("=====================================")
        print("Interactive visualization enabled")
        print("=====================================")

    ply_editing_dir = os.path.join(
        scene.model_path,
        "point_cloud",
        "iteration_" + str(scene.loaded_iter)
    )

    part_level_flag = (dataset.feature_type == "clip_part")
    my_feat_decoder = skip_feat_decoder(dataset.distill_feature_dim, part_level=part_level_flag).cuda()
    decoder_weight_path = os.path.join(dataset.model_path, "feat_decoder.pth")
    assert os.path.exists(decoder_weight_path)
    decoder_weight_dict = torch.load(decoder_weight_path)
    my_feat_decoder.load_state_dict(decoder_weight_dict, strict=True)
    my_feat_decoder.eval()

    clip_segmeter = featsplat_editor.clip_segmenter(gaussians, my_feat_decoder, clip_device='cuda')
    ground_estimator = featsplat_editor.ground_estimator(rotation_flip=True)

    print(bcolors.OKBLUE + "Estimating ground plane..." + bcolors.ENDC)
    ground_similarity = clip_segmeter.compute_similarity_one(ground_plane_name)
    ground_idx = ground_similarity > threshold
    ground_R, ground_T, ground_inliers = ground_estimator.estimate(
        gaussians.get_xyz.cpu().numpy()[ground_idx]
    )

    segmented_results = []
    for obj_spec in object_specs:
        print(bcolors.OKCYAN + f"\nSegmenting object: {obj_spec['name']}" + bcolors.ENDC)

        result = segment_one_object(
            gaussians=gaussians,
            clip_segmeter=clip_segmeter,
            ground_R=ground_R,
            ground_T=ground_T,
            fg_obj_list=obj_spec["fg_obj_list"],
            bg_obj_list=obj_spec["bg_obj_list"],
            threshold=threshold,
            object_select_eps=obj_spec.get("object_select_eps", object_select_eps),
            inward_selection_eps=obj_spec.get("inward_bbox_offset", inward_selection_eps),
            final_noise_filtering=final_noise_filtering,
            interactive_viz=interactive_viz,
            rigid_object_name=obj_spec.get("rigid_object_name", ""),
        )

        result["obj_spec"] = obj_spec
        segmented_results.append(result)

    # optional overlap resolution: assign each Gaussian to the first matching object
    used_mask = np.zeros(gaussians.get_xyz.shape[0], dtype=bool)
    object_entries = []

    for result in segmented_results:
        obj_flag = result["final_obj_flag"].copy()
        obj_flag = obj_flag & (~used_mask)
        used_mask |= obj_flag

        if obj_flag.sum() == 0:
            print(bcolors.WARNING + f"Object {result['obj_spec']['name']} has empty selection after overlap removal, skipped." + bcolors.ENDC)
            continue

        pts_on_disk_n3 = densify_surface_points(
            gaussians,
            obj_flag=obj_flag,
            resolution=voxel_resolution
        )

        spec = result["obj_spec"]
        action = {
            "action": "physics",
            "particle_type": spec.get("particle_type", "elastic"),
            "infilling_surface_pts": pts_on_disk_n3,
            "static_idx": result["rigid_obj_similarity"] if spec.get("rigid_object_name", "") else None,
            "youngs_modulus_scale": spec.get("youngs_modulus_scale", 2.5),
            "poisson_ratio": spec.get("poisson_ratio", 0.24),
            "gravity_y": spec.get("gravity_y", -4.5),
            "rigid_speed": spec.get("rigid_speed", 0.0),
            "num_frames": spec.get("num_frames", 500),
            "use_rigidity": spec.get("use_rigidity", False),
            "source_prompt_fg": spec["fg_obj_list"],
            "source_prompt_bg": spec["bg_obj_list"],
        }

        object_entries.append({
            "name": spec["name"],
            "affected_gaussian_idx": obj_flag,
            "actions": [action],
        })

    if interactive_viz and len(object_entries) > 0:
        xyz = gaussians.get_xyz.cpu().numpy()
        colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)
        palette = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        for i, obj in enumerate(object_entries):
            colors[obj["affected_gaussian_idx"]] = palette[i % len(palette)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        up_gravity_vec = np.array((0, 1, 0))
        up_gravity_vec = ground_R.T @ up_gravity_vec
        arrow_mesh = get_arrow(up_gravity_vec)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        mesh_frame = mesh_frame.rotate(ground_R.T, center=(0, 0, 0))
        mesh_frame = mesh_frame.translate(-(ground_R.T @ ground_T))

        print(bcolors.WARNING + "Final multi-object check" + bcolors.ENDC)
        input("Press enter to continue")
        o3d.visualization.draw_geometries([pcd, mesh_frame, arrow_mesh])

    editing_modifier_dict = {
        "scene": {
            "ground_R": ground_R,
            "ground_T": ground_T,
        },
        "objects": object_entries
    }

    editing_modifier_save_path = os.path.join(ply_editing_dir, "editing_modifier.pkl")
    with open(editing_modifier_save_path, "wb") as f:
        pickle.dump(editing_modifier_dict, f)

    print(bcolors.OKGREEN + f"Saved: {editing_modifier_save_path}" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"Objects saved: {len(object_entries)}" + bcolors.ENDC)


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object CLIP-based segmentation for physics")
    model = ModelParams(parser, sentinel=True)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ground_plane_name", default="tabletop", type=str)
    parser.add_argument("--threshold", default=0.6, type=float)
    parser.add_argument("--object_select_eps", default=0.1, type=float)
    parser.add_argument("--inward_bbox_offset", default=99, type=float)
    parser.add_argument("--final_noise_filtering", action="store_true")
    parser.add_argument("--interactive_viz", action="store_true")
    parser.add_argument("--voxel_resolution", default=256, type=int)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # define your 3 objects here
    object_specs = [
        {
            "name": "vase_with_flower",
            "fg_obj_list": ["vase", "flower", "flowers", "plant"],
            "bg_obj_list": ["turntable", "tabletop", "wooden table"],
            "rigid_object_name": "",
            "particle_type": "elastic",
            "youngs_modulus_scale": 2.5,
            "poisson_ratio": 0.24,
            "gravity_y": -4.5,
            "rigid_speed": 0.0,
            "num_frames": 500,
            "use_rigidity": False,
        },
        {
            "name": "turntable",
            "fg_obj_list": ["gramophone horn", "brass horn", "speaker horn", "phonograph horn"],
            "bg_obj_list": ["vase", "flower", "flowers", "tabletop", "wooden table"],
            "rigid_object_name": "",
            "particle_type": "elastic",
            "youngs_modulus_scale": 2.5,
            "poisson_ratio": 0.24,
            "gravity_y": -4.5,
            "rigid_speed": 0.0,
            "num_frames": 500,
            "use_rigidity": False,
        },
        {
            "name": "table",
            "fg_obj_list": ["table", "tabletop", "wooden table"],
            "bg_obj_list": ["gramophone horn", "brass horn", "speaker horn", "vase", "flower", "flowers", "plant"],
            "rigid_object_name": "",
            "particle_type": "elastic",
            "youngs_modulus_scale": 2.5,
            "poisson_ratio": 0.24,
            "gravity_y": -4.5,
            "rigid_speed": 0.0,
            "num_frames": 500,
            "use_rigidity": False,
        },
    ]

    select_gs_for_phys_multi(
        dataset=model.extract(args),
        iteration=args.iteration,
        object_specs=object_specs,
        ground_plane_name=args.ground_plane_name,
        threshold=args.threshold,
        object_select_eps=args.object_select_eps,
        inward_selection_eps=args.inward_bbox_offset,
        final_noise_filtering=args.final_noise_filtering,
        interactive_viz=args.interactive_viz,
        voxel_resolution=args.voxel_resolution,
    )