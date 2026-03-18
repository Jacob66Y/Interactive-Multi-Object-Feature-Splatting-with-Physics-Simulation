import os
import pickle
import numpy as np
import taichi as ti

from tqdm import trange
from argparse import ArgumentParser

from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration
from submodules.mpm_engine.mpm_solver import MPMSolver

def resolve_material_type(material_value):
    """
    Accept either:
      - MPMSolver material constants
      - strings like 'elastic', 'snow', 'sand', 'water'
    and return the solver constant.
    """
    if isinstance(material_value, str):
        material_key = material_value.lower().strip()
        if material_key == "elastic":
            return MPMSolver.material_elastic
        elif material_key == "snow":
            return MPMSolver.material_snow
        elif material_key == "sand":
            return MPMSolver.material_sand
        elif material_key == "water":
            return MPMSolver.material_water
        else:
            raise ValueError(f"Invalid material type string: {material_value}")

    # already an enum/int constant
    return material_value


def infill_particles(
    infilling_method,
    infilling_voxel_res,
    support_per_particles,
    real_gaussian_particle,
    rigid_idx,
    surface_particles,
    particles,
):
    voxel_occupancy_arr = np.zeros(
        (infilling_voxel_res, infilling_voxel_res, infilling_voxel_res), dtype=np.uint8
    )
    rigid_voxel = np.zeros(
        (infilling_voxel_res, infilling_voxel_res, infilling_voxel_res), dtype=bool
    )

    for particles_idx in range(real_gaussian_particle.shape[0]):
        start_pos = particles[:real_gaussian_particle.shape[0]].mean(axis=0)
        end_pos = particles[particles_idx]
        for support_idx in range(support_per_particles):
            pos = (
                start_pos * (support_per_particles - support_idx) + end_pos * support_idx
            ) / support_per_particles
            voxel_pos = (pos * infilling_voxel_res).astype(int)
            voxel_pos = np.clip(voxel_pos, 0, infilling_voxel_res - 1)
            voxel_occupancy_arr[voxel_pos[0], voxel_pos[1], voxel_pos[2]] = 1
            rigid_voxel[voxel_pos[0], voxel_pos[1], voxel_pos[2]] = (
                rigid_idx[particles_idx] or rigid_voxel[voxel_pos[0], voxel_pos[1], voxel_pos[2]]
            )

    if infilling_method == "simple_interpolation":
        infilled_particles = np.mgrid[
            0:infilling_voxel_res, 0:infilling_voxel_res, 0:infilling_voxel_res
        ].reshape(3, -1).T
        valid = voxel_occupancy_arr.flatten() == 1
        infilled_particles = infilled_particles[valid] / infilling_voxel_res
        infilled_rigid_flag = rigid_voxel.flatten()[valid]
    elif infilling_method == "ray_testing":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError(f"Invalid infilling method: {infilling_method}")

    particles = np.concatenate([particles, infilled_particles], axis=0)

    rigid_flag = rigid_idx.astype(np.int32)
    rigid_flag = np.concatenate(
        [
            rigid_flag,
            np.zeros(surface_particles.shape[0], dtype=np.int32),
            np.array(infilled_rigid_flag, dtype=np.int32),
        ],
        axis=0,
    )
    return particles, rigid_flag


def get_action_value(action_dict, key, fallback):
    return action_dict[key] if key in action_dict else fallback


def sim_mpm_physics(
    obj_dict,
    action_dict,
    xyz,
    sim_res,
    ground_R,
    ground_T,
    infilling_method,
    default_material_type,
    default_rigid_speed,
    default_use_rigidity,
    default_youngs_modulus_scale,
    default_poisson_ratio,
    default_num_frames,
    default_gravity_y,
    show_gui,
):
    infilling_voxel_res = 128
    support_per_particles = 20
    max_surface_particles = int(get_action_value(action_dict, "max_surface_particles", 10000))
    ground_y = float(get_action_value(action_dict, "ground_y", 0.05))

    material_type = resolve_material_type(
    get_action_value(action_dict, "particle_type", default_material_type))
    '''
    youngs_modulus_scale = float(
        get_action_value(action_dict, "youngs_modulus_scale", default_youngs_modulus_scale)
    )
    poisson_ratio = float(get_action_value(action_dict, "poisson_ratio", default_poisson_ratio))
    gravity_y = float(get_action_value(action_dict, "gravity_y", default_gravity_y))
    rigid_speed = float(get_action_value(action_dict, "rigid_speed", default_rigid_speed))
    use_rigidity = bool(get_action_value(action_dict, "use_rigidity", default_use_rigidity))
    num_frames = int(get_action_value(action_dict, "num_frames", default_num_frames))
    '''

    youngs_modulus_scale = float(default_youngs_modulus_scale)
    poisson_ratio = float(default_poisson_ratio)
    gravity_y = float(default_gravity_y)
    rigid_speed = float(default_rigid_speed)
    use_rigidity = bool(default_use_rigidity)
    num_frames = int(default_num_frames)

    real_gaussian_particle = xyz[obj_dict["affected_gaussian_idx"]]
    real_gaussian_particle_size = real_gaussian_particle.shape[0]

    rigid_idx = np.zeros(real_gaussian_particle_size, dtype=bool)
    if use_rigidity and action_dict.get("static_idx", None) is not None:
        rigid_idx = action_dict["static_idx"]

    surface_particles = action_dict["infilling_surface_pts"]
    if surface_particles.shape[0] > max_surface_particles:
        surface_particles = np.random.permutation(surface_particles)[:max_surface_particles]

    all_particles = np.concatenate([real_gaussian_particle, surface_particles], axis=0)

    particles = all_particles @ ground_R.T
    particles += ground_T

    particle_max = particles.max(axis=0)
    particle_min = particles.min(axis=0)
    particle_min[1] = min(particle_min[1], ground_y)

    longest_side = max(particle_max - particle_min)
    if longest_side <= 1e-8:
        raise RuntimeError(f"Object '{obj_dict['name']}' has degenerate bounding box.")

    particles[:, 0] /= longest_side
    particles[:, 1] /= longest_side
    particles[:, 2] /= longest_side

    shift_constant = np.array([
        -particles[:, 0].mean() + 0.5,
        -particles[:, 1].min(),
        -particles[:, 2].mean() + 0.5,
    ])

    particles += shift_constant

    particles, rigid_flag = infill_particles(
        infilling_method,
        infilling_voxel_res,
        support_per_particles,
        real_gaussian_particle,
        rigid_idx,
        surface_particles,
        particles,
    )

    particles = np.asarray(particles, dtype=np.float32)
    particles = np.ascontiguousarray(particles)

    rigid_flag = np.asarray(rigid_flag, dtype=np.int32).reshape(-1)
    rigid_flag = np.ascontiguousarray(rigid_flag)

    assert particles.ndim == 2 and particles.shape[1] == 3, f"Bad particles shape: {particles.shape}"
    assert rigid_flag.ndim == 1, f"Bad rigid_flag shape: {rigid_flag.shape}"
    assert rigid_flag.shape[0] == particles.shape[0], (
        f"rigid_flag length {rigid_flag.shape[0]} != particles {particles.shape[0]}"
    )
    if not np.isfinite(particles).all():
        raise RuntimeError("Particles contain NaN or Inf")

    eps = 2.0 / sim_res
    particles = np.clip(particles, eps, 1.0 - eps)

    print(f"[DEBUG] object: {obj_dict['name']}")
    print(f"[DEBUG] real_gaussian_particle_size: {real_gaussian_particle_size}")
    print(f"[DEBUG] surface_particles shape: {surface_particles.shape}")
    print(f"[DEBUG] particles after infill: {particles.shape}")
    print(f"[DEBUG] rigid_flag shape: {rigid_flag.shape}")
    print(f"[DEBUG] particles min: {particles.min(axis=0)}")
    print(f"[DEBUG] particles max: {particles.max(axis=0)}")

    gui = None
    if show_gui:
        gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

    mpm = MPMSolver(
        res=(sim_res, sim_res, sim_res),
        size=1,
        max_num_particles=2 ** 21,
        E_scale=youngs_modulus_scale,
        poisson_ratio=poisson_ratio,
        unbounded=True,
    )

    mpm.add_particles(
        particles=particles,
        material=material_type,
        color=0xFFFF00,
        motion_override_flag_arr=rigid_flag,
    )

    mpm.add_surface_collider(
        point=(0.0, ground_y, 0.0),
        normal=(0, 1, 0),
        surface=mpm.surface_sticky,
    )

    mpm.set_gravity((0, gravity_y, 0))

    particles_trajectory = []

    for frame in trange(num_frames, desc=f"Sim {obj_dict['name']}"):
        particles_info = mpm.particle_info()
        real_gaussian_pos = particles_info["position"][:real_gaussian_particle_size]
        particles_trajectory.append(real_gaussian_pos.copy())

        if gui is not None:
            np_x = particles_info["position"][:real_gaussian_particle_size]
            screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2 ** 0.5) - 0.2
            screen_y = np_x[:, 1]
            screen_pos = np.stack([screen_x, screen_y], axis=-1)
            gui.circles(screen_pos, radius=2, color=particles_info["color"][:real_gaussian_particle_size])
            gui.show()

        if frame < 100:
            override_velocity = [0, 0, 0]
        else:
            cycle_idx = frame // 30
            if cycle_idx % 2 == 0:
                override_velocity = [rigid_speed, 0, 0]
            else:
                override_velocity = [-rigid_speed, 0, 0]

        mpm.step(4e-3, override_velocity=override_velocity)

    particles_trajectory_tn3 = np.stack(particles_trajectory)

    particles_trajectory_tn3 -= shift_constant
    particles_trajectory_tn3 *= longest_side

    particles_trajectory_tn3 = particles_trajectory_tn3 - ground_T
    particles_trajectory_tn3 = particles_trajectory_tn3 @ ground_R

    assert particles_trajectory_tn3.shape[1] == real_gaussian_particle_size

    action_dict["particles_trajectory_tn3"] = particles_trajectory_tn3
    action_dict["used_material_type"] = material_type
    action_dict["used_youngs_modulus_scale"] = youngs_modulus_scale
    action_dict["used_poisson_ratio"] = poisson_ratio
    action_dict["used_gravity_y"] = gravity_y
    action_dict["used_rigid_speed"] = rigid_speed
    action_dict["used_num_frames"] = num_frames

    return action_dict


def main(
    dataset,
    sim_res,
    infilling_method,
    material_type,
    rigid_speed,
    use_rigidity,
    youngs_modulus_scale,
    poisson_ratio,
    num_frames,
    gravity_y,
    show_gui,
    device_memory_GB,
):
    load_iters = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))

    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    gaussians.load_ply(
        os.path.join(
            dataset.model_path,
            "point_cloud",
            "iteration_" + str(load_iters),
            "point_cloud.ply",
        )
    )

    modifier_dict_pt = os.path.join(
        dataset.model_path,
        "point_cloud",
        "iteration_" + str(load_iters),
        "editing_modifier.pkl",
    )

    with open(modifier_dict_pt, "rb") as f:
        editing_modifier_dict = pickle.load(f)

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    ground_R = editing_modifier_dict["scene"]["ground_R"]
    ground_T = editing_modifier_dict["scene"]["ground_T"]

    ti.init(arch=ti.cuda, device_memory_GB=device_memory_GB)
    #ti.init(arch=ti.cpu)

    for obj_idx in range(len(editing_modifier_dict["objects"])):
        obj_dict = editing_modifier_dict["objects"][obj_idx]
        for action_idx in range(len(obj_dict["actions"])):
            action_dict = obj_dict["actions"][action_idx]
            if action_dict["action"] != "physics":
                continue

            editing_modifier_dict["objects"][obj_idx]["actions"][action_idx] = sim_mpm_physics(
                obj_dict=obj_dict,
                action_dict=action_dict,
                xyz=xyz,
                sim_res=sim_res,
                ground_R=ground_R,
                ground_T=ground_T,
                infilling_method=infilling_method,
                default_material_type=material_type,
                default_rigid_speed=rigid_speed,
                default_use_rigidity=use_rigidity,
                default_youngs_modulus_scale=youngs_modulus_scale,
                default_poisson_ratio=poisson_ratio,
                default_num_frames=num_frames,
                default_gravity_y=gravity_y,
                show_gui=show_gui,
            )

    with open(modifier_dict_pt, "wb") as f:
        pickle.dump(editing_modifier_dict, f)

    print(f"Updated physics trajectories in: {modifier_dict_pt}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run MPM physics for selected objects")
    model = ModelParams(parser, sentinel=True)

    parser.add_argument("--sim_res", type=int, default=64)
    parser.add_argument("--infilling_method", type=str, default="simple_interpolation")
    parser.add_argument("--material_type", type=str, default="elastic")
    parser.add_argument("--rigid_speed", type=float, default=0.0)
    parser.add_argument("--use_rigidity", action="store_true")
    parser.add_argument("--youngs_modulus_scale", type=float, default=2.5)
    parser.add_argument("--poisson_ratio", type=float, default=0.24)
    parser.add_argument("--num_frames", type=int, default=500)
    parser.add_argument("--gravity_y", type=float, default=-4.5)
    parser.add_argument("--show_gui", action="store_true")
    parser.add_argument("--device_memory_GB", type=float, default=6.0)

    args = get_combined_args(parser)

    main(
        dataset=model.extract(args),
        sim_res=args.sim_res,
        infilling_method=args.infilling_method,
        material_type=args.material_type,
        rigid_speed=args.rigid_speed,
        use_rigidity=args.use_rigidity,
        youngs_modulus_scale=args.youngs_modulus_scale,
        poisson_ratio=args.poisson_ratio,
        num_frames=args.num_frames,
        gravity_y=args.gravity_y,
        show_gui=args.show_gui,
        device_memory_GB=args.device_memory_GB,
    )