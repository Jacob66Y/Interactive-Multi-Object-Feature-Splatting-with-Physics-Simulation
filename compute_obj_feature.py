import os
from utils.general_utils import pytorch_gc
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
from tqdm import trange
import cv2
from typing import Any, Generator, List, Tuple

import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import maskclip_onnx

# Rate limit workaround
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


def resize_image(img, longest_edge):
    width, height = img.size
    if width > height:
        ratio = longest_edge / width
    else:
        ratio = longest_edge / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return img.resize((new_width, new_height), Image.BILINEAR)


def interpolate_to_patch_size(img_bchw, patch_size):
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W


def is_valid_image(filename):
    ext_test_flag = any(filename.lower().endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
    is_file_flag = os.path.isfile(filename)
    return ext_test_flag and is_file_flag


def show_anns(anns):
    if len(anns) == 0:
        return
    img = np.ones((anns.shape[1], anns.shape[2], 4), dtype=np.float32)
    img[:, :, 3] = 0
    rng = np.random.default_rng(0)  # deterministic visualization colors
    for ann in range(anns.shape[0]):
        m = anns[ann].bool().cpu().numpy()
        color_mask = np.concatenate([rng.random(3), [1.0]]).astype(np.float32)
        img[m] = color_mask
    return img


def build_mask_id_map(
    masks_hw_bool: np.ndarray,
    sort_by_area_desc: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a stable integer label map from boolean masks.

    Returns:
        mask_id_map: (H, W), int32
            0 = background
            1..N = mask index in sorted order
        sorted_indices: (N,), int64
            index mapping from sorted mask order back to original mask order
    """
    num_masks = masks_hw_bool.shape[0]
    H, W = masks_hw_bool.shape[1], masks_hw_bool.shape[2]

    if num_masks == 0:
        return np.zeros((H, W), dtype=np.int32), np.zeros((0,), dtype=np.int64)

    areas = masks_hw_bool.reshape(num_masks, -1).sum(axis=1)
    if sort_by_area_desc:
        sorted_indices = np.argsort(-areas)
    else:
        sorted_indices = np.arange(num_masks, dtype=np.int64)

    mask_id_map = np.zeros((H, W), dtype=np.int32)

    # Larger masks first, smaller masks later overwrite them.
    # This matches the visual compositing behavior and preserves visible finer pieces.
    for sorted_rank, original_idx in enumerate(sorted_indices):
        mask = masks_hw_bool[original_idx]
        mask_id_map[mask] = sorted_rank + 1

    return mask_id_map, sorted_indices


def resize_mask_id_map_nearest(mask_id_map: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    resized = cv2.resize(mask_id_map.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int32)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]


class MaskCLIPFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(
            "ViT-L/14@336px",
            download_root=os.getenv("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)


def autocast_if_cuda(device):
    if device == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def main(args):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    yolo_iou = 0.9
    yolo_conf = 0.4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # For part-level CLIP
    transform = T.Compose([
        T.Resize((args.part_resolution, args.part_resolution)),
        T.ToTensor(),
        norm
    ])

    # For object-level CLIP
    raw_transform = T.Compose([
        T.ToTensor(),
        norm
    ])

    dino_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    clip_model = MaskCLIPFeaturizer().to(device).eval()

    mobilesamv2, ObjAwareModel, predictor = torch.hub.load(
        "RogerQi/MobileSAMV2", args.mobilesamv2_encoder_name
    )
    mobilesamv2.to(device=device)
    mobilesamv2.eval()

    base_dir = args.source_path
    image_dir = os.path.join(base_dir, "images")
    if not os.path.exists(image_dir):
        image_dir = os.path.join(base_dir, "color")
    assert os.path.isdir(image_dir), f"Image directory {image_dir} does not exist."

    obj_clip_feat_dir = os.path.join(base_dir, "sam_clip_features")
    os.makedirs(obj_clip_feat_dir, exist_ok=True)
    part_clip_feat_dir = os.path.join(base_dir, "part_level_features")
    os.makedirs(part_clip_feat_dir, exist_ok=True)
    dinov2_feat_dir = os.path.join(base_dir, "dinov2_vits14")
    os.makedirs(dinov2_feat_dir, exist_ok=True)

    image_paths = [os.path.join(image_dir, fn) for fn in os.listdir(image_dir)]
    image_paths = [fn for fn in image_paths if is_valid_image(fn)]
    image_paths.sort()

    assert len(image_paths) > 0, f"No valid images found in {image_dir}."
    print(f"Found {len(image_paths)} images.")

    obj_feat_path_list = []
    part_feat_path_list = []
    dinov2_feat_path_list = []
    sam_masks_path_list = []
    sam_mask_ids_path_list = []
    sam_mask_ids_final_path_list = []
    sam_boxes_path_list = []
    sam_meta_path_list = []

    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        feat_fn = stem + ".npy"

        obj_feat_path = os.path.join(obj_clip_feat_dir, feat_fn)
        part_feat_path = os.path.join(part_clip_feat_dir, feat_fn)
        dinov2_feat_path = os.path.join(dinov2_feat_dir, feat_fn)

        obj_feat_path_list.append(obj_feat_path)
        part_feat_path_list.append(part_feat_path)
        dinov2_feat_path_list.append(dinov2_feat_path)

        sam_masks_path_list.append(os.path.join(obj_clip_feat_dir, stem + "_masks.npy"))
        sam_mask_ids_path_list.append(os.path.join(obj_clip_feat_dir, stem + "_mask_ids.npy"))
        sam_mask_ids_final_path_list.append(os.path.join(obj_clip_feat_dir, stem + "_mask_ids_final.npy"))
        sam_boxes_path_list.append(os.path.join(obj_clip_feat_dir, stem + "_boxes.npy"))
        sam_meta_path_list.append(os.path.join(obj_clip_feat_dir, stem + "_mask_meta.npz"))

    print("Loading DINOv2 model...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2 = dinov2.to(device).eval()

    # ======================
    # DINOv2 features
    # ======================
    for i in trange(len(image_paths), desc="DINOv2"):
        image = Image.open(image_paths[i]).convert("RGB")
        image = resize_image(image, args.dino_resolution)
        image = dino_transform(image)[:3].unsqueeze(0)
        image, target_H, target_W = interpolate_to_patch_size(image, dinov2.patch_size)
        image = image.to(device)

        with torch.no_grad():
            with autocast_if_cuda(device):
                features = dinov2.forward_features(image)["x_norm_patchtokens"][0]

        features = features.float().cpu().numpy()
        features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
        features_chw = features_hwc.transpose((2, 0, 1))
        np.save(dinov2_feat_path_list[i], features_chw)

        del image, features, features_hwc, features_chw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del dinov2
    pytorch_gc()

    # ======================
    # Object / part features + raw SAM masks
    # ======================
    for i in trange(len(image_paths), desc="SAM/CLIP"):
        image_file_path = image_paths[i]

        image = cv2.imread(image_file_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_file_path}")

        if max(image.shape[:2]) > args.sam_size:
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (int(args.sam_size * image.shape[1] / image.shape[0]), args.sam_size))
            else:
                image = cv2.resize(image, (args.sam_size, int(args.sam_size * image.shape[0] / image.shape[1])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_img_H, raw_img_W = image.shape[:2]

        # part level
        small_W = args.part_feat_res
        small_H = raw_img_H * small_W // raw_img_W

        # obj level
        object_W = args.obj_feat_res
        object_H = raw_img_H * object_W // raw_img_W

        final_W = args.final_feat_res
        final_H = raw_img_H * final_W // raw_img_W

        # ===== Object-aware Model =====
        with torch.no_grad():
            obj_results = ObjAwareModel(
                image, device=device, imgsz=args.sam_size, conf=yolo_conf, iou=yolo_iou, verbose=False
            )

        predictor.set_image(image)
        input_boxes1 = obj_results[0].boxes.xyxy

        if input_boxes1.shape[0] == 0:
            # no bbox found, save zero features
            dummy_obj_feat = np.zeros((768, final_H, final_W), dtype=np.float32)
            dummy_part_feat = np.zeros((768, final_H, final_W), dtype=np.float32)
            np.save(obj_feat_path_list[i], dummy_obj_feat)
            np.save(part_feat_path_list[i], dummy_part_feat)

            # save empty SAM data too
            np.save(sam_masks_path_list[i], np.zeros((0, raw_img_H, raw_img_W), dtype=np.uint8))
            np.save(sam_mask_ids_path_list[i], np.zeros((raw_img_H, raw_img_W), dtype=np.int32))
            np.save(sam_mask_ids_final_path_list[i], np.zeros((final_H, final_W), dtype=np.int32))
            np.save(sam_boxes_path_list[i], np.zeros((0, 4), dtype=np.float32))
            np.savez(
                sam_meta_path_list[i],
                sorted_indices=np.zeros((0,), dtype=np.int64),
                original_box_indices=np.zeros((0,), dtype=np.int64),
                areas=np.zeros((0,), dtype=np.int64),
                image_hw=np.array([raw_img_H, raw_img_W], dtype=np.int32),
                final_hw=np.array([final_H, final_W], dtype=np.int32),
            )
            continue

        # Save original detector boxes in image coordinates
        input_boxes1_np = input_boxes1.detach().cpu().numpy().astype(np.float32)
        np.save(sam_boxes_path_list[i], input_boxes1_np)

        input_boxes = predictor.transform.apply_boxes(input_boxes1_np, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).to(device)

        sam_mask = []

        image_embedding_single = predictor.features
        prompt_embedding_single = mobilesamv2.prompt_encoder.get_dense_pe()

        for (boxes,) in batch_iterator(args.sam_batch_size, input_boxes):
            with torch.no_grad():
                with autocast_if_cuda(device):
                    curr_bs = boxes.shape[0]
                    image_embedding = image_embedding_single.expand(curr_bs, -1, -1, -1).contiguous()
                    prompt_embedding = prompt_embedding_single.expand(curr_bs, -1, -1, -1).contiguous()

                    sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                        points=None,
                        boxes=boxes,
                        masks=None,
                    )

                    low_res_masks, _ = mobilesamv2.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=prompt_embedding,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        simple_type=True,
                    )

                    low_res_masks = predictor.model.postprocess_masks(
                        low_res_masks, predictor.input_size, predictor.original_size
                    )
                    sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold).float()
                    sam_mask.append(sam_mask_pre.squeeze(1).float().cpu())

            del boxes, image_embedding, prompt_embedding, sparse_embeddings, dense_embeddings, low_res_masks, sam_mask_pre
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sam_mask = torch.cat(sam_mask, dim=0).to(device)  # (N, H, W), float on device

        # ------------------------------------------------------------------
        # Save raw SAM masks in machine-readable form
        # ------------------------------------------------------------------
        sam_mask_bool_cpu = sam_mask.bool().cpu().numpy()  # (N, H, W), bool
        sam_mask_u8_cpu = sam_mask_bool_cpu.astype(np.uint8)
        np.save(sam_masks_path_list[i], sam_mask_u8_cpu)

        mask_id_map, sorted_indices = build_mask_id_map(sam_mask_bool_cpu, sort_by_area_desc=True)
        np.save(sam_mask_ids_path_list[i], mask_id_map)

        mask_id_map_final = resize_mask_id_map_nearest(mask_id_map, final_H, final_W)
        np.save(sam_mask_ids_final_path_list[i], mask_id_map_final)

        areas = sam_mask_bool_cpu.reshape(sam_mask_bool_cpu.shape[0], -1).sum(axis=1).astype(np.int64)
        np.savez(
            sam_meta_path_list[i],
            sorted_indices=sorted_indices.astype(np.int64),
            original_box_indices=np.arange(sam_mask_bool_cpu.shape[0], dtype=np.int64),
            areas=areas,
            image_hw=np.array([raw_img_H, raw_img_W], dtype=np.int32),
            final_hw=np.array([final_H, final_W], dtype=np.int32),
        )

        # Visualize SAM mask
        show_img = sam_mask[torch.from_numpy(sorted_indices).to(sam_mask.device)]
        ann_img = show_anns(show_img)
        save_img_path = obj_feat_path_list[i].replace(".npy", "_mask.png")
        Image.fromarray((ann_img * 255).astype(np.uint8)).save(save_img_path)

        # ===== Object-level CLIP feature =====
        raw_input_image = raw_transform(Image.fromarray(image))
        with torch.no_grad():
            with autocast_if_cuda(device):
                whole_image_feature = clip_model(raw_input_image[None].to(device))[0]

        clip_feat_shape = whole_image_feature.shape[0]

        resized_clip_feat_map_bchw = torch.nn.functional.interpolate(
            whole_image_feature.unsqueeze(0).float(),
            size=(object_H, object_W),
            mode="bilinear",
            align_corners=False
        )

        mask_tensor_bchw = sam_mask.unsqueeze(1)

        resized_mask_tensor_bchw = torch.nn.functional.interpolate(
            mask_tensor_bchw.float(),
            size=(object_H, object_W),
            mode="nearest"
        ).bool()

        aggregated_feat_map_obj = torch.zeros(
            (clip_feat_shape, object_H, object_W),
            dtype=torch.float32,
            device=device
        )
        aggregated_feat_cnt_map_obj = torch.zeros(
            (object_H, object_W),
            dtype=torch.int32,
            device=device
        )

        for mask_idx in range(resized_mask_tensor_bchw.shape[0]):
            if resized_mask_tensor_bchw[mask_idx, 0].any():
                aggregated_clip_feat = resized_clip_feat_map_bchw[0, :, resized_mask_tensor_bchw[mask_idx, 0]]
                aggregated_clip_feat = aggregated_clip_feat.mean(dim=1)

                aggregated_feat_map_obj[:, resized_mask_tensor_bchw[mask_idx, 0]] += aggregated_clip_feat[:, None]
                aggregated_feat_cnt_map_obj[resized_mask_tensor_bchw[mask_idx, 0]] += 1

        aggregated_feat_map_obj = aggregated_feat_map_obj / (aggregated_feat_cnt_map_obj[None, :, :].float() + 1e-6)
        aggregated_feat_map_obj = F.interpolate(
            aggregated_feat_map_obj[None], (final_H, final_W), mode="bilinear", align_corners=False
        )[0]

        np.save(obj_feat_path_list[i], aggregated_feat_map_obj.detach().cpu().numpy())

        # visualize bbox
        viz_img = image.copy()
        for bbox_idx in range(input_boxes1.shape[0]):
            bbox = input_boxes1[bbox_idx]
            bbox_xyxy = bbox.cpu().numpy().astype(int)
            cv2.rectangle(viz_img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0), 2)
            cv2.putText(viz_img, f"{bbox_idx}", (bbox_xyxy[0], bbox_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        save_img_path = obj_feat_path_list[i].replace(".npy", "_bbox.png")
        Image.fromarray(viz_img).save(save_img_path)

        # crop images from bbox
        cropped_image_list = []
        bbox_xyxy_list = []
        for bbox_idx in range(input_boxes1.shape[0]):
            bbox = input_boxes1[bbox_idx]
            bbox_xyxy = bbox.cpu().numpy().astype(int)
            bbox_xyxy_list.append(bbox_xyxy)
            crop_img = image[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]
            cropped_image_list.append(crop_img)

        image_tensor_list = []
        for cropped_image in cropped_image_list:
            if cropped_image.size == 0:
                continue
            if not isinstance(cropped_image, Image.Image):
                cropped_image = Image.fromarray(cropped_image)
            image_tensor = transform(cropped_image).unsqueeze(0).to(device)
            image_tensor_list.append(image_tensor)

        # ===== Part-level CLIP feature =====
        aggregated_features = []
        for batch_idx in range(0, len(image_tensor_list), args.part_batch_size):
            with torch.no_grad():
                batch = image_tensor_list[batch_idx:batch_idx + args.part_batch_size]
                batch = torch.cat(batch, dim=0)
                with autocast_if_cuda(device):
                    features = clip_model(batch)
                aggregated_features.append(features.float().cpu())

            del batch, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(aggregated_features) > 0:
            aggregated_features = torch.cat(aggregated_features, dim=0).to(device)
        else:
            aggregated_features = torch.zeros((0, clip_feat_shape, 1, 1), dtype=torch.float32, device=device)

        aggregated_feat_map_part = torch.zeros(
            (clip_feat_shape, small_H, small_W),
            dtype=torch.float32,
            device=device
        )
        aggregated_feat_cnt_map_part = torch.zeros(
            (small_H, small_W),
            dtype=torch.int32,
            device=device
        )

        valid_obj_count = min(len(bbox_xyxy_list), aggregated_features.shape[0])

        for obj_idx in range(valid_obj_count):
            resized_bbox = (bbox_xyxy_list[obj_idx] * (small_W / image.shape[1])).astype(int)
            resized_bbox[0] = np.clip(resized_bbox[0], 0, small_W)
            resized_bbox[2] = np.clip(resized_bbox[2], 0, small_W)
            resized_bbox[1] = np.clip(resized_bbox[1], 0, small_H)
            resized_bbox[3] = np.clip(resized_bbox[3], 0, small_H)

            feat_h = int(resized_bbox[3] - resized_bbox[1])
            feat_w = int(resized_bbox[2] - resized_bbox[0])

            if feat_h <= 0 or feat_w <= 0:
                continue

            resized_feature = F.interpolate(
                aggregated_features[obj_idx].unsqueeze(0),
                (feat_h, feat_w),
                mode="bilinear",
                align_corners=False
            )[0]

            aggregated_feat_map_part[:, resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]] += resized_feature
            aggregated_feat_cnt_map_part[resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]] += 1

        aggregated_feat_map_part = aggregated_feat_map_part / (aggregated_feat_cnt_map_part[None, :, :].float() + 1e-6)
        aggregated_feat_map_part = F.interpolate(
            aggregated_feat_map_part[None], (final_H, final_W), mode="bilinear", align_corners=False
        )[0]
        aggregated_feat_map_part = aggregated_feat_map_part.detach().cpu().numpy()

        np.save(part_feat_path_list[i], aggregated_feat_map_part)

        # cleanup for each image
        del obj_results, input_boxes1, input_boxes1_np, input_boxes
        del image_embedding_single, prompt_embedding_single
        del sam_mask, sam_mask_bool_cpu, sam_mask_u8_cpu, mask_id_map, mask_id_map_final, sorted_indices, areas
        del show_img, ann_img
        del raw_input_image, whole_image_feature, resized_clip_feat_map_bchw
        del mask_tensor_bchw, resized_mask_tensor_bchw
        del aggregated_feat_map_obj, aggregated_feat_cnt_map_obj
        del aggregated_feat_map_part, aggregated_feat_cnt_map_part
        del viz_img, cropped_image_list, bbox_xyxy_list, image_tensor_list, aggregated_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pytorch_gc()


if __name__ == "__main__":
    parser = ArgumentParser("Compute reference features for feature splatting")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--part_batch_size", type=int, default=8, help="Part-level CLIP inference batch size")
    parser.add_argument("--part_resolution", type=int, default=224, help="Part-level CLIP input image resolution")
    parser.add_argument("--sam_size", type=int, default=1024, help="Longest edge for MobileSAMV2 segmentation")
    parser.add_argument("--sam_batch_size", type=int, default=32, help="SAM box batch size to reduce VRAM")
    parser.add_argument("--obj_feat_res", type=int, default=100, help="Intermediate (for MAP) SAM-enhanced Object-level feature resolution")
    parser.add_argument("--part_feat_res", type=int, default=400, help="Intermediate (for MAP) SAM-enhanced Part-level feature resolution")
    parser.add_argument("--final_feat_res", type=int, default=64, help="Final hierarchical CLIP feature resolution")
    parser.add_argument("--dino_resolution", type=int, default=800, help="Longest edge for DINOv2 feature generation")
    parser.add_argument("--mobilesamv2_encoder_name", type=str, default="mobilesamv2_efficientvit_l2", help="MobileSAMV2 encoder name")
    args = parser.parse_args()

    with torch.no_grad():
        main(args)