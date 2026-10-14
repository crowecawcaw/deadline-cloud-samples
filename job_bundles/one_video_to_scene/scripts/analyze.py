"""Analyze stage: sample frames, label objects, track their masks, and estimate depth and cameras.

Outputs (analyze/):
  frames/00000.jpg ...  Sampled frames, longest edge FrameSize
  objects.json          [{id, label, deformable, bbox, crop_frame, crop}]
  masks/obj_NN.npz      Bit-packed boolean masks, shape (frames, height, width)
  depth.npz             Metric depth in meters along the camera axis, float16 (frames, height, width)
  cameras.json          Per-frame intrinsics K (frame pixels) and OpenCV camera-to-world matrices, meters
  crops/obj_NN.png      RGBA crop of each object from its least-occluded frame
  labels.jpg            The labeling frame with the detected boxes drawn, for checking at a glance

Models: Qwen3-VL (Apache-2.0) for labels and boxes, SAM 2.1 (Apache-2.0) for mask tracking,
Depth Anything 3 DA3-BASE (Apache-2.0) for depth and cameras, DA3METRIC-LARGE (Apache-2.0) for scale.
"""

import argparse
import gc
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import stages

LABEL_PROMPT = (
    "Locate every distinct physical object in this image that could be picked up or moved, such as "
    "props, tools, food, toys, vehicles, animals, or people. Skip large surfaces and structure like the "
    "floor, walls, tables, ceilings, and sky. {hint}"
    'Report the result in JSON as a list like [{{"bbox_2d": [x1, y1, x2, y2], "label": "short noun", '
    '"deformable": false}}], with one entry per object instance. Set "deformable" to true for objects '
    "that bend or change shape, like cloth, people, and animals."
)


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def extract_frames(video: Path, num_frames: int, frame_size: int, frames_dir: Path):
    """Sample num_frames evenly from the video, resized so the longest edge is frame_size."""
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
             "stream=nb_read_packets,width,height,avg_frame_rate", "-of", "json", str(video)],
            check=True, stdout=subprocess.PIPE, text=True,
        ).stdout
    )["streams"][0]
    total = int(probe["nb_read_packets"])
    num, den = (int(x) for x in probe["avg_frame_rate"].split("/"))
    src_fps = num / den if den else 24.0
    num_frames = min(num_frames, total)
    indices = np.linspace(0, total - 1, num_frames).round().astype(int)
    scale = "scale='if(gte(iw,ih),{0},-2)':'if(gte(iw,ih),-2,{0})'".format(frame_size)
    with tempfile.TemporaryDirectory() as tmp:
        select = "+".join(f"eq(n\\,{i})" for i in indices)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{select}',{scale}", "-fps_mode", "passthrough",
             "-q:v", "2", "-start_number", "0", f"{tmp}/%05d.jpg"],
            check=True,
        )
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(sorted(Path(tmp).glob("*.jpg"))):
            shutil.move(path, frames_dir / f"{i:05d}.jpg")
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    # Playback rate that keeps the sampled frames at the video's real-time speed.
    fps = src_fps * (len(frame_paths) - 1) / max(indices[-1] - indices[0], 1)
    print(f"Sampled {len(frame_paths)} of {total} frames, effective {fps:.2f} fps")
    return frame_paths, fps


def parse_json_list(text: str):
    if "```" in text:
        text = re.sub(r"^```(json)?", "", text.strip()).split("```")[0]
    start = text.find("[")
    text = text[start:] if start >= 0 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Repair output cut off by max_new_tokens: keep the complete entries.
        cut = text.rfind("}")
        return json.loads(text[: cut + 1] + "]") if cut > 0 else []


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / max(area(a) + area(b) - inter, 1e-9)


def label_objects(image_path: Path, model_id: str, hint: str, max_objects: int):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_id)
    prompt = LABEL_PROMPT.format(hint=f"{hint.strip()} " if hint.strip() else "")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print("Labeling model output:\n" + text)
    del model
    free_gpu()

    objects = []
    for item in parse_json_list(text):
        try:
            # Qwen3-VL boxes are relative coordinates in 0..1000.
            x0, y0, x1, y1 = (float(v) for v in item["bbox_2d"])
        except (KeyError, TypeError, ValueError):
            continue
        box = [x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area <= 16 or area > 0.6 * W * H:
            continue
        if any(box_iou(box, o["bbox"]) > 0.7 for o in objects):
            continue
        label = re.sub(r"[^a-z0-9]+", "_", str(item.get("label", "object")).lower()).strip("_") or "object"
        objects.append({"label": label, "bbox": box, "deformable": bool(item.get("deformable", False))})
    # Keep the largest objects when there are more than MaxObjects.
    objects.sort(key=lambda o: -(o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
    objects = objects[:max_objects]
    for i, obj in enumerate(objects, start=1):
        obj["id"] = i
    return objects


def track_masks(frames_dir: Path, objects, num_frames: int, shape):
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
    masks = {o["id"]: np.zeros((num_frames, *shape), bool) for o in objects}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir))
        for obj in objects:
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj["id"], box=np.array(obj["bbox"], np.float32))
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(obj_ids):
                masks[oid][frame_idx] = (mask_logits[i, 0] > 0.0).cpu().numpy()
    del predictor, state
    free_gpu()
    return masks


def estimate_depth_and_cameras(frame_paths, shape):
    """Depth Anything 3: poses and depth from DA3-BASE, rescaled to meters with DA3METRIC-LARGE."""
    from depth_anything_3.api import DepthAnything3

    H, W = shape
    paths = [str(p) for p in frame_paths]
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to("cuda")
    with torch.inference_mode():
        pred = model.inference(paths)
    depth = np.asarray(pred.depth, np.float32)
    conf = np.asarray(pred.conf, np.float32)
    w2c = np.asarray(pred.extrinsics, np.float64)
    if w2c.shape[-2] == 3:
        w2c = np.concatenate([w2c, np.tile([[[0, 0, 0, 1]]], (len(w2c), 1, 1))], axis=1)
    K_proc = np.asarray(pred.intrinsics, np.float64)
    del model, pred
    free_gpu()

    # DA3METRIC-LARGE predicts depth that becomes meters when multiplied by focal / 300.
    metric_model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda")
    sample = np.linspace(0, len(paths) - 1, min(8, len(paths))).round().astype(int)
    with torch.inference_mode():
        metric_pred = metric_model.inference([paths[i] for i in sample])
    metric_raw = np.asarray(metric_pred.depth, np.float32)
    del metric_model, metric_pred
    free_gpu()
    ratios = []
    for j, i in enumerate(sample):
        focal = 0.5 * (K_proc[i][0, 0] + K_proc[i][1, 1])
        metric = metric_raw[j] * focal / 300.0
        if metric.shape != depth[i].shape:
            metric = cv2.resize(metric, depth[i].shape[::-1], interpolation=cv2.INTER_LINEAR)
        valid = (depth[i] > 0) & (metric > 0) & (conf[i] > np.quantile(conf[i], 0.3))
        ratios.append(np.median(metric[valid] / depth[i][valid]))
    scale = float(np.median(ratios))
    print(f"Metric scale factor {scale:.4f} (per-frame {np.round(ratios, 4).tolist()})")

    # Resize depth to the frame resolution and rescale intrinsics to frame pixels.
    h_proc, w_proc = depth.shape[1:]
    sx, sy = W / w_proc, H / h_proc
    depth_full = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in depth]) * scale
    cameras = []
    for i in range(len(paths)):
        K = K_proc[i].copy()
        K[0] *= sx
        K[1] *= sy
        c2w = np.linalg.inv(w2c[i])
        c2w[:3, 3] *= scale
        cameras.append({"K": K.tolist(), "cam_to_world": c2w.tolist()})
    return depth_full, cameras


def best_crop_frame(mask_stack):
    """The frame with the largest mask that doesn't touch the image border."""
    areas = mask_stack.sum((1, 2)).astype(float)
    touches = mask_stack[:, 0, :].any(1) | mask_stack[:, -1, :].any(1) | mask_stack[:, :, 0].any(1) | mask_stack[:, :, -1].any(1)
    score = np.where(touches, areas * 0.3, areas)
    return int(np.argmax(score))


def save_crop(frame_path: Path, mask, path: Path):
    rgb = np.asarray(Image.open(frame_path).convert("RGB"))
    ys, xs = np.nonzero(mask)
    pad = int(0.15 * max(np.ptp(ys), np.ptp(xs))) + 4
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
    alpha = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    rgba = np.dstack([rgb[y0:y1, x0:x1], alpha])
    Image.fromarray(rgba, "RGBA").save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--frame-size", type=int, default=768)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--object-hint", default="")
    args = parser.parse_args()

    out = stages.output_dir("analyze")
    frames_dir = out / "frames"
    print("openjd_status: Sampling frames")
    frame_paths, fps = extract_frames(Path(args.video), args.num_frames, args.frame_size, frames_dir)
    W, H = Image.open(frame_paths[0]).size

    print("openjd_status: Labeling objects")
    objects = label_objects(frame_paths[0], args.vlm_model, args.object_hint, args.max_objects)
    if not objects:
        raise SystemExit("ERROR: The labeling model found no objects. Try a different ObjectHint.")
    preview = Image.open(frame_paths[0]).convert("RGB")
    draw = ImageDraw.Draw(preview)
    for obj in objects:
        draw.rectangle(obj["bbox"], outline=(255, 0, 0), width=2)
        draw.text((obj["bbox"][0] + 3, obj["bbox"][1] + 2), f"{obj['id']} {obj['label']}", fill=(255, 0, 0))
    preview.save(out / "labels.jpg")
    print("Objects:", [(o["id"], o["label"]) for o in objects])

    print("openjd_status: Tracking masks")
    masks = track_masks(frames_dir, objects, len(frame_paths), (H, W))

    print("openjd_status: Estimating depth and cameras")
    depth, cameras = estimate_depth_and_cameras(frame_paths, (H, W))

    (out / "masks").mkdir(exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    for obj in objects:
        m = masks[obj["id"]]
        np.savez_compressed(out / "masks" / f"obj_{obj['id']:02d}.npz", packed=np.packbits(m, axis=-1), width=W)
        if not m.any():
            print(f"WARNING: Object {obj['id']} ({obj['label']}) has an empty mask")
            continue
        obj["crop_frame"] = best_crop_frame(m)
        obj["crop"] = f"crops/obj_{obj['id']:02d}.png"
        save_crop(frame_paths[obj["crop_frame"]], m[obj["crop_frame"]], out / obj["crop"])
        print(f"Object {obj['id']} {obj['label']}: crop frame {obj['crop_frame']}")

    np.savez_compressed(out / "depth.npz", depth=depth.astype(np.float16))
    stages.write_json(out / "cameras.json", {"width": W, "height": H, "fps": fps, "frames": cameras})
    stages.write_json(out / "objects.json", {"objects": objects})
    print(f"openjd_status: Found {len(objects)} objects: " + ", ".join(o["label"] for o in objects))


if __name__ == "__main__":
    main()
