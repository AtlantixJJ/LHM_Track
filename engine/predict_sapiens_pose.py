import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.append("./")


def consolidate_sapiens_pose(sapiens_dir, delete_json=True, delete_png=True):
    """Merge per-frame sapiens JSON files into sapiens_pose.npy.

    Set delete_json=False to keep JSONs alive for downstream consumers (e.g.
    gaga_track) that read them directly; call again later with delete_json=True
    to clean up.
    """
    json_files = sorted(
        f for f in os.listdir(sapiens_dir) if f.endswith(".json")
    )
    if not json_files:
        return

    frame_ids, keypoints_list, scores_list = [], [], []
    for fname in json_files:
        try:
            frame_idx = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
        with open(os.path.join(sapiens_dir, fname)) as f:
            data = json.load(f)
        info = data["instance_info"]
        if info:
            kpts = np.array(info[0]["keypoints"], dtype=np.float32)    # [133, 2]
            scores = np.array(info[0]["keypoint_scores"], dtype=np.float32)  # [133]
        else:
            kpts = np.zeros((133, 2), dtype=np.float32)
            scores = np.zeros(133, dtype=np.float32)
        frame_ids.append(frame_idx)
        keypoints_list.append(kpts)
        scores_list.append(scores)

    npy_data = {
        "frame_ids": np.array(frame_ids, dtype=np.int32),
        "keypoints": np.stack(keypoints_list, axis=0),      # [N, 133, 2]
        "keypoint_scores": np.stack(scores_list, axis=0),   # [N, 133]
    }
    np.save(os.path.join(sapiens_dir, "sapiens_pose.npy"), npy_data)

    if delete_json:
        for fname in json_files:
            os.remove(os.path.join(sapiens_dir, fname))
    if delete_png:
        for fname in os.listdir(sapiens_dir):
            if fname.endswith(".png"):
                os.remove(os.path.join(sapiens_dir, fname))


def run_sapiens(model_path, output_dir, visualize=False):
    model_path = os.path.join(
        model_path,
        "sapiens/poses/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Sapiens pose model not found: {model_path}\n"
            "Download LHM_track_model.tar and extract into pretrained_models/."
        )

    img_path = os.path.join(output_dir, "imgs_png")
    output_path = os.path.join(output_dir, "sapiens_pose")

    cmd = (
        f"python ./engine/sapiens_api/core/vis_pose.py"
        f" {model_path}"
        f" --num_keypoints 133"
        f" --batch-size 1"
        f" --input {img_path}"
        f" --output-root={output_path}"
        f" --radius 6"
        f" --kpt-thr 0.3"
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"run_sapiens failed with exit code {ret}: {cmd}")

    # Keep JSONs alive: gaga_track reads them directly during flame estimation.
    # Call cleanup_sapiens_json after flame estimation completes.
    consolidate_sapiens_pose(output_path, delete_json=False, delete_png=not visualize)


def cleanup_sapiens_json(output_dir):
    """Delete per-frame sapiens JSON files after all consumers have finished."""
    sapiens_dir = os.path.join(output_dir, "sapiens_pose")
    for fname in os.listdir(sapiens_dir):
        if fname.endswith(".json"):
            os.remove(os.path.join(sapiens_dir, fname))


def _run_sapiens_chunk(model_full, chunk, img_paths, seg_paths, visualize):
    """Run sapiens on one chunk of (global_idx, work_dir) pairs."""
    with tempfile.TemporaryDirectory() as tmp:
        imgs_dir = os.path.join(tmp, "imgs_png")
        segs_dir = os.path.join(tmp, "samurai_seg")
        out_dir = os.path.join(tmp, "sapiens_output")
        os.makedirs(imgs_dir)
        os.makedirs(segs_dir)
        os.makedirs(out_dir)

        valid = []
        for local_idx, (global_idx, work_dir) in enumerate(chunk):
            src_img = (
                img_paths[global_idx] if img_paths is not None
                else os.path.join(work_dir, "imgs_png", "00001.png")
            )
            if not os.path.exists(src_img):
                print(f"[WARN] sapiens_batch: image not found, skipping: {src_img}")
                continue
            name = f"{local_idx:04d}_00001.png"
            os.symlink(os.path.abspath(src_img), os.path.join(imgs_dir, name))

            src_seg = (
                seg_paths[global_idx] if seg_paths is not None
                else os.path.join(work_dir, "samurai_seg", "00001.png")
            )
            if src_seg and os.path.exists(src_seg):
                os.symlink(os.path.abspath(src_seg), os.path.join(segs_dir, name))

            valid.append((local_idx, work_dir))

        if not valid:
            return

        cmd = (
            f"python ./engine/sapiens_api/core/vis_pose.py"
            f" {model_full}"
            f" --num_keypoints 133"
            f" --batch-size 1"
            f" --input {imgs_dir}"
            f" --output-root={out_dir}"
            f" --radius 6"
            f" --kpt-thr 0.3"
        )
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"run_sapiens_batch failed with exit code {ret}: {cmd}")

        for local_idx, work_dir in valid:
            src_json = os.path.join(out_dir, f"{local_idx:04d}_00001.json")
            if not os.path.exists(src_json):
                print(f"[WARN] sapiens_batch: no output json for {work_dir}")
                continue
            sap_dir = os.path.join(work_dir, "sapiens_pose")
            os.makedirs(sap_dir, exist_ok=True)
            shutil.copy2(src_json, os.path.join(sap_dir, "00001.json"))
            consolidate_sapiens_pose(sap_dir, delete_json=False, delete_png=not visualize)


def run_sapiens_batch(model_path, work_dirs, img_paths=None, seg_paths=None,
                      visualize=False, chunk_size=500):
    """Run sapiens pose estimation on all work_dirs, split into chunks to avoid OOM.

    Images and masks are collected into a temporary flat directory per chunk.
    When img_paths / seg_paths are provided they are used directly (no workspace
    copy needed); otherwise falls back to work_dir/imgs_png/ and samurai_seg/.
    Outputs are redistributed to each work_dir's ``sapiens_pose/`` folder.
    """
    model_full = os.path.join(
        model_path,
        "sapiens/poses/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2",
    )
    if not os.path.exists(model_full):
        raise FileNotFoundError(
            f"Sapiens pose model not found: {model_full}\n"
            "Download LHM_track_model.tar and extract into pretrained_models/."
        )

    indexed = list(enumerate(work_dirs))
    chunks = [indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size)]
    for chunk in chunks:
        _run_sapiens_chunk(model_full, chunk, img_paths, seg_paths, visualize)

