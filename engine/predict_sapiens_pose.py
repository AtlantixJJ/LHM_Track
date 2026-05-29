import json
import os
import sys

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
