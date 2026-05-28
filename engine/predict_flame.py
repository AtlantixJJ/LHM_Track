# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.
import sys

sys.path.append("./")
sys.path.append("./gaga_track")
import os
import shutil
import traceback
import warnings
from os.path import join

import numpy as np
from gaga_track.track_video import Tracker
from tqdm.std import TqdmExperimentalWarning

warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=True)


def init_gaga_track(model_path, device):
    return Tracker(focal_length=12.0, model_path=model_path, device=device)


def save_flame_to_npy(flame_results, out_path):
    frame_keys = sorted(flame_results.keys(), key=lambda k: int(k.split("_")[-1]))
    n = len(frame_keys)
    frame_ids = np.array([int(k.split("_")[-1]) + 1 for k in frame_keys], dtype=np.int32)
    data = {
        "frame_ids": frame_ids,
        "bbox": np.zeros((n, 4), dtype=np.float32),
        "frame_bbox": np.zeros((n, 4), dtype=np.float32),
        "shapecode": np.zeros((n, 300), dtype=np.float32),
        "expcode": np.zeros((n, 100), dtype=np.float32),
        "posecode": np.zeros((n, 6), dtype=np.float32),
        "neckcode": np.zeros((n, 3), dtype=np.float32),
        "eyecode": np.zeros((n, 6), dtype=np.float32),
        "transform_matrix": np.zeros((n, 12), dtype=np.float32),
    }
    for i, k in enumerate(frame_keys):
        v = flame_results[k]
        data["bbox"][i] = np.asarray(v["bbox"], dtype=np.float32).flatten()[:4]
        data["frame_bbox"][i] = np.asarray(v["frame_bbox"], dtype=np.float32).flatten()[:4]
        data["shapecode"][i] = np.asarray(v["shapecode"], dtype=np.float32).flatten()[:300]
        data["expcode"][i] = np.asarray(v["expcode"], dtype=np.float32).flatten()[:100]
        data["posecode"][i] = np.asarray(v["posecode"], dtype=np.float32).flatten()[:6]
        data["neckcode"][i] = np.asarray(v["neckcode"], dtype=np.float32).flatten()[:3]
        data["eyecode"][i] = np.asarray(v["eyecode"], dtype=np.float32).reshape(-1)[:6]
        data["transform_matrix"][i] = np.asarray(v["transform_matrix"], dtype=np.float32).reshape(-1)[:12]
    np.save(join(out_path, "flame_params.npy"), data)


def estimate_flame(gagatrack, video_dir):
    tmp_path = join(video_dir, "flame_params")
    os.makedirs(tmp_path, exist_ok=True)

    try:
        optim_results = gagatrack.track_video(video_dir, tmp_path)

        if len(optim_results) > 0:
            save_flame_to_npy(optim_results, video_dir)

    except:
        traceback.print_exc()
