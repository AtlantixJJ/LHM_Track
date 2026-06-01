# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

import argparse
import json
import os
import traceback

import numpy as np
import torch

from engine.pose_estimation.video2motion import Video2MotionPipeline
from engine.predict_box import init_box_model
from engine.predict_flame import init_gaga_track
from track_image_utils import (
    assert_valid_image_path,
    image_stem,
    prepare_single_image_workspace,
)
from track_video import BaseTracker


class ImageTracker(BaseTracker):
    def _init_models(self, model_path):
        self.sam2seg = init_box_model(model_path)
        self.video2motion = Video2MotionPipeline(
            os.path.join(model_path, "human_model_files"),
            device=self.device,
            kp_mode="sapiens",
            track_mode="samurai",
            is_smooth=False,
            is_smooth_fitting=False,
            pad_ratio=self.opt.pad_ratio,
            min_track_len=1,
            visualize=self.opt.visualize,
        )
        if self.opt.with_flame:
            self.gaga_track = init_gaga_track(
                os.path.join(model_path, "gagatracker"), self.device
            )
        else:
            self.gaga_track = None

    def process_image(self, image_path, output_root):
        assert_valid_image_path(image_path)
        print(f"Processing image: {image_path}")

        work_dir = prepare_single_image_workspace(
            image_path,
            output_root,
            overwrite=self.opt.overwrite,
        )

        self.run_common_stages(
            work_dir,
            output_root,
            fps=1,
            with_flame=self.opt.with_flame,
        )
        print(f"Finish processing image: {image_path}")
        return work_dir


def compare_smplx_json(single_json_path, reference_json_path):
    print(f"\nComparing results:")
    print(f"Single: {single_json_path}")
    print(f"Reference: {reference_json_path}")

    with open(single_json_path, "r") as f:
        single_data = json.load(f)
    with open(reference_json_path, "r") as f:
        ref_data = json.load(f)

    fields = [
        "betas",
        "root_pose",
        "body_pose",
        "jaw_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
        "focal",
        "princpt",
    ]

    report = {}
    for field in fields:
        if field not in single_data or field not in ref_data:
            print(f"Field {field} missing in one of the files.")
            continue

        s_val = np.array(single_data[field])
        r_val = np.array(ref_data[field])

        if s_val.shape != r_val.shape:
            print(f"Shape mismatch for {field}: {s_val.shape} vs {r_val.shape}")
            continue

        diff = np.abs(s_val - r_val)
        report[field] = {
            "max_error": float(np.max(diff)),
            "mean_error": float(np.mean(diff)),
        }
        print(
            f"{field:12}: Max Err = {report[field]['max_error']:.6f}, Mean Err = {report[field]['mean_error']:.6f}"
        )

    return report


def get_parse():
    parser = argparse.ArgumentParser(description="LHM Single-Image Pose Estimation")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to exactly one image file",
    )
    parser.add_argument(
        "--output_path", type=str, default="./train_data/custom_image"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pretrained_models",
        help="Path to pretrained models",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="CUDA device"
    )
    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.2,
        help="Padding ratio for crop",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization"
    )
    parser.add_argument(
        "--with_flame", action="store_true", help="Enable FLAME estimation"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing workspace"
    )
    parser.add_argument(
        "--compare_json",
        type=str,
        help="Path to reference SMPL-X JSON for comparison",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_parse()
    assert torch.cuda.is_available(), "CUDA is not available"

    tracker = ImageTracker(opt.model_path, opt.device, opt)
    try:
        work_dir = tracker.process_image(opt.image_path, opt.output_path)

        if opt.compare_json:
            stem = image_stem(opt.image_path)
            single_json = os.path.join(work_dir, "smplx_params", "00001.json")
            if os.path.exists(single_json):
                compare_smplx_json(single_json, opt.compare_json)
            else:
                print(f"Error: Single-image result not found at {single_json}")

    except Exception:
        traceback.print_exc()
