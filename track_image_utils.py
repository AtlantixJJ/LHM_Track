# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

import json
import os
import shutil

import cv2

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def assert_valid_image_path(image_path: str) -> None:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.lower().endswith(IMAGE_SUFFIXES):
        raise ValueError(
            f"Unsupported image format: {image_path}. Supported: {IMAGE_SUFFIXES}"
        )


def image_stem(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def prepare_single_image_workspace(
    image_path: str,
    output_root: str,
    overwrite: bool = False,
) -> str:
    stem = image_stem(image_path)
    work_dir = os.path.join(output_root, stem)

    if os.path.exists(work_dir):
        if not overwrite:
            print(f"Workspace already exists: {work_dir}. Skipping preparation.")
            return work_dir
        shutil.rmtree(work_dir)

    os.makedirs(work_dir, exist_ok=True)

    imgs_png_dir = os.path.join(work_dir, "imgs_png")
    os.makedirs(imgs_png_dir, exist_ok=True)

    # LHM expects 00001.png
    target_path = os.path.join(imgs_png_dir, "00001.png")

    # Read and write to normalize format and ensure BGR (OpenCV default)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    cv2.imwrite(target_path, img)

    write_image_manifest(work_dir, image_path)

    return work_dir


def prepare_image_sequence_workspace(
    image_paths: list[str],
    output_root: str,
    sequence_name: str = "_subject_visualization",
    overwrite: bool = False,
) -> str:
    if not image_paths:
        raise ValueError("No images provided for sequence workspace")

    work_dir = os.path.join(output_root, sequence_name)

    if os.path.exists(work_dir):
        if not overwrite:
            print(f"Workspace already exists: {work_dir}. Skipping preparation.")
            return work_dir
        shutil.rmtree(work_dir)

    os.makedirs(work_dir, exist_ok=True)
    imgs_png_dir = os.path.join(work_dir, "imgs_png")
    os.makedirs(imgs_png_dir, exist_ok=True)

    frames = []
    for idx, image_path in enumerate(image_paths, start=1):
        assert_valid_image_path(image_path)
        generated_name = f"{idx:05d}.png"
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        cv2.imwrite(os.path.join(imgs_png_dir, generated_name), img)
        frames.append(
            {
                "frame_id": idx - 1,
                "generated_name": generated_name,
                "source_path": os.path.abspath(image_path),
            }
        )

    write_image_manifest(work_dir, None, frames=frames, mode="image_sequence")
    return work_dir


def write_image_manifest(
    work_dir: str,
    image_path: str | None,
    frames: list[dict] | None = None,
    mode: str = "single_image",
) -> None:
    manifest_path = os.path.join(work_dir, "image_manifest.json")
    if frames is None:
        frames = [
            {
                "frame_id": 0,
                "generated_name": "00001.png",
                "source_path": os.path.abspath(image_path),
            }
        ]
    manifest = {
        "mode": mode,
        "frames": frames,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
