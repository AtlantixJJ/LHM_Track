# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

import os
import shutil

import cv2
import numpy as np

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
    skip_imgs_png: bool = False,
) -> str:
    stem = image_stem(image_path)
    work_dir = os.path.join(output_root, stem)

    if os.path.exists(work_dir):
        if not overwrite:
            print(f"Workspace already exists: {work_dir}. Skipping preparation.")
            return work_dir
        shutil.rmtree(work_dir)

    if not skip_imgs_png:
        imgs_png_dir = os.path.join(work_dir, "imgs_png")
        os.makedirs(imgs_png_dir, exist_ok=True)

        # LHM expects 00001.png
        target_path = os.path.join(imgs_png_dir, "00001.png")

        # Read and write to normalize format and ensure BGR (OpenCV default)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        cv2.imwrite(target_path, img)

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

    imgs_png_dir = os.path.join(work_dir, "imgs_png")
    os.makedirs(imgs_png_dir, exist_ok=True)

    target_h = None
    target_w = None
    for idx, image_path in enumerate(image_paths, start=1):
        assert_valid_image_path(image_path)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        orig_h, orig_w = img.shape[:2]
        if target_h is None or target_w is None:
            target_h, target_w = orig_h, orig_w

        normalized_img, _ = letterbox_image(img, target_h, target_w)
        if not cv2.imwrite(os.path.join(imgs_png_dir, f"{idx:05d}.png"), normalized_img):
            raise ValueError(f"Failed to write normalized sequence image: {idx:05d}.png")

    return work_dir


def letterbox_image(img, target_h: int, target_w: int):
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img, {"scale": 1.0, "pad_x": 0, "pad_y": 0}

    scale = min(target_w / w, target_h / h)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=interpolation)

    canvas = np.full((target_h, target_w, img.shape[2]), 255, dtype=img.dtype)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return canvas, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y}
