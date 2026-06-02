# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

import argparse
import os
import pickle
import shutil
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataclasses import dataclass

from engine.pose_estimation.video2motion import Video2MotionPipeline
from engine.predict_box import init_box_model, predict_box
from engine.predict_flame import estimate_flame, init_gaga_track
from engine.predict_samurai import run_samurai
from engine.predict_sapiens_pose import run_sapiens_batch
from track_image_utils import (
    assert_valid_image_path,
    image_stem,
    letterbox_image,
    prepare_single_image_workspace,
)
from track_video import BaseTracker


def _resolve_mask_path(mask_path, img_path):
    """Return the existing mask path, trying .png extension when the original doesn't exist."""
    if mask_path == img_path:
        return None
    if os.path.exists(mask_path):
        return mask_path
    base = os.path.splitext(mask_path)[0]
    png_path = base + ".png"
    if png_path != mask_path and os.path.exists(png_path):
        return png_path
    return None


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

    def process_image(self, image_path, output_root, mask_path=None):
        """Run full pipeline on one image; returns the work_dir path.

        If *mask_path* points to an existing file it is used as the segmentation
        mask (copied to ``samurai_seg/00001.png``) and the Samurai tracking step
        is skipped.
        """
        assert_valid_image_path(image_path)
        work_dir = prepare_single_image_workspace(
            image_path, output_root, overwrite=True
        )
        skip_samurai = False
        if mask_path and os.path.exists(mask_path):
            seg_dir = os.path.join(work_dir, "samurai_seg")
            os.makedirs(seg_dir, exist_ok=True)
            shutil.copy2(mask_path, os.path.join(seg_dir, "00001.png"))
            skip_samurai = True
        self.run_common_stages(
            work_dir,
            output_root,
            fps=1,
            with_flame=self.opt.with_flame,
            visualize=self.opt.visualize,
            skip_samurai=skip_samurai,
        )
        return work_dir

    def visualize_from_records(self, subj_records, subject_output):
        """Generate pose visualization reusing already-computed stage results.

        Combines per-image sapiens keypoints and segmentation masks into a
        sequence workspace, then calls video2motion with visualize=True,
        skipping predict_box and sapiens which were already run per-image.
        Returns the work_dir path, or None if no valid frames were found.
        """
        work_dir = os.path.join(subject_output, "_subject_visualization")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        imgs_png_dir = os.path.join(work_dir, "imgs_png")
        seg_dir = os.path.join(work_dir, "samurai_seg")
        sap_dir = os.path.join(work_dir, "sapiens_pose")
        os.makedirs(imgs_png_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(sap_dir, exist_ok=True)

        sap_frame_ids = []
        sap_keypoints = []
        sap_scores = []
        valid_idx = 0
        target_h, target_w = None, None

        for rec in subj_records:
            img_src = os.path.join(rec.work_dir, "imgs_png", "00001.png")
            if not os.path.exists(img_src):
                img_src = rec.img_path

            seg_src = os.path.join(rec.work_dir, "samurai_seg", "00001.png")
            if not os.path.exists(seg_src):
                if rec.mask_path and os.path.exists(rec.mask_path):
                    seg_src = rec.mask_path
                else:
                    continue

            sap_src = os.path.join(rec.work_dir, "sapiens_pose", "sapiens_pose.npy")
            if not os.path.exists(sap_src):
                continue

            img = cv2.imread(img_src)
            if img is None:
                continue

            valid_idx += 1
            frame_name = f"{valid_idx:05d}.png"

            orig_h, orig_w = img.shape[:2]
            if target_h is None:
                target_h, target_w = orig_h, orig_w

            normalized, _ = letterbox_image(img, target_h, target_w)
            cv2.imwrite(os.path.join(imgs_png_dir, frame_name), normalized)
            shutil.copy2(seg_src, os.path.join(seg_dir, frame_name))

            sap_data = np.load(sap_src, allow_pickle=True).item()
            sap_frame_ids.append(valid_idx)
            sap_keypoints.append(sap_data["keypoints"][0])
            sap_scores.append(sap_data["keypoint_scores"][0])

        if valid_idx == 0:
            return None

        np.save(os.path.join(sap_dir, "sapiens_pose.npy"), {
            "frame_ids": np.array(sap_frame_ids, dtype=np.int32),
            "keypoints": np.stack(sap_keypoints),
            "keypoint_scores": np.stack(sap_scores),
        })

        self.video2motion.visualize = True
        try:
            self.video2motion(work_dir, subject_output, fps=1)
        finally:
            self.video2motion.visualize = False

        return work_dir

# ---------------------------------------------------------------------------
# per-image result loaders
# ---------------------------------------------------------------------------

def _load_smplx(work_dir):
    p = os.path.join(work_dir, "smplx_params.npy")
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True).item()


def _load_sapiens(work_dir):
    p = os.path.join(work_dir, "sapiens_pose", "sapiens_pose.npy")
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True).item()


def _load_flame(work_dir):
    p = os.path.join(work_dir, "flame_params.npy")
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True).item()


def _load_bbox(work_dir):
    p = os.path.join(work_dir, "bbox", "first_frame.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# batch runner helpers
# ---------------------------------------------------------------------------

@dataclass
class _ImageRecord:
    img_path: str
    img_rel: str
    tmp_root: str        # parent dir; work_dir = tmp_root/stem(img_path)
    work_dir: str
    skip_samurai: bool   # True when a pre-existing mask was pre-populated
    subj_key: str
    ds_key: str
    subject_output: str
    mask_path: str = None   # original external mask (set when skip_samurai=True)
    sapiens_done: bool = False  # True when sapiens_pose.npy already exists (resume)


def _prepare_record(img_path, img_rel, img_idx, subject_output, ds_key, subj_key,
                    skip_folder_check=False):
    """Create per-image workspace and return an _ImageRecord, or None on failure."""
    _raw_mask_path = img_path.replace("/input_images/", "/input_masks/").replace("/images/", "/masks/")
    tmp_root = os.path.join(subject_output, f"_img_{img_idx:04d}")
    # Derive work_dir unconditionally so resume detection works in both modes.
    work_dir = os.path.join(tmp_root, image_stem(img_path))

    if skip_folder_check:
        # Always stat the mask to resolve the correct extension (e.g. .png vs .jpg).
        found_mask_path = _resolve_mask_path(_raw_mask_path, img_path)
    else:
        if not os.path.exists(img_path):
            print(f"    [WARN] image not found, skipping: {img_path}")
            return None

        found_mask_path = _resolve_mask_path(_raw_mask_path, img_path)

    # Check for a resumable previous run regardless of skip_folder_check.
    sapiens_done = os.path.exists(
        os.path.join(work_dir, "sapiens_pose", "sapiens_pose.npy")
    )

    if not skip_folder_check and not sapiens_done:
        assert_valid_image_path(img_path)
        # When a mask exists the pipeline stages receive img_path / mask_path
        # directly, so there is no need to copy the image into imgs_png/.
        work_dir = prepare_single_image_workspace(
            img_path, tmp_root, overwrite=True,
            skip_imgs_png=(found_mask_path is not None),
        )

    return _ImageRecord(
        img_path=img_path,
        img_rel=img_rel,
        tmp_root=tmp_root,
        work_dir=work_dir,
        skip_samurai=(found_mask_path is not None) or sapiens_done,
        subj_key=subj_key,
        ds_key=ds_key,
        subject_output=subject_output,
        mask_path=found_mask_path,
        sapiens_done=sapiens_done,
    )


def _run_stages_bulk(tracker, records):
    """Run predict_box, samurai, and sapiens stages across all records in bulk.

    Stage order: predict_box → samurai → sapiens.  video2motion and flame are
    intentionally omitted here; they are run per-subject in Phase C so each
    subject can be visualised as soon as it finishes.
    """
    if not records:
        return

    samurai_records = [r for r in records if not r.skip_samurai]
    sapiens_records = [r for r in records if not r.sapiens_done]
    n_skip_samurai = len(records) - len(samurai_records)
    n_skip_sapiens = len(records) - len(sapiens_records)
    print(f"  [BULK] samurai: {len(samurai_records)} images ({n_skip_samurai} skipped)")
    print(f"  [BULK] sapiens: {len(sapiens_records)} images ({n_skip_sapiens} already done)")

    for rec in tqdm(samurai_records, desc="predict_box"):
        predict_box(tracker.sam2seg, rec.work_dir, img_path=rec.img_path)
    for rec in tqdm(samurai_records, desc="samurai"):
        run_samurai(tracker.model_path, rec.work_dir, visualize=False)

    if sapiens_records:
        run_sapiens_batch(
            tracker.model_path,
            [rec.work_dir for rec in sapiens_records],
            img_paths=[rec.img_path for rec in sapiens_records],
            seg_paths=[rec.mask_path for rec in sapiens_records],
            chunk_size=tracker.opt.sapiens_chunk_size,
        )


def _aggregate_subject(subj_records, subject_output, visualize, tracker, use_symlink=False):
    """Collect per-image results into subject-level files and clean up temp dirs."""
    smplx_params = {}
    sapiens_poses = {}
    flame_params = {}
    bboxes = {}
    processed_image_paths = []

    for rec in subj_records:
        smplx = _load_smplx(rec.work_dir)
        if smplx is not None:
            smplx_params[rec.img_rel] = smplx

        sap = _load_sapiens(rec.work_dir)
        if sap is not None:
            sapiens_poses[rec.img_rel] = sap

        flame = _load_flame(rec.work_dir)
        if flame is not None:
            flame_params[rec.img_rel] = flame

        bbox = _load_bbox(rec.work_dir)
        if bbox is not None:
            bboxes[rec.img_rel] = bbox

        seg_dir = os.path.join(subject_output, "samurai_seg")
        key_name = rec.img_rel.replace("/", "_").replace(os.sep, "_")
        seg_dest = os.path.join(seg_dir, f"{key_name}.png")
        if use_symlink:
            assert rec.skip_samurai and rec.mask_path, (
                f"use_symlink requires skip_samurai=True and a known mask_path, got: {rec.img_rel}"
            )
            os.makedirs(seg_dir, exist_ok=True)
            if not os.path.lexists(seg_dest):
                os.symlink(rec.mask_path, seg_dest)
        else:
            seg_src = os.path.join(rec.work_dir, "samurai_seg", "00001.png")
            if os.path.exists(seg_src):
                os.makedirs(seg_dir, exist_ok=True)
                shutil.copy2(seg_src, seg_dest)

        processed_image_paths.append(rec.img_path)

    if smplx_params or sapiens_poses or flame_params or bboxes:
        os.makedirs(subject_output, exist_ok=True)
    if smplx_params:
        with open(os.path.join(subject_output, "smplx_params.pkl"), "wb") as f:
            pickle.dump(smplx_params, f)
    if sapiens_poses:
        np.save(os.path.join(subject_output, "sapiens_pose.npy"), sapiens_poses)
    if flame_params:
        with open(os.path.join(subject_output, "flame_params.pkl"), "wb") as f:
            pickle.dump(flame_params, f)
    if bboxes:
        with open(os.path.join(subject_output, "bbox.pkl"), "wb") as f:
            pickle.dump(bboxes, f)

    # Visualize before cleanup so per-image work_dirs are still available.
    if visualize and processed_image_paths:
        viz_work_dir = tracker.visualize_from_records(subj_records, subject_output)
        if viz_work_dir:
            viz_src = os.path.join(viz_work_dir, "pose_visualized.mp4")
            if os.path.exists(viz_src):
                shutil.move(viz_src, os.path.join(subject_output, "pose_visualized.mp4"))
            shutil.rmtree(viz_work_dir)

    for rec in subj_records:
        if os.path.isdir(rec.tmp_root):
            shutil.rmtree(rec.tmp_root)

    return len(smplx_params)


# ---------------------------------------------------------------------------
# batch runner
# ---------------------------------------------------------------------------

def run_batch(tracker, opt):
    """Process all subjects in all datasets from a data_val_list.yaml config.

    Phase A: collect all per-image records and prepare workspaces.
    Phase B: predict_box → samurai → sapiens in bulk across all images.
    Phase C: for each subject: video2motion → flame → aggregate → visualize.
    Running Phase B in bulk avoids reloading model weights for each image.
    Running Phase C per-subject allows visualisation immediately after each
    subject finishes rather than waiting for the entire dataset.

    Resume: subjects where both ``smplx_params.pkl`` and ``sapiens_pose.npy``
    already exist are skipped.  Images where ``sapiens_pose/sapiens_pose.npy``
    exists in the temp workspace skip predict_box/samurai/sapiens.  Images
    where ``smplx_params.npy`` exists skip video2motion.

    Visualisation (smplx render + overlay) is produced for the first
    ``--n_vis_subjects`` subjects per dataset.
    Pass ``-1`` to visualise every subject.

    With ``--rank R --n_rank N`` only subjects whose global index satisfies
    ``index % N == R`` are processed, enabling multi-process parallelism.
    """
    sam3dgs_root = opt.sam3dgs_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    with open(opt.data_config) as f:
        cfg = yaml.safe_load(f)

    ds_list = cfg["data"]["val"]["ds_list"]
    ds_keys = cfg["data"]["val"].get("ds_keys", [f"ds{i}" for i in range(len(ds_list))])

    global_subj_idx = 0

    # -----------------------------------------------------------------------
    # Phase A: collect records and prepare per-image workspaces
    # -----------------------------------------------------------------------
    all_records = []       # flat list of _ImageRecord across all subjects
    subj_meta_list = []    # per-subject metadata for Phase C aggregation

    for ds_key, ds_cfg in zip(ds_keys, ds_list):
        image_list_path = os.path.join(sam3dgs_root, ds_cfg["image_list"])
        if not os.path.exists(image_list_path):
            print(f"[WARN] image_list not found, skipping: {image_list_path}")
            continue

        with open(image_list_path) as f:
            val_list = yaml.safe_load(f)

        data_root = os.path.join(sam3dgs_root, val_list["data_root"])
        subjects = list(val_list["items"].keys())
        print(f"\n=== Dataset: {ds_key}  ({len(subjects)} subjects) ===")

        ds_viz_count = 0  # reset per dataset
        for ds_subj_idx, subj_key in enumerate(subjects):
            cur_idx = global_subj_idx
            global_subj_idx += 1

            if cur_idx % opt.n_rank != opt.rank:
                continue

            visualize = opt.n_vis_subjects < 0 or ds_viz_count < opt.n_vis_subjects
            if visualize:
                ds_viz_count += 1

            subj_dir = subj_key.replace("/", "_").replace(os.sep, "_")
            subject_output = os.path.join(opt.output_path, ds_key, subj_dir)

            done_pkl = os.path.join(subject_output, "smplx_params.pkl")
            done_npy = os.path.join(subject_output, "sapiens_pose.npy")
            if os.path.exists(done_pkl) and os.path.exists(done_npy):
                print(f"  [SKIP] {subj_key} (already done)")
                continue

            subj_item = val_list["items"][subj_key]
            input_images = subj_item.get("input_images", [])
            extra_images = subj_item.get("images", [])
            seen = set(input_images)
            all_images = list(input_images) + [img for img in extra_images if img not in seen]
            n_extra = len(all_images) - len(input_images)

            print(
                f"  Queuing [{ds_subj_idx+1}/{len(subjects)}] {subj_key}"
                f"  images={len(all_images)} (input={len(input_images)} extra={n_extra})"
                f"  visualize={visualize}"
                + (f"  rank={opt.rank}/{opt.n_rank}" if opt.n_rank > 1 else "")
            )

            range_start = len(all_records)
            for img_idx, img_rel in enumerate(tqdm(all_images, desc=f"prepare {subj_key}", leave=False)):
                img_path = os.path.join(data_root, img_rel)
                rec = _prepare_record(
                    img_path, img_rel, img_idx, subject_output, ds_key, subj_key,
                    skip_folder_check=opt.skip_folder_check,
                )
                if rec is not None:
                    all_records.append(rec)
            range_end = len(all_records)

            subj_meta_list.append({
                "subj_key": subj_key,
                "ds_key": ds_key,
                "subject_output": subject_output,
                "visualize": visualize,
                "input_count": len(input_images),
                "range": (range_start, range_end),
            })

    if not all_records:
        print("No images to process.")
        return

    print(f"\n[BULK] {len(all_records)} images across {len(subj_meta_list)} subjects")

    # -----------------------------------------------------------------------
    # Phase B: run predict_box, samurai, sapiens across all images in bulk
    # -----------------------------------------------------------------------
    _run_stages_bulk(tracker, all_records)

    # -----------------------------------------------------------------------
    # Phase C: per-subject video2motion → flame → aggregate → visualize
    # -----------------------------------------------------------------------
    print("\n[SUBJECTS] Running video2motion + flame per subject...")
    for meta in tqdm(subj_meta_list, desc="subjects"):
        start, end = meta["range"]
        subj_records = all_records[start:end]

        for rec in subj_records:
            if not os.path.exists(os.path.join(rec.work_dir, "smplx_params.npy")):
                tracker.video2motion(rec.work_dir, rec.tmp_root, fps=1,
                                     img_path=rec.img_path, seg_path=rec.mask_path)

        if opt.with_flame and tracker.gaga_track is not None:
            for rec in subj_records:
                if not os.path.exists(os.path.join(rec.work_dir, "flame_params.npy")):
                    estimate_flame(tracker.gaga_track, rec.work_dir)

        n_done = _aggregate_subject(
            subj_records, meta["subject_output"], meta["visualize"], tracker,
            use_symlink=opt.use_symlink,
        )
        tqdm.write(f"  {meta['subj_key']}: {n_done}/{meta['input_count']} images")


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------

def get_parse():
    parser = argparse.ArgumentParser(description="LHM Single-Image / Batch Pose Estimation")

    # mode selection
    parser.add_argument("--image_path", type=str, default=None,
                        help="Single-image mode: path to one image file")
    parser.add_argument("--data_config", type=str, default=None,
                        help="Batch mode: path to data_val_list.yaml")

    # batch-mode options
    parser.add_argument("--sam3dgs_root", type=str, default=None,
                        help="SAM3DGS repo root for resolving relative paths (default: ../../ from this file)")
    parser.add_argument("--n_vis_subjects", type=int, default=10,
                        help="Number of subjects per dataset for which to save smplx visualisations; -1 = all")
    parser.add_argument("--rank", type=int, default=0,
                        help="This process rank for multi-process splitting (0-indexed)")
    parser.add_argument("--n_rank", type=int, default=1,
                        help="Total number of parallel processes")

    # common options
    parser.add_argument("--output_path", type=str, default="./train_data/custom_image")
    parser.add_argument("--model_path", type=str, default="./pretrained_models")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument("--visualize", action="store_true",
                        help="Enable smplx render + overlay visualisation (single-image mode)")
    parser.add_argument("--with_flame", action="store_true",
                        help="Also estimate FLAME face parameters")
    parser.add_argument("--use_symlink", action="store_true",
                        help="Symlink masks instead of copying (requires all images to have pre-existing masks)")
    parser.add_argument("--skip_folder_check", action="store_true",
                        help="Skip os.path.exists checks during record preparation (assumes images and masks exist)")
    parser.add_argument("--sapiens_chunk_size", type=int, default=500,
                        help="Max images per sapiens subprocess call to avoid OOM (default: 500)")

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    opt = get_parse()
    assert torch.cuda.is_available(), "CUDA is not available"
    assert opt.image_path or opt.data_config, "Provide either --image_path or --data_config"
    assert not (opt.image_path and opt.data_config), "--image_path and --data_config are mutually exclusive"
    assert 0 <= opt.rank < opt.n_rank, f"--rank must be in [0, n_rank): got {opt.rank}/{opt.n_rank}"

    print(
        f"[rank {opt.rank}/{opt.n_rank}] Loading models on {opt.device}...",
        flush=True,
    )
    tracker = ImageTracker(opt.model_path, opt.device, opt)

    if opt.data_config:
        run_batch(tracker, opt)
    else:
        os.makedirs(opt.output_path, exist_ok=True)
        work_dir = tracker.process_image(opt.image_path, opt.output_path)
        print(f"Finish processing image: {opt.image_path}")
        print(f"Output: {work_dir}")

