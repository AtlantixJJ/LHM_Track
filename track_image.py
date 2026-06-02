# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

import argparse
import json
import os
import pickle
import shutil
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
    prepare_image_sequence_workspace,
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

    def process_image_sequence_visualization(self, image_paths, output_root):
        work_dir = prepare_image_sequence_workspace(
            image_paths,
            output_root,
            overwrite=True,
        )
        self.run_common_stages(
            work_dir,
            output_root,
            fps=1,
            with_flame=False,
            visualize=True,
        )
        return work_dir

    def set_visualize(self, visualize: bool):
        self.opt.visualize = visualize
        self.video2motion.visualize = visualize


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
# subject-level processing
# ---------------------------------------------------------------------------

def process_subject(tracker, subj_key, input_images, data_root, subject_output, visualize):
    """Process all input images for one subject and aggregate into per-subject files.

    Output layout::

        subject_output/
          smplx_params.pkl      {img_rel -> smplx_dict}
          sapiens_pose.npy      {img_rel -> {frame_ids, keypoints, keypoint_scores}}
          flame_params.pkl      {img_rel -> flame_dict}  (only when --with_flame)
          bbox.pkl              {img_rel -> bbox_str}
          samurai_seg/          <img_rel sanitized>.png
          pose_visualized.mp4   subject-level visualization (when visualize=True)

    Each image is processed in an isolated temp dir ``_img_{idx:04d}/`` under
    ``subject_output`` so that stem collisions across views cannot occur.
    Temp dirs are deleted after results are extracted.
    """
    smplx_params = {}
    sapiens_poses = {}
    flame_params = {}
    bboxes = {}
    tmp_roots = []
    processed_image_paths = []

    for img_idx, img_rel in enumerate(input_images):
        img_path = os.path.join(data_root, img_rel)
        if not os.path.exists(img_path):
            print(f"    [WARN] image not found, skipping: {img_path}")
            continue

        # Isolate each image in its own temp root to avoid stem-name collisions.
        tmp_root = os.path.join(subject_output, f"_img_{img_idx:04d}")
        os.makedirs(tmp_root, exist_ok=True)
        tmp_roots.append(tmp_root)

        mask_path = img_path.replace("/input_images/", "/input_masks/").replace("/images/", "/masks/")
        if mask_path == img_path:
            mask_path = None

        print(f"    [{img_idx+1}/{len(input_images)}] {img_rel}" + (" [mask]" if mask_path and os.path.exists(mask_path) else ""))
        tracker.set_visualize(False)
        work_dir = tracker.process_image(img_path, tmp_root, mask_path=mask_path)
        processed_image_paths.append(img_path)

        # collect results
        smplx = _load_smplx(work_dir)
        if smplx is not None:
            smplx_params[img_rel] = smplx

        sap = _load_sapiens(work_dir)
        if sap is not None:
            sapiens_poses[img_rel] = sap

        flame = _load_flame(work_dir)
        if flame is not None:
            flame_params[img_rel] = flame

        bbox = _load_bbox(work_dir)
        if bbox is not None:
            bboxes[img_rel] = bbox

        # copy segmentation mask into subject_output/samurai_seg/
        seg_src = os.path.join(work_dir, "samurai_seg", "00001.png")
        if os.path.exists(seg_src):
            seg_dir = os.path.join(subject_output, "samurai_seg")
            os.makedirs(seg_dir, exist_ok=True)
            key_name = img_rel.replace("/", "_").replace(os.sep, "_")
            shutil.copy2(seg_src, os.path.join(seg_dir, f"{key_name}.png"))

    # save aggregated results
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

    # clean up per-image temp dirs
    for tmp_root in tmp_roots:
        if os.path.isdir(tmp_root):
            shutil.rmtree(tmp_root)

    if visualize and processed_image_paths:
        tracker.set_visualize(True)
        viz_work_dir = tracker.process_image_sequence_visualization(
            processed_image_paths,
            subject_output,
        )
        viz_src = os.path.join(viz_work_dir, "pose_visualized.mp4")
        if os.path.exists(viz_src):
            shutil.move(viz_src, os.path.join(subject_output, "pose_visualized.mp4"))
        shutil.rmtree(viz_work_dir)

    return len(smplx_params)


def process_subject_visualization(tracker, input_images, data_root, subject_output):
    image_paths = []
    for img_rel in input_images:
        img_path = os.path.join(data_root, img_rel)
        if not os.path.exists(img_path):
            print(f"    [WARN] image not found, skipping visualization: {img_path}")
            continue
        image_paths.append(img_path)

    if not image_paths:
        raise ValueError(f"No valid images available for visualization: {subject_output}")

    tracker.set_visualize(True)
    viz_work_dir = tracker.process_image_sequence_visualization(
        image_paths,
        subject_output,
    )
    viz_src = os.path.join(viz_work_dir, "pose_visualized.mp4")
    if not os.path.exists(viz_src):
        raise FileNotFoundError(f"Subject visualization missing: {viz_src}")
    shutil.move(viz_src, os.path.join(subject_output, "pose_visualized.mp4"))
    shutil.rmtree(viz_work_dir)


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
    mask_path = img_path.replace("/input_images/", "/input_masks/").replace("/images/", "/masks/")
    tmp_root = os.path.join(subject_output, f"_img_{img_idx:04d}")
    # Derive work_dir unconditionally so resume detection works in both modes.
    work_dir = os.path.join(tmp_root, image_stem(img_path))

    if skip_folder_check:
        found_mask_path = mask_path if mask_path != img_path else None
    else:
        if not os.path.exists(img_path):
            print(f"    [WARN] image not found, skipping: {img_path}")
            return None

        found_mask_path = None
        if mask_path != img_path and os.path.exists(mask_path):
            found_mask_path = mask_path

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
            if not os.path.exists(seg_dest):
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

    for rec in subj_records:
        if os.path.isdir(rec.tmp_root):
            shutil.rmtree(rec.tmp_root)

    if visualize and processed_image_paths:
        tracker.set_visualize(True)
        viz_work_dir = tracker.process_image_sequence_visualization(
            processed_image_paths, subject_output,
        )
        viz_src = os.path.join(viz_work_dir, "pose_visualized.mp4")
        if os.path.exists(viz_src):
            shutil.move(viz_src, os.path.join(subject_output, "pose_visualized.mp4"))
        shutil.rmtree(viz_work_dir)

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
    ``--n_vis_subjects`` subjects (globally across all datasets).
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
    viz_count = 0

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

        for ds_subj_idx, subj_key in enumerate(subjects):
            cur_idx = global_subj_idx
            global_subj_idx += 1

            if cur_idx % opt.n_rank != opt.rank:
                continue

            visualize = opt.n_vis_subjects < 0 or viz_count < opt.n_vis_subjects
            if visualize:
                viz_count += 1

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
# single-image helpers
# ---------------------------------------------------------------------------

def compare_smplx_json(single_json_path, reference_json_path):
    print(f"\nComparing results:")
    print(f"Single: {single_json_path}")
    print(f"Reference: {reference_json_path}")

    with open(single_json_path, "r") as f:
        single_data = json.load(f)
    with open(reference_json_path, "r") as f:
        ref_data = json.load(f)

    fields = [
        "betas", "root_pose", "body_pose", "jaw_pose",
        "lhand_pose", "rhand_pose", "trans", "focal", "princpt",
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
        print(f"{field:12}: Max Err = {report[field]['max_error']:.6f}, Mean Err = {report[field]['mean_error']:.6f}")
    return report


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
    parser.add_argument("--n_vis_subjects", type=int, default=5,
                        help="Number of subjects (globally) for which to save smplx visualisations; -1 = all")
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

    # single-image-only options
    parser.add_argument("--compare_json", type=str, default=None,
                        help="Reference SMPL-X JSON to compare against (single-image mode)")

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

        if opt.compare_json:
            single_json = os.path.join(work_dir, "smplx_params", "00001.json")
            if os.path.exists(single_json):
                compare_smplx_json(single_json, opt.compare_json)
            else:
                print(f"Error: result not found at {single_json}")
