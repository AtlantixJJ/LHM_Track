#!/usr/bin/env bash
# set -m gives every background job its own process group so that
# kill -PGID reaches python and all os.system/subprocess children.
set -m

PIDS=()

shutdown_ranks() {
    echo "" >&2
    echo "Interrupted — stopping rank processes..." >&2
    for pid in "${PIDS[@]:-}"; do
        kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}

trap shutdown_ranks INT TERM

for rank in 0 1 2 3; do
    (
        echo "[$(date '+%F %T')] rank $rank starting (CUDA_VISIBLE_DEVICES=$rank)"
        exec env CUDA_VISIBLE_DEVICES=$rank PYTHONUNBUFFERED=1 python -u track_image.py \
            --data_config ../../configs/_shared/data_val_list.yaml \
            --sam3dgs_root /home/jianjin/SAM3DGS \
            --output_path ./train_data/val_dataset \
            --n_vis_subjects 5 \
            --device cuda:0 \
            --rank $rank \
            --n_rank 4
    ) > rank${rank}.log 2>&1 &
    PIDS+=( $! )
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

trap - INT TERM
