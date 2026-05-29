#!/usr/bin/env bash
# Ctrl+C traps INT/TERM and stops every rank (setsid + kill process group) so
# conda run / track_image.py subprocesses exit too.

PIDS=()

shutdown_ranks() {
    local pid
    echo "" >&2
    echo "Interrupted — stopping rank processes..." >&2
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            if kill -TERM -"$pid" 2>/dev/null; then
                :
            else
                kill -TERM "$pid" 2>/dev/null || true
            fi
        fi
    done
    for pid in "${PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}

for rank in 0 1 2 3; do
    {
        echo "[$(date '+%F %T')] rank $rank starting (CUDA_VISIBLE_DEVICES=$rank)"
        CUDA_VISIBLE_DEVICES=$rank PYTHONUNBUFFERED=1 setsid python -u track_image.py \
            --data_config ../../configs/_shared/data_val_list.yaml \
            --sam3dgs_root /home/jianjin/SAM3DGS \
            --output_path ./train_data/val_dataset \
            --n_vis_subjects -1 \
            --device cuda:0 \
            --rank $rank \
            --n_rank 4
    } > rank${rank}.log 2>&1 &
    PIDS+=( $! )
done

trap shutdown_ranks INT TERM

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

trap - INT TERM
