#!/usr/bin/env bash
# Queue several run_all_tasks.py sweeps on one GPU, in order.
#
#   ./run_sweeps.sh configllama3b configqwen           # both on cuda:0, seeds 25,99
#   DEVICE=cuda:1 ./run_sweeps.sh configllama8b
#   SEEDS=25,99,42 METHODS='[cie,lrp,if_patching]' ./run_sweeps.sh configllama3b
#   DRY=1 ./run_sweeps.sh configllama3b configqwen     # print the commands, run nothing
#
# Detached:
#   nohup ./run_sweeps.sh configllama3b configqwen > results/sweep_queue.log 2>&1 &
#
# Each sweep gets its own timestamped log under results/. A failing sweep does not
# cancel the ones queued behind it; its exit code is recorded and the queue moves on.
#
# The device is always passed explicitly rather than left to the per-config
# default, since two configs pinning the same GPU is how queued runs end up
# fighting over one device. Override with DEVICE=cuda:N.

set -u

DEVICE=${DEVICE:-cuda:0}
SEEDS=${SEEDS:-25,99}
METHODS=${METHODS:-}   # e.g. '[cie,lrp,if_patching]'; empty keeps conf/base.yaml
DRY=${DRY:-0}
PY=${PY:-.venv/bin/python}

cd "$(dirname "$0")" || exit 1
mkdir -p results

if [ $# -eq 0 ]; then
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    exit 1
fi

echo "queue: $* | device=$DEVICE seeds=$SEEDS methods=${METHODS:-<config default>}"
failed=0

for cfg in "$@"; do
    if [ ! -f "conf/$cfg.yaml" ]; then
        echo "!! no conf/$cfg.yaml -- skipping"
        failed=$((failed + 1))
        continue
    fi

    log="results/sweep_${cfg#config}_$(date +%Y%m%d_%H%M%S).log"
    args=(--config-name "$cfg" --multirun "seed=$SEEDS" "device=$DEVICE")
    [ -n "$METHODS" ] && args+=("methods=$METHODS")

    if [ "$DRY" = "1" ]; then
        echo "would run: $PY -u run_all_tasks.py ${args[*]}  > $log"
        continue
    fi

    echo "=== $(date '+%F %T')  start $cfg  -> $log"
    start=$SECONDS
    "$PY" -u run_all_tasks.py "${args[@]}" > "$log" 2>&1
    rc=$?
    mins=$(((SECONDS - start) / 60))
    echo "=== $(date '+%F %T')  done  $cfg  exit=$rc after ${mins}min"
    [ $rc -ne 0 ] && failed=$((failed + 1)) && echo "    see $log"
done

echo "queue finished with $failed failure(s)"
exit $((failed > 0))