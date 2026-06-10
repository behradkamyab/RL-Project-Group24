#!/usr/bin/env bash
set -e
MODEL=$1 # e.g. results/sac_push_none_source_1000k
ENV_TYPE=$2 # source or target
for OFFSET in 0 1000 2000; do
 echo "=== eval offset=$OFFSET ==="
 python eval_sb3.py --model-path "$MODEL" --episodes 50 \
 --env-type "$ENV_TYPE" --seed-offset $OFFSET \
 | tee -a "${MODEL##*/}.${ENV_TYPE}.evallog.txt"
done
