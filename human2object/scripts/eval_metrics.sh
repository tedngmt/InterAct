#!/bin/bash
# Human -> Object: metrics over already-sampled results.
# Run eval_res.sh first; MILESTONE must match that run.
set -e
MILESTONE="${MILESTONE:-1}"
EXP_NAME="grab_h2o_bps256"
MOTION_PATH="${MOTION_PATH:-./eval_out/$EXP_NAME/ms_$MILESTONE}"

if [ ! -d "$MOTION_PATH" ]; then
  echo "No sampled results at $MOTION_PATH" >&2
  echo "Run: MILESTONE=$MILESTONE bash ./scripts/eval_res.sh" >&2
  exit 1
fi

python eval_saved.py --motion_path "$MOTION_PATH" --best_metric mpmpe
python eval_saved.py --motion_path "$MOTION_PATH" --best_metric contact_f1_score
