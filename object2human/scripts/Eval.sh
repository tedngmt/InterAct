#!/bin/bash
# Object -> Human: metrics over already-sampled results.
# Run Sample_markerContact_VecDist.sh first; MILESTONE must match that run.
set -e
MILESTONE="${MILESTONE:-1}"
EXP_NAME="grab_o2h_markerVectorDist_bps256"
MOTION_PATH="${MOTION_PATH:-./eval_out/$EXP_NAME/ms_$MILESTONE}"

if [ ! -d "$MOTION_PATH" ]; then
  echo "No sampled results at $MOTION_PATH" >&2
  echo "Run: MILESTONE=$MILESTONE bash ./scripts/Sample_markerContact_VecDist.sh" >&2
  exit 1
fi

python eval_saved.py --motion_path "$MOTION_PATH" --best_metric full_jpe --contact_thre 0.05
python eval_saved.py --motion_path "$MOTION_PATH" --best_metric full_contact_f1_score --contact_thre 0.05
