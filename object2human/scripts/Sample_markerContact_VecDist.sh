#!/bin/bash
# Object -> Human: sample results for quantitative evaluation.
# Model flags MUST match Train_markerContact_VecDist.sh or the checkpoint will not load.
# MILESTONE selects which saved checkpoint to sample from (see runs_grab/.../weights).
set -e
export INTERACT_DATASETS="${INTERACT_DATASETS:-grab}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export INTERACT_NUM_WORKERS="${INTERACT_NUM_WORKERS:-2}"
MILESTONE="${MILESTONE:-1}"
EXP_NAME="grab_o2h_markerVectorDist_bps256"

python trainer_full_body_manip_diffusion.py \
--window=120 \
--batch_size="${BATCH_SIZE:-32}" \
--project="./runs_grab" \
--exp_name="$EXP_NAME" \
--wandb_pj_name="interact_grab_o2h_eval" \
--entity="none" \
--data_root_folder="../data" \
--use_object_split \
--use_all_data \
--multi_task \
--bps_dim 256 \
--use_bps_info \
--use_human_contact_vec \
--use_human_contact_dist \
--milestone "$MILESTONE" \
--test_sample_res \
--for_quant_eval \
--eval_save_path="./eval_out" \
--n_dec_layers 4 \
--d_k 256 \
--d_v 256 \
--d_model 512

echo "Samples written to ./eval_out/$EXP_NAME/ms_$MILESTONE"
