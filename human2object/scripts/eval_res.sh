#!/bin/bash
# Human -> Object: sample results for quantitative evaluation.
# Model flags MUST match train.sh or the checkpoint will not load.
set -e
export INTERACT_DATASETS="${INTERACT_DATASETS:-grab}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export INTERACT_NUM_WORKERS="${INTERACT_NUM_WORKERS:-2}"
MILESTONE="${MILESTONE:-1}"
EXP_NAME="grab_h2o_bps256"

python trainer_full_body_manip_diffusion.py \
--window=120 \
--batch_size="${BATCH_SIZE:-32}" \
--project="./runs_grab" \
--exp_name="$EXP_NAME" \
--wandb_pj_name="interact_grab_h2o_eval" \
--entity="none" \
--data_root_folder="../data" \
--multi_task \
--use_all_data \
--bps_dim 256 \
--use_bps_info \
--use_human_contact_dist \
--w_humanContactDist 0.5 \
--token_obj_geom \
--use_quat \
--split_train_val \
--save_and_sample_every 10000 \
--test_sample_res \
--for_quant_eval \
--eval_save_path="./eval_out" \
--milestone "$MILESTONE" \
--n_dec_layers 4 \
--d_k 256 \
--d_v 256 \
--d_model 512

echo "Samples written to ./eval_out/$EXP_NAME/ms_$MILESTONE"
