#!/bin/bash
# Human -> Object on GRAB. See object2human/scripts/Train_markerContact_VecDist.sh for notes.
set -e
export INTERACT_DATASETS="${INTERACT_DATASETS:-grab}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

python trainer_full_body_manip_diffusion.py \
--window=120 \
--batch_size="${BATCH_SIZE:-32}" \
--project="./runs_grab" \
--exp_name="grab_h2o_bps256" \
--wandb_pj_name="interact_grab_h2o" \
--entity="none" \
--data_root_folder="../data" \
--multi_task \
--use_all_data \
--bps_dim 256 \
--use_bps_info \
--use_human_contact_dist \
--w_humanContactDist 0.5 \
--token_obj_geom \
--load_last_ckpt \
--use_quat \
--split_train_val \
--save_and_sample_every 10000 \
--n_dec_layers 4 \
--d_k 256 \
--d_v 256 \
--d_model 512
