#!/bin/bash
# Object -> Human on GRAB. INTERACT_DATASETS selects which datasets to load.
# WANDB_MODE=disabled keeps logging local; set to "online" and fill --entity to use W&B.
set -e
export INTERACT_DATASETS="${INTERACT_DATASETS:-grab}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

python trainer_full_body_manip_diffusion.py \
--window=120 \
--batch_size="${BATCH_SIZE:-32}" \
--project="./runs_grab" \
--exp_name="grab_o2h_markerVectorDist_bps256" \
--wandb_pj_name="interact_grab_o2h" \
--entity="none" \
--data_root_folder="../data" \
--use_object_split \
--use_all_data \
--multi_task \
--bps_dim 256 \
--use_bps_info \
--use_human_contact_vec \
--use_human_contact_dist \
--load_last_ckpt \
--save_and_sample_every 10000 \
--n_dec_layers 4 \
--d_k 256 \
--d_v 256 \
--d_model 512
