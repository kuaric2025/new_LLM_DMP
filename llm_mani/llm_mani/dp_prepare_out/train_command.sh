#!/usr/bin/env bash
set -e

# 1) fix common dependency mismatch for diffusion_policy
cd "/home/mhumais/Huang/diffusion_policy" && python -m pip install 'huggingface_hub<0.26' 'diffusers==0.11.1' 'numba>=0.59,<0.61'

# 2) disable xformers/flash-attn to avoid import/runtime conflicts
cd "/home/mhumais/Huang/diffusion_policy" && python -m pip uninstall -y xformers flash-attn || true

# 3) launch training
cd "/home/mhumais/Huang/diffusion_policy" && PYTHONPATH="/home/mhumais/Huang/DMP:$PYTHONPATH" XFORMERS_DISABLED=1 HYDRA_FULL_ERROR=1 python train.py --config-name=train_diffusion_unet_lowdim_workspace task=custom_lowdim_xyzrpy horizon=144 n_obs_steps=2 n_action_steps=8 training.num_epochs=1000 training.device=cuda:0 logging.mode=offline checkpoint.topk.monitor_key=val_loss checkpoint.topk.mode=min
