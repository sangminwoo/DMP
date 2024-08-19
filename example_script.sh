#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1

# Possibile Hyper-Parameters to be modified. Please check config/base_config.yaml to set as default parameters.

# Common Parameters
dataset="FFHQ"
scheduling="cosine"
loss_weight_type="constant" # constant, min_snr_5(or, min_snr_* is also available), uw
gpu_offset=0 # GPU ID where the job starts
model_name="DiT-B/2"

# Train Parameters
DATA_PATH="" # default is null, so you have to set the data_path or errors will raise up!
PRETRAINED_PATH=""
iterations=50000

# Sample Parameters
num_fid_samples=50000 # 10000, 50000 (commonly used)
cfg_scale=1.5 # 1.0 1.5 (commonly used)
version=0 # default : 0, have to set if you have multiple versions
eval_at=50000 # default : train iteration, have to set this for iteration-wise evaluation


# Train Script
torchrun --nnodes=1 --nproc_per_node=4 --master_port 8773 train.py \
data=${dataset} \
general.data_path=${DATA_PATH} \
general.pretrained_path=${PRETRAINED_PATH} \
general.iterations=${iterations} \
general.schedule_name=${scheduling} \
general.loss_weight_type=${loss_weight_type} \
models.name=${model_name} \
general.gpu_offset=${gpu_offset}

######### Sample Script, argument "loss_weight" is required to set the ckpt path
# torchrun --nnodes=1 --nproc_per_node=1 --master_port 1558 sample_ddp.py \
# data=${dataset} \
# general.schedule_name=${scheduling} \
# general.loss_weight_type=${loss_weight_type} \
# models.name=${model_name} \
# eval.num_fid_samples=${num_fid_samples} \
# eval.cfg_scale=${cfg_scale} \
# eval.ckpt_path.version=${version} \
# eval.ckpt_path.iterations=${eval_at} \
# general.gpu_offset=${gpu_offset}

# Sample Only few samples (i.e., sample.py)
#CUDA_VISIBLE_DEVICES=${gpu_offset} python sample.py \
#data=${dataset} \
#general.schedule_name=${scheduling} \
#general.loss_weight_type=${loss_weight_type} \
#models.name=${model_name} \
#models.routing.init_method=${init_method} \
#models.routing.sharing_ratio=${sharing_ratio} \
#eval.cfg_scale=${cfg_scale} \
#eval.ckpt_path.version=${version} \
#eval.ckpt_path.iterations=${eval_at}
