# Multi-nodes are also supported
# use flash attention to lower memory usage
pip install flash-attn==1.0.4

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

export HF_HOME=
export TRANSFORMERS_CACHE=
path= # path to llama2-chinese
train_path=$path/train/run_clm_llms_mem.py
model_path=$path/model/llama2-7B-HF # place original model here
model_save=$path/checkpoint/llama2-7b-llama2_coig_dt_ca-all/

# MASTER_ADDR set to localhost
HOST_NUM=2
torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 8 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed $path/train/deepspeed_config_bf16.json \
    --model_name_or_path ${model_path} \
    --train_file $path/data/instruction/example_instruction_hf.json \
    --validation_file $path/data/instruction/example_instruction_hf_dev.json \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 4096 \
    --do_train \
    --bf16 True \
    --bf16_full_eval True \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --streaming \
    --ddp_timeout 72000 \
    --seed 1 \
    --overwrite_output_dir\
    --gradient_checkpointing True \
    --output_dir ${model_save}\
