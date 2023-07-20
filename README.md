# Chinese-Llama-2

<!-- **Authors:** -->

**_¹² <sup>&dagger;</sup>[Zefeng Du](https://seeledu.github.io/index-en.html), ³ <sup>&dagger;</sup>[Minghao Wu](https://minghao-wu.github.io/), ¹ <sup>*</sup> [Longyue Wang](http://www.longyuewang.com/)_**


<!-- **Affiliations:** -->

_¹ Tencent AI Lab,  ²University of Macau, ³ Monash University_,

_<sup>&dagger;</sup>equal contribution_

_<sup>*</sup>Longyue Wang is the corresponding author: [vinnlywang@tencent.com](mailto:{vinnlywang@tencent.com)_
</div>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)]()

## News
* [2023.07.20] :rocket: We fine-tune the Llama-2 on the Chinese instruction dataset using LoRA technique, known as Chinese-Llama-2-LoRA, and release the [Chinese-Llama-2-LoRA-7B](some link).
* [2023.07.18] :tada::tada::tada: [Llama-2 is announced!](https://ai.meta.com/llama/)

## Overview

Chinese-Llama-2 is a project that aims to expand the impressive capabilities of the Llama-2 language model to the Chinese language. Developed by MetaAI, Llama-2 has already proven to be a powerful language model. In this project, we focus on three key areas of research:

1. Parameter-efficient fine-tuning: We employ the [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) technique to fine-tune Llama-2 specifically for the Chinese instruction dataset. This approach optimizes the model's performance while minimizing the number of required parameters.

2. Full instruction fine-tuning: We fine-tune all parameters of Llama-2 on the Chinese instruction dataset. By allowing the model to adapt fully to the characteristics of the Chinese language, we enhance its proficiency and accuracy in generating Chinese text.

3. Continued pre-training: To further enhance Llama-2's Chinese language understanding, we continue its pre-training on large-scale Chinese corpora. By exposing the model to vast amounts of Chinese text data, we enable it to capture intricate linguistic patterns and nuances, resulting in improved language generation.

This repository contains all the necessary code and resources to implement the aforementioned areas of research, facilitating experimentation and advancement in Chinese natural language processing using the Llama-2 model.


## Installation

To use Chinese-Llama-2, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/longyuewangdcu/chinese-llama-2.git
   ```

2. Change into the project directory:

   ```bash
   cd chinese-llama-2
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```


## Parameter-Efficient Fine-Tuning

### Model Checkpoints

The LoRA weights for [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) can be found at here. More model weights will be available soon.

### Using the Model
You use our fine-tuned model as follows:
```python
import transformers
# TODO: ADD THE INFERENCE CODE
```

### Showcases


### Datasets
We finetune the Llama-2 on the Chinese instruction dataset [COIG](https://huggingface.co/datasets/BAAI/COIG) and 200K Chinese-English parallel documents translation data. 


### Fine-Tuning

To finetune Llama-2 using LoRA on the Chinese instruction dataset, Run the command to start lora finetune.:
```bash
# Multi-nodes are also supported

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

train_path=transformers/examples/pytorch/language-modeling/run_clm_lora.py

model_path=$path/model/llama2-7B-HF
model_save=$path/checkpoint/chinese-llama2-7b-4096-enzh/

torchrun --nnodes 1 --node_rank $INDEX --nproc_per_node 8 \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
  ${train_path} \
  --deepspeed $path/train/deepspeed_config_bf16.json \
  --model_name_or_path ${model_path} \
  --train_file $path/data/instruction/all_instruction_hf.json \
  --validation_file $path/data/instruction/all_instruction_hf_dev.json \
  --preprocessing_num_workers 32 \
  --dataloader_num_workers 16 \
  --dataloader_pin_memory True \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --block_size 4096 \
  --use_lora True \
  --lora_config $path/train/lora_config.json \
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
  --output_dir ${model_save}
```

## TODO

1. Full instruction fine-tuning
2. Continued pre-training

Stay tuned!

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. We appreciate your contributions to make Chinese-Llama-2 even better.


## Acknowledgments

Chinese-Llama-2 builds upon the Llama-2 developed by MetaAI. We would like to express our gratitude to the MetaAI team for their outstanding work and contributions to the field of natural language processing.
