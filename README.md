# Chinese-Llama-2-LoRA: Low-Rank Adaptation for Llama on Chinese Instruction Dataset

<!-- **Authors:** -->

**_¹² [Zefeng Du](https://seeledu.github.io/index-en.html), ³ [Minghao Wu](https://minghao-wu.github.io/), ¹ <sup>*</sup> [Longyue Wang](http://www.longyuewang.com/)_**


<!-- **Affiliations:** -->

_¹ Tencent AI Lab,  ²University of Macau, ³ Monash University_,

_<sup>*</sup>Longyue Wang is the corresponding author: [vinnlywang@tencent.com](mailto:{vinnlywang@tencent.com)_
</div>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)]()

Chinese-Llama-2-LoRA is a project that focuses on applying [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) technique to fine-tune the Llama-2 on the Chinese instruction dataset [COIG](https://huggingface.co/datasets/BAAI/COIG) and sampled Chinese-English parallel documents. Llama-2 is a powerful language model developed by MetaAI, and with the help of LoRA, it can be further optimized for Chinese language understanding and generation tasks.

This repository contains the code and resources required to finetune Llama-2 using LoRA on a Chinese instruction dataset. By leveraging the unique characteristics of the LoRA technique, Chinese-Llama-2-LoRA enhances the capabilities of Llama-2 for handling Chinese natural language processing tasks.

## Key Features

- Fine-tune Llama-2 on Chinese instruction dataset using LoRA.
- Improve Chinese language understanding and generation capabilities of Llama.
- High-quality and state-of-the-art results on various Chinese NLP tasks.
- Easy-to-use codebase with clear instructions for usage.

## Model Checkpoints

The LoRA weights for [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) can be found at here. More model weights will be available soon.


## Datasets
We finetune the Llama-2 on the Chinese instruction dataset [COIG](https://huggingface.co/datasets/BAAI/COIG) and 20K Chinese-English parallel documents translation data. 


## Installation

To use Chinese-Llama-2-LoRA, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/longyuewangdcu/chinese-llama-2.git
   ```

2. Change into the project directory:

   ```bash
   cd Chinese-Llama-2-LoRA
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pretrained Llama model and the Chinese instruction dataset.

   > Note: The Llama model should be obtained separately from OpenAI's website, and the Chinese instruction dataset should be acquired based on the licensing and data rights.


## Usage

To finetune Llama-2 using LoRA on the Chinese instruction dataset, Run the command to start lora finetune.:

    ```
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

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. We appreciate your contributions to make Chinese-Llama-2-LoRA even better.


## Acknowledgments

Chinese-Llama-2-LoRA builds upon the Llama-2 developed by MetaAI. We would like to express our gratitude to the MetaAI team for their outstanding work and contributions to the field of natural language processing.
