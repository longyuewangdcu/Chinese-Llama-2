# Chinese-Llama-2

<!-- **Authors:** -->

**_¹² <sup>&dagger;</sup>[Zefeng Du](https://seeledu.github.io/index-en.html), ³ <sup>&dagger;</sup>[Minghao Wu](https://minghao-wu.github.io/), ¹ <sup>*</sup> [Longyue Wang](http://www.longyuewang.com/)_**


<!-- **Affiliations:** -->

_¹ Tencent AI Lab,  ² University of Macau, ³ Monash University_,

_<sup>&dagger;</sup>equal contribution_

_<sup>*</sup>Longyue Wang is the corresponding author: [vinnlywang@tencent.com](mailto:{vinnlywang@tencent.com)_
</div>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)]()

## News
* [2023.07.20] :rocket: We fine-tune the Llama-2 on the Chinese instruction dataset using LoRA technique, known as Chinese-Llama-2-LoRA, and release the [Chinese-Llama-2-LoRA-7B](https://huggingface.co/seeledu/Chinese-Llama-2-LoRA-7B/).
* [2023.07.18] :tada::tada::tada: [Llama-2 is announced!](https://ai.meta.com/llama/)

## Overview

Chinese-Llama-2 is a project that aims to expand the impressive capabilities of the Llama-2 language model to the Chinese language. Developed by MetaAI, Llama-2 has already proven to be a powerful language model. In this project, we focus on three key areas of research:

1. **Parameter-efficient fine-tuning**: We employ the [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) technique to fine-tune Llama-2 specifically for the Chinese instruction dataset. This approach optimizes the model's performance while minimizing the number of required parameters.

2. **Full instruction fine-tuning**: We fine-tune all parameters of Llama-2 on the Chinese instruction dataset. By allowing the model to adapt fully to the characteristics of the Chinese language, we enhance its proficiency and accuracy in generating Chinese text.

3. **Continued pre-training**: To further enhance Llama-2's Chinese language understanding, we continue its pre-training on large-scale Chinese corpora. By exposing the model to vast amounts of Chinese text data, we enable it to capture intricate linguistic patterns and nuances, resulting in improved language generation.

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

The LoRA weights for [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) can be found at [here](https://huggingface.co/seeledu/Chinese-Llama-2-LoRA-7B/). More model weights will be available soon.

### Using the Model
You use our fine-tuned model as follows:

Prepare the llama2 model and lora model in your local path. You can download the lora model weight use this [link](https://huggingface.co/seeledu/Chinese-Llama-2-LoRA-7B/) or [百度网盘](https://pan.baidu.com/s/1SweESeylSd7eUkPsFucyyQ?pwd=zq4r). 

```bash 
path= #path to the project
model_path= #path to the origin model
lora_model_path== #path to the lora model
python3 $path/test/inference_lora.py --model-name-or-path $model_path \
    --lora-weights $lora_model_path\
    -t 0.7 \
    -sa 'sample' \
    -i $path/test/test_case.txt \
    -o $path/test/test_case.general-task.txt
```

### Showcases

```text
Question: 素数是什么？

Llama-2 7B Chat： I'm glad you're interested in learning about prime numbers! However, I must point out that the term "素数" is not a commonly used term in mathematics. In fact, the term "prime number" is the more commonly used term in English.

A prime number is a positive integer that is divisible only by itself and 1. For example, the first few prime numbers are 2, 3, 5, 7, 11, and 13.

If you have any other questions or if there's anything else I can help you with, feel free to ask!

Chinese-Llama-2-LoRA-7B: 素数是那些没有正整数因子的整数。

```

```text
Question: 说一下嫦娥的故事。

Llama-2 7B Chat: I'm glad you're interested in learning about prime numbers! However, I must point out that the term "素数" is not a commonly used term in mathematics. In fact, the term "prime number" is the more commonly used term in English.

A prime number is a positive integer that is divisible only by itself and 1. For example, the first few prime numbers are 2, 3, 5, 7, 11, and 13.

If you have any other questions or if there's anything else I can help you with, feel free to ask!

Chinese-Llama-2-LoRA-7B: 嫦娥的故事描绘了她的艰苦、勇敢和独特的故事。嫦娥是中国神话中的一个女神，被认为是月亮的化身。嫦娥的故事描述了她的艰苦童年，她被囚禁在一个石笋里，直到她独自一人奔走，寻找解救，并最终成功。她的故事象征着勇敢和坚韧，以及爱情的力量。
```
### Datasets
We finetune Llama-2 on the Chinese Alpaca instruction dataset, which consists of 51K examples.


### Fine-Tuning

To finetune Llama-2 using LoRA on the Chinese instruction dataset, Run the command to start LoRA finetune.:
```bash
# Multi-nodes are also supported

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

path= #path to the project
train_path=$path/train/run_clm_lora.py

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
