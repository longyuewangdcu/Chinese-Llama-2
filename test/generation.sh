# Text generation
path= #path to the project
model_path= #path to the origin model
lora_model_path== #path to the lora model
python3 $path/test/inference_lora.py --model-name-or-path $model_path \
    --lora-weights $lora_model_path\
    -t 0.7 \
    -sa 'sample' \
    -i $path/test/test_case.txt \
    -o $path/test/test_case.general-task.txt