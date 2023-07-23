# Text generation

path= #path to the project
model_path= #path to the  model
python3 $path/test/inference.py --model-name-or-path $model_path \
    -t 0.7 \
    -sa 'sample' \
    -i $path/test/test_case.txt \
    -o $path/test/test_case.general-task.txt\
    --length 4096