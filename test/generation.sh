# Text generation
# path to the project
path=
# path to the model
model_path=
python3 $path/test/inference.py --model-name-or-path $model_path \
    -t 0.7 \
    -sa 'sample' \
    -i $path/test/test_case.txt \
    -o $path/test/test_case.general-task.txt\
    --length 4096