#!/bin/bash

python script.py \
    --baichuan_api_keys "sk-b9e63f78b784fa8027c5b0c0a9a39adf" \
    --knowledge_file_path_list [/path/to/file1.pdf, /path/to/file2.pdf] \
    --em_model_path "/path/to/em_model" \
    --fassi_save_path "/path/to/save_path" \
    --url_setting "127.0.0.1"

# 如果需要，停用Python虚拟环境
# deactivate
