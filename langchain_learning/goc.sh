#!/bin/bash

# 执行Python脚本并传递参数
cd C:\Users\NewNLPer\Desktop\za\nlp_learning\nlp_learnning\langchain_learning

python langchain_project.py \
    --baichuan_api_keys "eeb680feddd64f2867b5f156a4f1d3e2.Gub5EM1kBosl6894" \
    --knowledge_file_path_list ["C:\Users\NewNLPer\Desktop\基于环境影响的合作演化模型.pdf"] \
    --em_model_path "D:\google下载" \
    --fassi_save_path "C:\Users\NewNLPer\Desktop\test\fassi_save" \
    --url_setting "127.0.0.1" \
    --pdf_combine_path "C:\Users\NewNLPer\Desktop\test\combine_file.pdf" \
    --n_gram 3



