#!/bin/bash

# 执行Python脚本并传递参数
cd C:\Users\NewNLPer\Desktop\za\nlp_learning\nlp_learnning\langchain_learning

python langchain_project.py \
    --baichuan_api_keys "eeb680feddd64f2867b5f156a4f1d3e2.Gub5EM1kBosl6894" \
    --knowledge_file_path_list /home/ubuntu/jijiezhou/data_save/pdf_data/compute_net.pdf \
    --em_model_path "/home/ubuntu/em_model" \
    --fassi_save_path "/home/ubuntu/jijiezhou/data_save/fassi_save_path" \
    --url_setting "192.168.32.23" \
    --pdf_combine_path "/home/ubuntu/jijiezhou/data_save/pdf_data/combine_file.pdf" \
    --n_gram 3



