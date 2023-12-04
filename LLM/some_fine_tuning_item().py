# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/4/23 9:58
coding with comment！！！
"""

# 1. Freeze方法
from transformers import BertTokenizer, BertForPreTraining
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model=BertForPreTraining.from_pretrained('bert-base-chinese')
for name, param in model.named_parameters():###查看模型各个层数以及参数，并选择冻结层
    if not any(nd in name for nd in ["layers.27", "layers.26", "layers.25", "layers.24", "layers.23"]):
        param.requires_grad = False

# 2.基于PEFT的Lora微调方法(P-Tuning,P-Tuning-v,Prefix-Tuning,Prompt-Tuning,AdaLoRA)
from transformers import AutoModelForSe
q2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bert-base-uncased"
tokenizer_name_or_path = "bert-base-uncased"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=['**','**'] ###所处理的层id,可以通过遍历来寻找
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
