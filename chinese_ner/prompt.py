# -*- coding: utf-8 -*-
"""
@author: some_model_test
@time: 2023/4/12 14:39
coding with comment！！！
"""
# step1:
import torch
import torch.nn as nn
from openprompt.data_utils import InputExample
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "Albert Einstein was one of the greatest intellects of his time.",
    ),
    InputExample(
        guid = 1,
        text_a = "The film was badly made.",
    ),
]
# step 2
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
#step3
from openprompt.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It was {"mask"}',
    tokenizer = tokenizer,
)
#step4
from openprompt.prompts import ManualVerbalizer
promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer = tokenizer,
)
#step 5
from openprompt import PromptForClassification

promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)
#step 6
from openprompt import PromptDataLoader

data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)
# step 7
# making zero-shot inference using pretrained MLM with prompt
# promptModel.eval()
# with torch.no_grad():




optimizer=torch.optim.Adam(promptModel.parameters(),lr=0.005)
loss_func=nn.CrossEntropyLoss()
for epoch in range(5):
    for batch in data_loader:

        logits = promptModel(batch)
        print(logits.size())
        print(logits)
        exit()
        # loss=loss_func(logits,torch.tensor([0,1]))
        # preds = torch.argmax(logits, dim=-1)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # print(loss)
# predictions would be 1, 0 for classes 'positive', 'negative'
