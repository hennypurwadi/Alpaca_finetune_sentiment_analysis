# alpaca_finetune_sentiment_analysis

The model is available on Hugging Face 
## Model Hub: https://huggingface.co/RinInori/alpaca_finetune_6_sentiments

![Image description](https://github.com/hennypurwadi/Bert_FineTune_Sentiment_Analysis/blob/main/images/SaveModel_Tokenizer_To_HuggingFace_1.jpg?raw=true)
---
language: en

license: MIT

datasets:
- custom

task_categories:
- text-classification

task_ids:
- sentiment-classification

---

```python

import os
import sys
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

!git clone https://github.com/tloen/alpaca-lora.git
%cd alpaca-lora
!git c!python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'RinInori/alpaca_finetune_6_sentiments' \
    --share_gradio    

!python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'RinInori/alpaca_finetune_6_sentiments' \
    --share_gradio

#Click on output(For example:)
Running on local URL:  http://

Running on public URL: https://
