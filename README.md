# alpaca_finetune_sentiment_analysis

Alpaca-LoRA is an open-source project that reproduces results from Stanford Alpaca using Low-Rank Adaptation (LoRA) techniques. It provides an Instruct model of similar quality to text-davinci-003.
Alpaca-LoRA uses the resource-efficient low-rank adaptation (LoRA) method, also widely used in Stable Diffusion, with Meta’s LLaMA to achieve results comparable to Alpaca
Alpaca formula is open source, but may not be used commercially. However, the LLaMA model used for Alpaca is not released for commercial use, and the OpenAI GPT-3.5 terms of use prohibit using the model to develop AI models that compete with OpenAI. Stanford has therefore not yet released the model, only the training data and the code to generate the data and fine-tune the model.
-----------------

The labeled dataset I used to fine-tune the Alpaca model can be found at: 
https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?select=train.txt

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
