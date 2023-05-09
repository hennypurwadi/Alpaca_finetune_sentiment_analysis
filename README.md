# Alpaca Fine-tuned for Sentiment Analysis

Alpaca-LoRA is an open-source project that reproduces results from Stanford Alpaca using Low-Rank Adaptation (LoRA) techniques. It provides an Instruct model of similar quality to text-davinci-003.
Alpaca-LoRA uses the resource-efficient low-rank adaptation (LoRA) method, also widely used in Stable Diffusion, with Meta’s LLaMA to achieve results comparable to Alpaca
Alpaca formula is open source, but may not be used commercially. However, the LLaMA model used for Alpaca is not released for commercial use, and the OpenAI GPT-3.5 terms of use prohibit using the model to develop AI models that compete with OpenAI. Stanford has therefore not yet released the model, only the training data and the code to generate the data and fine-tune the model.

-----------------
This model is a fine-tuned version of the `Alpaca` model for sentiment analysis. 
It is trained on a dataset of texts with six different emotions: anger, fear, joy, love, sadness, and surprise.
The model was trained and tested on a labeled dataset from [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).

## Model Hub: https://huggingface.co/RinInori/alpaca_finetune_6_sentiments

![Image description](https://github.com/hennypurwadi/Bert_FineTune_Sentiment_Analysis/blob/main/images/imagesSaveModel_Tokenizer_To_HuggingFace_1.jpg?raw=true)
---

##Inference @HuggingFace: 

To create Space in HuggingFace: https://huggingface.co/new-space (Select for CPU Upgrade or above).

https://huggingface.co/spaces/RinInori/alpaca_finetune_6_sentiments

Upload app.py and requirements.txt to https://huggingface.co/spaces/RinInori/alpaca_finetune_6_sentiments/tree/main
