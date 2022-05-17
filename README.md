# Automation of Systematic literature reviews classifications

## Objective

The present repository is an implementation of First-order MAML with SCIBERT for Systematic Literature Review Classification. 
The main objective is to implement few-shot classification by training our model with multiple cycles of limited datasets
sourced from high-resource domains. Therefore, the model will acheive a good initialization of weights in order to perform
few-shot classification from low-resource domains. Our model is a direct adaptation of `FONTE mailong25` paired with `fonte sciBERT`
for the purpose of Systematic Literature Review. 

## Notebook requirements (versions used)

* python (3.7.13)
* transformers (4.16.2)
* torchmetrics (0.8.0)
* matplotlib (3.5.1)

## Datasets

The 64 pre-processed topic-agnostic labeled datasets proposed can be downloaded from the `Meta_learning_EFL.ipynb` notebook and make sure all data is in a `SLR_data` folder. Alternatively, the originals datasets can be downloaded in [dropbox source](https://www.dropbox.com/sh/bs7eawof65l39ny/AAB_WucrCX04o-IAPjtYLMlva?dl=0). 
