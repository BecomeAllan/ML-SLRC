# RevisaoSistematica

## Objective

The present repository is an implementation of First-order MAML with BERT for Systematic Literature Review. 
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

The 64 topic-agnostic labeled datasets proposed can be downloaded from the `PreTest_Meta_learning.ipynb` notebook.
Alternatively, download here (upload to gitHub). Make sure all data is in a `SLR_data` folder. 

All data treatment is done in the `PreTest_Meta_learning.ipynb` notebook.