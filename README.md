# Automation of Systematic literature reviews classifications

## Objective

The present repository is an implementation of First-order Model-Agnostic Meta-Learning (MAML) with SciBERT for Systematic Literature Review Classification (Finn et al., 2017; Nichol et al., 2018; Beltagy et al., 2019).

The main objective is to implement few-shot classification by training our learner with multiple cycles of few examples sourced by domains datasets using just the Title concatenated with the abstract of a research paper. Therefore, the model will acheive a good initialization of weights in order to perform a classification/ranking on the unlabeled data training the model with few labaled domain's examples. Our model is a direct adaptation of Wang work (Wang et al.,2021) paired with some layers of SciBERT, for the purpose of Systematic Literature Review (SLR) automation. 

To use the model, the folder [Application](https://github.com/BecomeAllan/ML-SLRC/tree/main/Application) has a notebook example that can be used to conduct a Systematic Literature Review classification using the Semantic Scholar as a tool for retrive scientific literature data or can be direct downloaded from [Hugging Face](https://huggingface.co/becomeallan/ML-SLRC).

## Notebook requirements (versions used)

* python (3.7.13)
* transformers (4.16.2)
* torchmetrics (0.8.0)
* matplotlib (3.5.1)

## Datasets

The 64 pre-processed topics labeled datasets proposed can be downloaded from the `Meta_learning_EFL.ipynb` notebook and make sure all data is in a `SLR_data` folder. Alternatively, the originals datasets can be downloaded in [dropbox source](https://www.dropbox.com/sh/bs7eawof65l39ny/AAB_WucrCX04o-IAPjtYLMlva?dl=0). 

## References


Beltagy, I., Cohan, A., and Lo, K. (2019). SciBERT:
Pretrained contextualized embeddings for scientific
text. CoRR, abs/1903.10676.

Finn, C., Abbeel, P., and Levine, S. (2017). Model-agnostic
meta-learning for fast adaptation of deep networks.
CoRR, abs/1703.03400.

Nichol, A., Achiam, J., and Schulman, J. (2018).
On first-order meta-learning algorithms. CoRR,
abs/1803.02999.

Wang, S., Fang, H., Khabsa, M., Mao, H., and Ma, H.
(2021). Entailment as few-shot learner. CoRR,
abs/2104.14690.




