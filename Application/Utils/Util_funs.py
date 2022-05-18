import os
import torch
import numpy as np
import random
import json, pickle
# from ML_SLRC import SLR_DataSet, SLR_Classifier

import torch.nn.functional as F
import torch.nn as nn
import math
import torch
import numpy as np
import pandas as pd
import time
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE
from copy import deepcopy, copy
import seaborn as sns
import matplotlib.pylab as plt
from pprint import pprint
import shutil
import datetime
import re
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification
from copy import deepcopy
import gc
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import torchmetrics
from torchmetrics import functional as fn


SEED = 2222

gen_seed = torch.Generator().manual_seed(SEED)


# Random seed function
def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

# Batch creation function
def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]



def prepare_data(data, batch_size,tokenizer,max_seq_length,
                 input = 'text', output = 'label',
                 train_size_per_class = 5):
  data = data.reset_index().drop("index", axis=1)

  labaled_data = data.loc[~data['label'].isna()]

  data_train = labaled_data.groupby('label').sample(train_size_per_class)

  rest_labaled_data = labaled_data.loc[~labaled_data.index.isin(data_train.index),:]
  unlabaled_data = data.loc[data['label'].isna()]

  data_test = pd.concat([rest_labaled_data, unlabaled_data])


  # Train
  ## Transforma em dataset
  dataset_train = SLR_DataSet(
    data = data_train.sample(frac=1),
    input = input,
    output = output,
    tokenizer=tokenizer,
    max_seq_length =max_seq_length)

  # Test
  # Dataloaders
    ## Transforma em dataset
  dataset_test = SLR_DataSet(
    data = data_test,
    input = input,
    output = output,
    tokenizer=tokenizer,
    max_seq_length =max_seq_length)
  
  # Dataloaders
  ## Treino 
  data_train_loader = DataLoader(dataset_train,
                           shuffle=True,
                           batch_size=batch_size['train']
                                )
  
  if len(dataset_test) % batch_size['test'] == 1 :
    data_test_loader = DataLoader(dataset_test,
                                    batch_size=batch_size['test'],
                                    drop_last=True)
  else:
    data_test_loader = DataLoader(dataset_test,
                                    batch_size=batch_size['test'],
                                    drop_last=False)

  return data_train_loader, data_test_loader, data_train, data_test





from tqdm import tqdm

def meta_train(data, model, device, Info, print_epoch =True, size_layer=0, Test_resource =None):

  learner = Learner(model = model, device = device, **Info)
  
  # Testing tasks
  if isinstance(Test_resource, pd.DataFrame):
    test = MetaTask(Test_resource, num_task = 0, k_support=10, k_query=10,
                  training=False, **Info)


  torch.clear_autocast_cache()
  gc.collect()
  torch.cuda.empty_cache()

  # Meta epoca
  for epoch in tqdm(range(Info['meta_epoch']), desc= "Meta epoch ", ncols=80):
    # print("Meta Epoca:", epoch)
      
      # Tarefas de treino
      train = MetaTask(data,
                      num_task = Info['num_task_train'],
                      k_support=Info['k_qry'],
                      k_query=Info['k_spt'], **Info)

      # Batchs de tarefas    
      db = create_batch_of_tasks(train, is_shuffle = True, batch_size = Info["outer_batch_size"])

      if print_epoch:
      # Outer loop bach training
        for step, task_batch in enumerate(db):          
            print("\n-----------------Training Mode","Meta_epoch:", epoch ,"-----------------\n")
            # meta-feedfoward
            acc = learner(task_batch, valid_train= print_epoch)
            print('Step:', step, '\ttraining Acc:', acc)
        if isinstance(Test_resource, pd.DataFrame):
          # Validating Model 
          if ((epoch+1) % 4) + step == 0:
              random_seed(123)
              print("\n-----------------Testing Mode-----------------\n")
              db_test = create_batch_of_tasks(test, is_shuffle = False, batch_size = 1)
              acc_all_test = []

              # Looping testing tasks
              for test_batch in db_test:
                  acc = learner(test_batch, training = False)
                  acc_all_test.append(acc)

              print('Test acc:', np.mean(acc_all_test))
              del acc_all_test, db_test

              # Restarting training randomly
              random_seed(int(time.time() % 10))
          
        
      else:
        for step, task_batch in enumerate(db):
            acc = learner(task_batch, print_epoch, valid_train= print_epoch)

  torch.clear_autocast_cache()
  gc.collect()
  torch.cuda.empty_cache()



def train_loop(data_train_loader, data_test_loader, model, device, epoch = 4, lr = 1, print_info = True, name = 'name'):
  # Inicia o modelo
  model_meta = deepcopy(model)
  optimizer = Adam(model_meta.parameters(), lr=lr)

  model_meta.to(device)
  model_meta.train()

  # Loop de treino da tarefa
  for i in range(0, epoch):
      all_loss = []

      # Inner training batch (support set)
      for inner_step, batch in enumerate(data_train_loader):
          batch = tuple(t.to(device) for t in batch)
          input_ids, attention_mask,q_token_type_ids, label_id = batch
          
          # Feedfoward
          loss, _, _ = model_meta(input_ids, attention_mask,q_token_type_ids, labels = label_id.squeeze())
          
          # Calcula gradientes
          loss.backward()

          # Atualiza os parametros
          optimizer.step()
          optimizer.zero_grad()
          
          all_loss.append(loss.item())
      

      if (i % 2 == 0) & print_info:
          print("Loss: ", np.mean(all_loss))


  # Predicao no banco de teste
  model_meta.eval()
  all_loss = []
  # all_acc = []
  features = []
  labels = []
  predi_logit = []

  with torch.no_grad():
      for inner_step, batch in enumerate(tqdm(data_test_loader,
                                              desc="Test validation | " + name,
                                              ncols=80)) :
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask,q_token_type_ids, label_id = batch

        # Predicoes
        _, feature, prediction = model_meta(input_ids, attention_mask,q_token_type_ids, labels = label_id.squeeze())

        prediction = prediction.detach().cpu().squeeze()
        label_id = label_id.detach().cpu()
        logit = feature[1].detach().cpu()
        feature_lat = feature[0].detach().cpu()

        labels.append(label_id.numpy().squeeze())
        features.append(feature_lat.numpy())
        predi_logit.append(logit.numpy())

        # acc = fn.accuracy(prediction, label_id).item()
        # all_acc.append(acc)
      del input_ids, attention_mask, label_id, batch

  # if print_info:
  #   print("acc:", np.mean(all_acc))

  model_meta.to('cpu')
  gc.collect()
  torch.cuda.empty_cache()

  del model_meta, optimizer


  features = np.concatenate(np.array(features,dtype=object))
  labels = np.concatenate(np.array(labels,dtype=object))
  logits = np.concatenate(np.array(predi_logit,dtype=object))

  features = torch.tensor(features.astype(np.float32)).detach().clone()
  labels = torch.tensor(labels.astype(int)).detach().clone()
  logits = torch.tensor(logits.astype(np.float32)).detach().clone()

  # Reducao de dimensionalidade
  X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(features.detach().clone())

  return logits.detach().clone(), X_embedded, labels.detach().clone(), features.detach().clone()


def wss_calc(logit, labels, trsh = 0.5):
  
  # Predicao com base nos treshould
  predict_trash = torch.sigmoid(logit).squeeze() >= trsh
  CM = confusion_matrix(labels, predict_trash.to(int) )
  tn, fp, fne, tp = CM.ravel()

  P = (tp + fne)  
  N = (tn + fp) 
  recall = tp/(tp+fne)

  # Wss antigo
  wss_old = (tn + fne)/len(labels) -(1- recall)

  # WSS novo
  wss_new = (tn/N - fne/P)

  return {
      "wss": round(wss_old,4),
      "awss": round(wss_new,4),
      "R": round(recall,4),
      "CM": CM
      }




from sklearn.metrics import confusion_matrix
from torchmetrics import functional as fn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import ipywidgets as widgets
from IPython.display import HTML, display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def plot(logits, X_embedded, labels, tresh, show = True,
         namefig = "plot", make_plot = True, print_stats = True, save = True):
  col = pd.MultiIndex.from_tuples([
                                   ("Predict", "0"),
                                   ("Predict", "1")
                                   ])
  index = pd.MultiIndex.from_tuples([
                                   ("Real", "0"),
                                   ("Real", "1")
                                   ])

  predict = torch.sigmoid(logits).detach().clone()

  roc_auc = dict()

  fpr, tpr, thresholds = roc_curve(labels, predict.squeeze())

  # Sem especificar o tresh
  # WSS
  ## indice do recall 0.95
  idx_wss95 = sum(tpr < 0.95)
  thresholds95 = thresholds[idx_wss95]

  wss95_info = wss_calc(logits,labels, thresholds95 )
  acc_wss95 = fn.accuracy(predict, labels, threshold=thresholds95)
  f1_wss95 = fn.f1_score(predict, labels, threshold=thresholds95)


  # Especificando o tresh
  # Treshold avaliation


  ## WSS
  wss_info = wss_calc(logits,labels, tresh )
  # Accuraci
  acc_wssR = fn.accuracy(predict, labels, threshold=tresh)
  f1_wssR = fn.f1_score(predict, labels, threshold=tresh)


  metrics= {
      # WSS
      "WSS@95": wss95_info['wss'],
      "AWSS@95": wss95_info['awss'],
      "WSS@R": wss_info['wss'],
      "AWSS@R": wss_info['awss'],
      # Recall
      "Recall_WSS@95": wss95_info['R'],
      "Recall_WSS@R": wss_info['R'],
      # acc
      "acc@95": acc_wss95.item(),
      "acc@R": acc_wssR.item(),
      # f1
      "f1@95": f1_wss95.item(),
      "f1@R": f1_wssR.item(),
      # treshould 95
      "treshould@95": thresholds95
  }

  # print stats

  if print_stats:
    wss95= f"WSS@95:{wss95_info['wss']}, R: {wss95_info['R']}"
    wss95_adj= f"ASSWSS@95:{wss95_info['awss']}"
    print(wss95)
    print(wss95_adj)
    print('Acc.:', round(acc_wss95.item(), 4))
    print('F1-score:', round(f1_wss95.item(), 4))
    print(f"Treshold to wss95: {round(thresholds95, 4)}")
    cm = pd.DataFrame(wss95_info['CM'],
              index=index,
              columns=col)
    
    print("\nConfusion matrix:")
    print(cm)
    print("\n---Metrics with threshold:", tresh, "----\n")
    wss= f"WSS@R:{wss_info['wss']}, R: {wss_info['R']}"
    print(wss)
    wss_adj= f"AWSS@R:{wss_info['awss']}"
    print(wss_adj)
    print('Acc.:', round(acc_wssR.item(), 4))
    print('F1-score:', round(f1_wssR.item(), 4))
    cm = pd.DataFrame(wss_info['CM'],
                index=index,
                columns=col)
      
    print("\nConfusion matrix:")
    print(cm)


  # Graficos

  if make_plot:

    fig, axes = plt.subplots(1, 4, figsize=(25,10))
    alpha = torch.squeeze(predict).numpy()

    # plots

    p1 = sns.scatterplot(x=X_embedded[:, 0],
                  y=X_embedded[:, 1],
                  hue=labels,
                  alpha=alpha, ax = axes[0]).set_title('Predictions-TSNE')
    
    t_wss = predict >= thresholds95
    t_wss = t_wss.squeeze().numpy()

    p2 = sns.scatterplot(x=X_embedded[t_wss, 0],
                  y=X_embedded[t_wss, 1],
                  hue=labels[t_wss],
                  alpha=alpha[t_wss], ax = axes[1]).set_title('WSS@95')

    t = predict >= tresh
    t = t.squeeze().numpy()

    p3 = sns.scatterplot(x=X_embedded[t, 0],
                  y=X_embedded[t, 1],
                  hue=labels[t],
                  alpha=alpha[t], ax = axes[2]).set_title(f'Predictions-Treshold {tresh}')


    roc_auc = auc(fpr, tpr)
    lw = 2

    axes[3].plot(
      fpr,
      tpr,
      color="darkorange",
      lw=lw,
      label="ROC curve (area = %0.2f)" % roc_auc)
    
    axes[3].plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    axes[3].axhline(y=0.95, color='r', linestyle='-')
    axes[3].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title= "ROC")
    axes[3].legend(loc="lower right")

    if show:
      plt.show()
    
    if save:
      fig.savefig(namefig, dpi=fig.dpi)

  return metrics

def auc_plot(logits,labels, color = "darkorange", label = "test"):
    predict = torch.sigmoid(logits).detach().clone()
    fpr, tpr, thresholds = roc_curve(labels, predict.squeeze())
    roc_auc = auc(fpr, tpr)
    lw = 2

    label = label + str(round(roc_auc,2))
    # print(label)

    plt.plot(
      fpr,
      tpr,
      color=color,
      lw=lw,
      label= label 
      )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.axhline(y=0.95, color='r', linestyle='-')


from sklearn.metrics import confusion_matrix
from torchmetrics import functional as fn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import ipywidgets as widgets
from IPython.display import HTML, display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class diagnosis():
  def __init__(self, names, Valid_resource, batch_size_test, model,Info,start = 0):
    self.names=names
    self.Valid_resource=Valid_resource
    self.batch_size_test=batch_size_test
    self.model=model
    self.start=start 

    self.value_trash = widgets.FloatText(
        value=0.95,
        description='tresh',
        disabled=False
    )

    self.valueb = widgets.IntText(
        value=10,
        description='size',
        disabled=False
    )

    self.train_b = widgets.Button(description="Train")
    self.next_b = widgets.Button(description="Next")
    self.eval_b = widgets.Button(description="Evaluation")

    self.hbox = widgets.HBox([self.train_b, self.valueb])

    self.next_b.on_click(self.Next_button)
    self.train_b.on_click(self.Train_button)
    self.eval_b.on_click(self.Evaluation_button)


  # Next button
  def Next_button(self,p):
    clear_output()
    self.i=self.i+1

    # global domain
    self.domain = names[self.i]
    print("Name:", self.domain)

    # global data
    self.data = self.Valid_resource[self.Valid_resource['domain'] == self.domain]
    print(self.data['label'].value_counts())

    display(self.hbox)
    display(self.next_b)

  # Train button
  def Train_button(self, y):
    clear_output()
    print(self.domain)

    # Preparing data for training
    self.data_train_loader, self.data_test_loader, self.data_train, self.data_test = prepare_data(self.data,
              train_size_per_class = self.valueb.value,
              batch_size = {'train': Info['inner_batch_size'],
                            'test': batch_size_test},
              max_seq_length = Info['max_seq_length'],
              tokenizer = Info['tokenizer'],
              input = "text",
              output = "label")

    self.logits, self.X_embedded, self.labels, self.features = train_loop(self.data_train_loader, self.data_test_loader,
                                                        model, device,
                                                        epoch = Info['inner_update_step'],
                                                        lr=Info['inner_update_lr'],
                                                        print_info=True,
                                                        name = self.domain)

    tresh_box = widgets.HBox([self.eval_b, self.value_trash])
    display(self.hbox)
    display(tresh_box)
    display(self.next_b)

  # Evaluation button
  def Evaluation_button(self, te):
    clear_output()
    tresh_box = widgets.HBox([self.eval_b, self.value_trash])

    print(self.domain)
    # print("\n")
    print("-------Train data-------")
    print(self.data_train['label'].value_counts())
    print("-------Test data-------")
    print(self.data_test['label'].value_counts())
    # print("\n")
    
    display(self.next_b)
    display(tresh_box)
    display(self.hbox)

    
    metrics = plot(self.logits, self.X_embedded, self.labels,
                    tresh=Info['tresh'], show = True,
                    # namefig= "./"+base_path +"/"+"Results/size_layer/"+ name_domain+'/' +str(n_layers) + '/img/' + str(attempt) + 'plots',
                    namefig= 'test',
                  make_plot = True,
                  print_stats = True,
                  save=False)

  def __call__(self):
    self.i= self.start-1

    clear_output()
    display(self.next_b)