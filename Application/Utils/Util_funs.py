from ML_SLRC import *

import os
import numpy as np
import pandas as pd


from torch.utils.data import  DataLoader
from torch.optim import Adam

import gc
from torchmetrics import functional as fn

import random


warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import ipywidgets as widgets
from IPython.display import  display, clear_output
import matplotlib.pyplot as plt
import warnings
import torch

import time
from sklearn.manifold import TSNE
from copy import deepcopy
import seaborn as sns
import matplotlib.pylab as plt
import json
from pathlib import Path

import re
from collections import defaultdict

# SEED = 2222

# gen_seed = torch.Generator().manual_seed(SEED)






# Random seed function
def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

# Tasks for meta-learner
def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]


# Prepare data to process by Domain-learner
def prepare_data(data, batch_size, tokenizer,max_seq_length,
                 input = 'text', output = 'label',
                 train_size_per_class = 5, global_datasets = False,
                 treat_text_fun =None):
  data = data.reset_index().drop("index", axis=1)

  if global_datasets:
    global data_train, data_test

  # Sample task for training
  data_train = data.groupby('label').sample(train_size_per_class, replace=False)
  idex = data.index.isin(data_train.index)

  # The Test set to label by the model
  data_test = data[~idex].reset_index()


  # Transform in dataset to model
  ## Train
  dataset_train = SLR_DataSet(
    data = data_train.sample(frac=1),
    input = input,
    output = output,
    tokenizer=tokenizer,
    max_seq_length =max_seq_length,
    treat_text =treat_text_fun)

  ## Test
  dataset_test = SLR_DataSet(
    data = data_test,
    input = input,
    output = output,
    tokenizer=tokenizer,
    max_seq_length =max_seq_length,
    treat_text =treat_text_fun)
  
  # Dataloaders
  ## Train 
  data_train_loader = DataLoader(dataset_train,
                           shuffle=True,
                           batch_size=batch_size['train']
                                )
  
  ## Test
  if len(dataset_test) % batch_size['test'] == 1 :
    data_test_loader = DataLoader(dataset_test,
                                    batch_size=batch_size['test'],
                                    drop_last=True)
  else:
    data_test_loader = DataLoader(dataset_test,
                                    batch_size=batch_size['test'],
                                    drop_last=False)

  return data_train_loader, data_test_loader, data_train, data_test


# Meta trainer
def meta_train(data, model, device, Info,
               print_epoch =True,
                Test_resource =None,
                treat_text_fun =None):

  # Meta-learner model
  learner = Learner(model = model, device = device, **Info)
  
  # Testing tasks
  if isinstance(Test_resource, pd.DataFrame):
    test = MetaTask(Test_resource, num_task = 0, k_support=10, k_query=10,
                  training=False,treat_text =treat_text_fun, **Info)


  torch.clear_autocast_cache()
  gc.collect()
  torch.cuda.empty_cache()

  # Meta epoch (Outer epoch)
  for epoch in tqdm(range(Info['meta_epoch']), desc= "Meta epoch ", ncols=80):
      
      # Train tasks
      train = MetaTask(data,
                      num_task = Info['num_task_train'],
                      k_support=Info['k_qry'],
                      k_query=Info['k_spt'],
                      treat_text =treat_text_fun, **Info)

      # Batch of train tasks
      db = create_batch_of_tasks(train, is_shuffle = True, batch_size = Info["outer_batch_size"])

      if print_epoch:
      # Outer loop bach training
        for step, task_batch in enumerate(db):          
            print("\n-----------------Training Mode","Meta_epoch:", epoch ,"-----------------\n")
            
            # meta-feedfoward (outer-feedfoward)
            acc = learner(task_batch, valid_train= print_epoch)
            print('Step:', step, '\ttraining Acc:', acc)
        
        if isinstance(Test_resource, pd.DataFrame):
          # Validating Model
          if ((epoch+1) % 4) + step == 0:
              random_seed(123)
              print("\n-----------------Testing Mode-----------------\n")
              
              # Batch of test tasks
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
            # meta-feedfoward (outer-feedfoward)
            acc = learner(task_batch, print_epoch, valid_train= print_epoch)

  torch.clear_autocast_cache()
  gc.collect()
  torch.cuda.empty_cache()



def train_loop(data_train_loader, data_test_loader, model, device, epoch = 4, lr = 1, print_info = True, name = 'name'):
  # Start the model's parameters
  model_meta = deepcopy(model)
  optimizer = Adam(model_meta.parameters(), lr=lr)

  model_meta.to(device)
  model_meta.train()

  # Task epoch (Inner epoch)
  for i in range(0, epoch):
      all_loss = []

      # Inner training batch (support set)
      for inner_step, batch in enumerate(data_train_loader):
          batch = tuple(t.to(device) for t in batch)
          input_ids, attention_mask,q_token_type_ids, label_id = batch
          
          # Inner Feedfoward
          loss, _, _ = model_meta(input_ids, attention_mask,q_token_type_ids, labels = label_id.squeeze())
          
          # compute grads
          loss.backward()

          # update parameters
          optimizer.step()
          optimizer.zero_grad()
          
          all_loss.append(loss.item())
      

      if (i % 2 == 0) & print_info:
          print("Loss: ", np.mean(all_loss))


  # Test evaluation
  model_meta.eval()
  all_loss = []
  all_acc = []
  features = []
  labels = []
  predi_logit = []

  with torch.no_grad():
      # Test's Batch loop
      for inner_step, batch in enumerate(tqdm(data_test_loader,
                                              desc="Test validation | " + name,
                                              ncols=80)) :
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask,q_token_type_ids, label_id = batch

        # Predictions
        _, feature, prediction = model_meta(input_ids, attention_mask,q_token_type_ids, labels = label_id.squeeze())

        # Save batch's predictions 
        prediction = prediction.detach().cpu().squeeze()
        label_id = label_id.detach().cpu()
        labels.append(label_id.numpy().squeeze())
        
        logit = feature[1].detach().cpu()
        predi_logit.append(logit.numpy())

        feature_lat = feature[0].detach().cpu()
        features.append(feature_lat.numpy())

        # Accuracy over the test's bach
        acc = fn.accuracy(prediction, label_id).item()
        all_acc.append(acc)
      del input_ids, attention_mask, label_id, batch

  if print_info:
    print("acc:", np.mean(all_acc))

  model_meta.to('cpu')
  gc.collect()
  torch.cuda.empty_cache()

  del model_meta, optimizer

  return map_feature_tsne(features, labels, predi_logit)

# Process predictions and map the feature_map in tsne
def map_feature_tsne(features, labels, predi_logit):
  
  features = np.concatenate(np.array(features,dtype=object))
  features = torch.tensor(features.astype(np.float32)).detach().clone()
  
  labels = np.concatenate(np.array(labels,dtype=object))
  labels = torch.tensor(labels.astype(int)).detach().clone()

  logits = np.concatenate(np.array(predi_logit,dtype=object))
  logits = torch.tensor(logits.astype(np.float32)).detach().clone()

  # Dimention reduction
  X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(features.detach().clone())

  return logits.detach().clone(), X_embedded, labels.detach().clone(), features.detach().clone()
  
def wss_calc(logit, labels, trsh = 0.5):
  
  # Prediction label given the threshold
  predict_trash = torch.sigmoid(logit).squeeze() >= trsh
  
  # Compute confusion matrix values
  CM = confusion_matrix(labels, predict_trash.to(int) )
  tn, fp, fne, tp = CM.ravel()

  P = (tp + fne)  
  N = (tn + fp) 
  recall = tp/(tp+fne)

  # WSS
  wss = (tn + fne)/len(labels) -(1- recall)

  # AWSS
  awss = (tn/N - fne/P)

  return {
      "wss": round(wss,4),
      "awss": round(awss,4),
      "R": round(recall,4),
      "CM": CM
      }


# Compute the metrics
def plot(logits, X_embedded, labels, threshold, show = True,
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

  # Roc curve
  fpr, tpr, thresholds = roc_curve(labels, predict.squeeze())

  # Given by a Recall of 95% (threshold avaliation)
  ## WSS
  ### Index to recall
  idx_wss95 = sum(tpr < 0.95)
  ### threshold
  thresholds95 = thresholds[idx_wss95]

  ### Compute the metrics
  wss95_info = wss_calc(logits,labels, thresholds95 )
  acc_wss95 = fn.accuracy(predict, labels, threshold=thresholds95)
  f1_wss95 = fn.f1_score(predict, labels, threshold=thresholds95)


  # Given by a threshold (recall avaliation)
  ### Compute the metrics
  wss_info = wss_calc(logits,labels, threshold )
  acc_wssR = fn.accuracy(predict, labels, threshold=threshold)
  f1_wssR = fn.f1_score(predict, labels, threshold=threshold)


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
      # threshold 95
      "threshold@95": thresholds95
  }

  # Print stats
  if print_stats:
    wss95= f"WSS@95:{wss95_info['wss']}, R: {wss95_info['R']}"
    wss95_adj= f"ASSWSS@95:{wss95_info['awss']}"
    print(wss95)
    print(wss95_adj)
    print('Acc.:', round(acc_wss95.item(), 4))
    print('F1-score:', round(f1_wss95.item(), 4))
    print(f"threshold to wss95: {round(thresholds95, 4)}")
    cm = pd.DataFrame(wss95_info['CM'],
              index=index,
              columns=col)
    
    print("\nConfusion matrix:")
    print(cm)
    print("\n---Metrics with threshold:", threshold, "----\n")
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


  # Plots

  if make_plot:

    fig, axes = plt.subplots(1, 4, figsize=(25,10))
    alpha = torch.squeeze(predict).numpy()

    # TSNE
    p1 = sns.scatterplot(x=X_embedded[:, 0],
                  y=X_embedded[:, 1],
                  hue=labels,
                  alpha=alpha, ax = axes[0]).set_title('Predictions-TSNE', size=20)
    
    
    # WSS@95
    t_wss = predict >= thresholds95
    t_wss = t_wss.squeeze().numpy()
    p2 = sns.scatterplot(x=X_embedded[t_wss, 0],
                  y=X_embedded[t_wss, 1],
                  hue=labels[t_wss],
                  alpha=alpha[t_wss], ax = axes[1]).set_title('WSS@95', size=20)

    # WSS@R
    t = predict >= threshold
    t = t.squeeze().numpy()
    p3 = sns.scatterplot(x=X_embedded[t, 0],
                  y=X_embedded[t, 1],
                  hue=labels[t],
                  alpha=alpha[t], ax = axes[2]).set_title(f'Predictions-threshold {threshold}', size=20)

    # ROC-Curve
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
    # axes[3].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[3].legend(loc="lower right")
    axes[3].set_title(label= "ROC", size = 20)
    axes[3].set_ylabel("True Positive Rate", fontsize = 15)
    axes[3].set_xlabel("False Positive Rate", fontsize = 15)
    

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

# Interface to evaluation
class diagnosis():
  def __init__(self, names, Valid_resource, batch_size_test,
   model,Info, device,treat_text_fun=None,start = 0):
    self.names=names
    self.Valid_resource=Valid_resource
    self.batch_size_test=batch_size_test
    self.model=model
    self.start=start
    self.Info = Info
    self.device = device
    self.treat_text_fun = treat_text_fun
    

    # BOX INPUT
    self.value_trash = widgets.FloatText(
        value=0.95,
        description='threshold',
        disabled=False
    )
    self.valueb = widgets.IntText(
        value=10,
        description='size',
        disabled=False
    )

    # Buttons
    self.train_b = widgets.Button(description="Train")
    self.next_b = widgets.Button(description="Next")
    self.eval_b = widgets.Button(description="Evaluation")

    self.hbox = widgets.HBox([self.train_b, self.valueb])

    # Click buttons functions
    self.next_b.on_click(self.Next_button)
    self.train_b.on_click(self.Train_button)
    self.eval_b.on_click(self.Evaluation_button)


  # Next button
  def Next_button(self,p):
    clear_output()
    self.i=self.i+1

    # Select the domain data
    self.domain = self.names[self.i]
    self.data = self.Valid_resource[self.Valid_resource['domain'] == self.domain]
    
    print("Name:", self.domain)
    print(self.data['label'].value_counts())
    display(self.hbox)
    display(self.next_b)


  # Train button
  def Train_button(self, y):
    clear_output()
    print(self.domain)

    # Prepare data for training (domain-learner)
    self.data_train_loader, self.data_test_loader, self.data_train, self.data_test = prepare_data(self.data,
              train_size_per_class = self.valueb.value,
              batch_size = {'train': self.Info['inner_batch_size'],
                            'test': self.batch_size_test},
              max_seq_length = self.Info['max_seq_length'],
              tokenizer = self.Info['tokenizer'],
              input = "text",
              output = "label",
              treat_text_fun=self.treat_text_fun)

    # Train the model and predict in the test set
    self.logits, self.X_embedded, self.labels, self.features = train_loop(self.data_train_loader, self.data_test_loader,
                                                        self.model, self.device,
                                                        epoch = self.Info['inner_update_step'],
                                                        lr=self.Info['inner_update_lr'],
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
    print(data_train['label'].value_counts())
    print("-------Test data-------")
    print(data_test['label'].value_counts())
    # print("\n")
    
    display(self.next_b)
    display(tresh_box)
    display(self.hbox)

    # Compute metrics    
    metrics = plot(self.logits, self.X_embedded, self.labels,
                    threshold=self.Info['threshold'], show = True,
                    namefig= 'test',
                  make_plot = True,
                  print_stats = True,
                  save=False)

  def __call__(self):
    self.i= self.start-1
    clear_output()
    display(self.next_b)




# Simulation attemps of domain learner
def pipeline_simulation(Valid_resource, names_to_valid, path_save,
                        model, Info, device, initializer_model,
                        treat_text_fun=None):
  n_attempt  = 5
  batch_test = 100

  # Create a directory to save informations
  for name in names_to_valid:
    name = re.sub("\.csv", "",name)
    Path(path_save  + name + "/img").mkdir(parents=True, exist_ok=True)

  # Dict to sabe roc curves
  roc_stats = defaultdict(lambda: defaultdict(
      lambda: defaultdict(
          list
          )
      )
  )


  

  all_metrics = []
  # Loop over a list of domains
  for name in names_to_valid:
    
    # Select a domain dataset
    data = Valid_resource[Valid_resource['domain'] == name].reset_index().drop("index", axis=1)

    # Attempts simulation
    for attempt in range(n_attempt):
      print("---"*4,"attempt", attempt, "---"*4)
      
      # Prepare data to pass to the model
      data_train_loader, data_test_loader,  _ , _ = prepare_data(data,
                train_size_per_class = Info['k_spt'],
                batch_size = {'train': Info['inner_batch_size'],
                              'test': batch_test},
                max_seq_length = Info['max_seq_length'],
                tokenizer = Info['tokenizer'],
                input = "text",
                output = "label",
                treat_text_fun=treat_text_fun)

      # Train the model and evaluate on the test set of the domain
      logits, X_embedded, labels, features = train_loop(data_train_loader, data_test_loader,
                                                        model, device,
                                                        epoch = Info['inner_update_step'],
                                                        lr=Info['inner_update_lr'],
                                                        print_info=False,
                                                        name = name)
      
      
      name_domain = re.sub("\.csv", "",name)

      # Compute the metrics
      metrics = plot(logits, X_embedded, labels,
                    threshold=Info['threshold'], show = False,
                    namefig= path_save  + name_domain + "/img/" + str(attempt) + 'plots',
        make_plot = True, print_stats = False, save =  True)

      # Compute the roc-curve
      fpr, tpr, _ = roc_curve(labels, torch.sigmoid(logits).squeeze())
      
      # Save the correspoud information of the domain
      metrics['name'] = name_domain
      metrics['layer_size'] = Info['bert_layers']
      metrics['attempt'] = attempt
      roc_stats[name_domain][str(Info['bert_layers'])]['fpr'].append(fpr.tolist())
      roc_stats[name_domain][str(Info['bert_layers'])]['tpr'].append(tpr.tolist())
      all_metrics.append(metrics)

      # Save the metrics and the roc curve  of the attemp
      pd.DataFrame(all_metrics).to_csv(path_save+ "metrics.csv")
      roc_path =  path_save + "roc_stats.json"
      with open(roc_path, 'w') as fp:
          json.dump(roc_stats, fp)


      del fpr, tpr, logits, X_embedded, labels
      del features, metrics,  _


  # Save the information used to evaluate the validation resource
  save_info = Info.copy()
  save_info['model'] = initializer_model.tokenizer.name_or_path
  save_info.pop("tokenizer")
  save_info.pop("bert_layers")

  info_path =  path_save+"info.json"
  with open(info_path, 'w') as fp:
      json.dump(save_info, fp)


# Loading dataset statistics
def load_data_statistics(paths, names):
  size = []
  pos = []
  neg = []
  for p in paths:
    data = pd.read_csv(p) 
    data = data.dropna()
    # Dataset size
    size.append(len(data))
    # Number of positive labels
    pos.append(data['labels'].value_counts()[1])
    # Number of negative labels
    neg.append(data['labels'].value_counts()[0])
  del data

  info_load = pd.DataFrame({
      "size":size,
      "pos":pos,
      "neg":neg,
      "names":names,
      "paths": paths })
  return info_load

# Loading the datasets
def load_data(train_info_load):

  col = ['abstract','title', 'labels', 'domain']

  data_train = pd.DataFrame(columns=col)
  for p in train_info_load['paths']:  
    data_temp = pd.read_csv(p).loc[:, ['labels', 'title', 'abstract']]
    data_temp = pd.read_csv(p).loc[:, ['labels', 'title', 'abstract']]
    data_temp['domain'] = os.path.basename(p)
    data_train = pd.concat([data_train, data_temp])
    
  data_train['text'] = data_train['title'] + data_train['abstract'].replace(np.nan, '')

  return( data_train \
            .replace({"labels":{0:"negative", 1:'positive'}})\
            .rename({"labels":"label"} , axis=1)\
            .loc[ :,("text","domain","label")]
        )


 
