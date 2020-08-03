"""
filename: utils
author: Geoffroy Palut

A collection of usefull functions 
"""

# TODO : Build a defense algo inspired by the Minority Report Defense: https://arxiv.org/abs/2004.13799

########################################################################
### Imports
########################################################################
# TODO : complete import section
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import time


########################################################################
### utility functions (plotting)
########################################################################

def plot_training(train_loss, valid_loss, values = "Loss"):
  """
  plot a comparison of performances on training and validation sets
  :param train_loss: performance values during training
  :param valid_loss: performance values on validation set
  :param values: string corresponding to what is displayed (loss or accuracy)
  :return: none, displays a plot of the input values
  """
  # Get the number of epochs
  epochs = range(len(train_loss))

  plt.title('Training vs Validation performance')
  plt.plot(epochs, train_loss, color='blue', label='Train')
  plt.plot(epochs, valid_loss, color='orange', label='Val')
  plt.xlabel('Epoch')
  plt.ylabel(values)
  plt.legend()


########################################################################
### Training iteration functions
########################################################################

# def training_epoch():
# TODO : code training/validation function
# Ideally with subfunction to run training epoch and validate



def adv_training_epoch(loader, model, optimizer, criterion, attack,
                       attack_params={'epsilon':8/255}, running_batches=2000):
  """
  Loop over the entire dataset provided by loader and performs adversarial
  training on the model.
  :param loader: pytorch dataloader that feeds the inputs used for training
  :param model: model to be trained
  :param optimizer:
  :param criterion:
  :param attack: attack function used to compute adversarial examples
  :param attack_params: set of additional parameters of the attack function
  (appart from model, natural input and label)
  :param running_batches: number of batches over which to average the losses
  :return: two lists containing the average natural and adversarial losses computed
  every running_batches mini-batch.
  """

  nat_loss_hist = []
  adv_loss_hist = []    
  running_nat_loss = 0.0
  running_adv_loss = 0.0
  for i, data in enumerate(loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      # sending them to the GPU at each step
      inputs, labels = data[0].to(device), data[1].to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # Compute adversarial examples 
      adv_inputs = attack(model, x_nat=inputs, y_nat=labels,**attack_params)
      
      # forward + backward + optimize
      adv_outputs = model(adv_inputs)
      adv_loss = criterion(adv_outputs, labels)
      adv_loss.backward()
      optimizer.step()

      # Compute natural loss
      with torch.no_grad():
        nat_outputs = model(inputs)
        nat_loss = criterion(nat_outputs, labels)

      # print statistics
      running_nat_loss += nat_loss.item()
      running_adv_loss += adv_loss.item()

      if i % running_batches == running_batches-1: # print loss regularly
        avg_nat_loss = running_nat_loss / running_batches
        avg_adv_loss = running_adv_loss / running_batches
  
        nat_loss_hist.append(avg_nat_loss)
        adv_loss_hist.append(avg_adv_loss)
        
        print('[%d, %5d] nat loss: %.3f ; adv loss: %.3f' %
              (epoch + 1, i + 1, avg_nat_loss, avg_adv_loss))
        running_nat_loss = 0
        running_adv_loss = 0

  return (nat_loss_hist, adv_loss_hist)