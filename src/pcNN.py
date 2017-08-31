import os
import pdb
import sys

import pylab
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

class pcNN(nn.Module):
    def __init__(self):
        super(pcNN, self).__init__()
        self.epochs_trained = 0
        # Layers
        self.pool2x2 = nn.MaxPool2d(2,2)

    def forward(self, x):
        """Feed Forward"""
        return x


def get_data_loaders():
    """get train and test data """
    classes = ("Cancer", "Not-Cancer")
    trainLoader = None
    testLoader = None
    return trainLoader, testLoader, classes

def get_loss_optim(net, lr = 0.001, mom = 0. ):
    """Get loss function and backprop algo."""
    lossfunc = nnCrossEntropyLoss()
    optim = optim.SGD( net.parameters(), lr = lr, momentum = mom)
    return optim, lossfunc

def training(net, train_loader, test_loader, optim, lossfunc, bsize = 1, number_of_epochs = 1):
    """ Train a NN for n epochs"""

    print( '{0:5s} \t {1:5s} \t {2:11s} \t {3:11s}'.format( 'Epoch', 'Input', 'Loss Train' , 'Loss Test') )
    for epoch in range(number_of_epochs):
        running_loss = 0.0
        for j, train_batch in enumerate(train_loader, 0):
            inputs, labels = train_batch[0], train_batch[1]
            inputs, labels = Variable(inputs), Variable(labels)

            optim.zero_grad()

            result = net(inputs)
            loss = lossfunc(result, labels)

            loss.backward()
            optim.step()

            running_loss += loss.data[0]

            print('{0:5d} \t {1:5d} \t'.format( epoch + 1, i + 1), end = "\r", flush = True)

        test_loss = testing(net, test_loader, lossfunc, bs)

    #
    if epoch < number_of_epochs - 1:
        print('{0:5d} \t {1:5d} \t {2:4.6f} \t {3:4.6}'
              .format( epoch + 1, i + 1, running_loss / (i + 1) , test_loss) , end = '\r', flush = True)
        print('')
        running_loss = 0.0
        net.epochs_trained += 1

    return test_loss

def testing(net, test_loader, lossfunc, bs):
    """ Loop of test data and compute test loss"""
    test_loss = 0.0
    for j, test_batch in enumerate(test_loader):
        inputs, labels = test_batch[0], test_batch[1]
        inputs, labels = Variable(inputs), Variable(labels)
        result = net(inputs)
        loss = lossfunc(result, labels)

    return loss.data[0]

if __name__ == '__main__':
    batch_size = 5
    train, test, classes = get_data_loaders()
    nn = pcNN()
    optim, lossfunc = get_loss_optim()
    training(nn, train, test, optim, lossfunc, batch_size, 0)
