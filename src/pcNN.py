import os
import pdb
import sys
import json
import uuid
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

PATH = None

class pcNN(nn.Module):
    def __init__(self):
        super(pcNN, self).__init__()
        self.epochs_trained = 0
        self.set_layers()


    def set_layers(self):
        # Convolutions
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Poolings
        self.pool  = nn.MaxPool2d(2, 2)

        # Linear
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)


    def set_info(self, optimzer_name, lossfunc_name):
        self.optimizer = optimzer_name
        self.lossfunc = lossfunc_name
        self.train_loss = np.inf
        self.test_loss = np.inf


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



def get_data_loaders():
    """get train and test data """
    classes = ("Cancer", "Not-Cancer")
    train_loader = None
    test_loader = None
    return train_loader, test_loader, classes


def training(net, train_loader, test_loader, optimizer, lossfunc, bsize = 1, number_of_epochs = 1):
    """ Train a NN for n epochs"""

    if number_of_epochs == 0:
        return np.inf, np.inf

    print( '{0:5s} \t {1:5s} \t {2:11s} \t {3:11s}'.format( 'Epoch', 'Input', 'Loss Train' , 'Loss Test') )
    for epoch in range(number_of_epochs):
        running_loss = 0.0
        for j, train_batch in enumerate(train_loader, 0):
            inputs, labels = train_batch[0], train_batch[1]
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            result = net(inputs)
            loss = lossfunc(result, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            print('{0:5d} \t {1:5d} \t'.format( epoch + 1, j + 1), end = "\r", flush = True)

        test_loss = testing(net, test_loader, lossfunc, bs)
        net.epochs_trained += 1

    if epoch < number_of_epochs - 1:
        train_loss = running_loss / (j + 1)

        print('{0:5d} \t {1:5d} \t {2:4.6f} \t {3:4.6}'
              .format( epoch + 1, j + 1, train_loss , test_loss) , end = '\r', flush = True)
        print('')

    net.train_loss = train_loss
    net.test_loss = test_loss
    return


def testing(net, test_loader, lossfunc, bs):
    """ Loop of test data and compute test loss"""
    test_loss = 0.0
    for j, test_batch in enumerate(test_loader):
        inputs, labels = test_batch[0], test_batch[1]
        inputs, labels = Variable(inputs), Variable(labels)
        result = net(inputs)
        test_loss = lossfunc(result, labels)

    return test_loss.data[0]


def save_net_info(net, optimizer, lossfunc):
    path = os.path.realpath(PATH+"/../../")
    with open(path + "/top5_info.json",'r') as jsoninfo:
        top_info = json.load(jsoninfo)

    compare = [net.test_loss < top_loss for top_loss in top_info['top_5_test_loss']]

    if any(compare):
        net_name = str(uuid.uuid4()).split("-")[-1]
        rank = 6 - sum(compare)

        print("Neural Net ranked {0:d}".format(rank))
        print("Saving net, optimizer, loss function and information")

        net_results = {
            "net_name" : net_name ,
            "rank" : rank,
            "net_parameters" : str(net),
            "train_loss" : net.train_loss ,
            "test_loss" : net.test_loss,
            "optimizer_name" : net.optimizer,
            "optimizer_info" : dict(optimizer.state_dict()),
            "lossfunc_name" : net.lossfunc,
            "lossfunc_info" : dict(lossfunc.state_dict())
        }

        top_info['top_5_test_loss'].insert(rank - 1, net.test_loss)
        top_info['top_5_train_loss'].insert(rank - 1, net.train_loss)
        top_info['info'].insert(rank - 1, net_results)

        top_info['top_5_test_loss'].pop(-1)
        top_info['top_5_train_loss'].pop(-1)
        old = top_info['info'].pop(-1)

        # Create dir move old
        os.mkdir( path + "/nets/" + net_name)
        movecall = "mv %s %s" % (path + "/nets/" + old['net_name'],
                             path + "/nets/old" )
        os.system(movecall)
        #os.rmdir( path + "/nets/" + old['net_name'])

        filename = path + "/nets/" + net_name + "/pcNN.pk"
        with open(filename, 'wb') as NNBinary:
            pickle.dump(net, NNBinary)

        filename = path + "/nets/" + net_name + "/optimizer.pk"
        with open(filename, 'wb') as optimizerBinary:
            pickle.dump(optimizer, optimizerBinary)

        filename = path + "/nets/" + net_name + "/lossfunc.pk"
        with open(filename, 'wb') as lossfuncBinary:
            pickle.dump(lossfunc, lossfuncBinary)

        filename = path + "/nets/" + net_name + "/info.json"
        with open(filename,'w') as jsoninfo:
            json.dump(net_results, jsoninfo, indent=2)

        # update top five info
        filename = path + "/top5_info.json"
        with open(filename, 'w') as jsoninfo:
            json.dump(top_info, jsoninfo, indent=2)

        print("Files saved in folder {0:s}".format(net_name))

    else:
        print("Neural Net did not rank in top 5")

    return


def load_net(dirname):
    """ Load pretrained Neural Net From Binary file"""
    path = os.path.realpath(PATH+"/../../nets/"+dirname)
    with open(path + "/pcNN.pk", 'rb') as NNBinary:
        net = pickle.load(NNBinary)
    with open(path + "/optimizer.pk", 'rb') as optimizerBinary:
        optimizer = pickle.load(optimizerBinary)
    with open(path + "/lossfunc.pk", 'rb') as lossfuncBinary:
        lossfunc = pickle.load(lossfuncBinary)

    return net, optimizer, lossfunc


def load_ranked_n(n = 1):
    """ Loads the neural network ranked n"""
    if n < 1 or n > 5:
        print("Only storing top 5")
        return None, None, None

    path = os.path.realpath(PATH+"/../../")
    with open(path + "/top5_info.json",'r') as jsoninfo:
        top_info = json.load(jsoninfo)

    dirname = top_info['info'][n - 1]["net_name"]
    net, optimizer, lossfunc = load_net(dirname)

    return net, optimizer, lossfunc


if __name__ == '__main__':
    PATH = os.path.realpath(sys.argv[0])
    batch_size = 5
    train_loader, test_loader, classes = get_data_loaders()
    net = pcNN()
    lr = 0.001
    mom = 0.9
    lossfunc = nn.CrossEntropyLoss()
    optimizer = optim.SGD( net.parameters(), lr = lr, momentum = mom)
    net.set_info("CrossEntropyLoss", "SGD")
    net.train_loss, net.test_loss = 1.1, 0.1
    save_net_info(net, optimizer, lossfunc)
    net1, optimizer1, lossfunc1 = load_ranked_n(1)

    #training(net, train_loader, test_loader, optimizer, lossfunc, batch_size, 1)
