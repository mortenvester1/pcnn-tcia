import os
import pdb
import sys
import json
import uuid
import time
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable

IMG_TYPE = "ADC"
PATH = None
HLEN = 106

TRAIN_LOADER = TEST_LOADER = CLASSES = None

NO_EPOCHS = 5
BATCH_SIZE = 100
LR = 0.001
MOM = 0.9

LOSSFUNC = nn.CrossEntropyLoss()
LOSS_NAME = "CrossEntropy"

OPTIM_NAME = "SGD"
get_optimer = lambda net: optim.SGD( net, lr = LR, momentum = MOM)

class pcNN(nn.Module):
    def __init__(self):
        super(pcNN, self).__init__()
        self.epochs_trained = 0
        self.set_layers()


    def set_layers(self):
        # Convolutions
        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels = 3,
                               kernel_size = 3,
                               stride = 1,
                               padding = 0,
                               dilation = 1,
                               groups = 1,
                               bias = True)

        self.BN = nn.BatchNorm2d(num_features = 3,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True)

        self.max_pool  = nn.MaxPool2d(kernel_size = 2,
                                  stride = 1,
                                  padding = 0,
                                  dilation = 1,
                                  ceil_mode = False,
                                  return_indices = False)

        self.fc1 = nn.Linear(48, 24)
        self.fc2 = nn.Linear(24, 2)

    def set_info(self, optimzer_name, lossfunc_name):
        self.optimizer = optimzer_name
        self.lossfunc = lossfunc_name
        self.train_loss = np.inf
        self.test_loss = np.inf
        self.train_pct = 0
        self.test_pct = 0
        self.total_time = 0


    def forward(self, x):
        # Layer 1, 3x3 Conv, Batch Norm, Relu
        x = self.conv1(x)
        x = F.relu(self.BN(x))

        # Layer 2, 3x3 Conv, Batch Norm, Relu
        x = self.conv1(x)
        x = F.relu(self.BN(x))

        # Layer 3, 2x2 Max_pool, 3x3 Conv, Batch Norm, Relu
        x = F.max_pool2d(x, kernel_size = 2)
        x = self.conv1(x)
        x = F.relu(self.BN(x))

        # Layer 4, 3x3 Conv, Batch Norm, Relu
        x = self.conv1(x)
        x = F.relu(self.BN(x))

        # Layer 5, 3x3 Conv, Batch Norm, Relu
        x = self.conv1(x)
        x = F.relu(self.BN(x))

        # Layer 6, 2x2 Max_pool
        x = F.max_pool2d(x, kernel_size = 2)

        # Layer 7, Reshape -> 1 x batch_size * PROD(img_dim)
        # FC1 48 -> BATCH_SIZE * 24
        x = x.view(-1, 3 * 4 * 4)
        x = self.fc1(x)
        x = F.relu(x)

        # Layer 8, BATCH_SIZE * 24 -> batch_size
        x = self.fc2(x)
        x = F.softmax(x)

        return x



def build_loaders(data_dir, batch_size, img_type):
    """get train and test data """

    print_border()
    print_header("Building dataloaders for {0} images".format(img_type))

    classes = ("POS", "NEG")

    train_root = data_dir + "/traindata/patches/" + img_type + "/"
    test_root = data_dir + "/testdata/patches/" + img_type + "/"
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = dset.ImageFolder(root=train_root,
                                transform=transform)

    testset = dset.ImageFolder(root = test_root,
                               transform = transform)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = 0)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = 0)

    print_header("Dataloaders Ready!")
    print_border()

    return train_loader, test_loader, classes


def training(net, optimizer, lossfunc, number_of_epochs = 1):
    """ Train a NN for n epochs"""
    #global TRAIN_LOADER, TEST_LOADER
    if number_of_epochs == 0:
        return np.inf, np.inf

    print_border()
    print_header("Training")
    print_border()

    #pdb.set_trace()
    print( '{0:<10s}\t{1:>10s}\t{2:>10s}\t{3:>10s}\t{4:>10s}\t{5:>10s}\t{6:>10s}'
        .format( 'Epoch', 'Batch#', 'Loss Train' , 'Loss Test', 'pct Train', 'pct Test', 'Time') )

    total_time = net.total_time
    for epoch in range(number_of_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct_count, total_count = 0, 0
        for j, train_batch in enumerate(TRAIN_LOADER, 0):
            inputs, labels = train_batch[0], train_batch[1]
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            result = net(inputs)
            loss = lossfunc(result, labels)
            corr, total = count_correct(result.data.numpy(), labels.data.numpy())
            correct_count += corr
            total_count += total
            train_pct = correct_count / total_count

            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if (j + 1) % 1000 == 0:
                temp_time = time.time() - start_time
                print('{0:>10d}\t{1:>10d}\t{2:>10.5f}\t{3:>10.5f}\t{4:>10.5f}\t{5:>10s}\t{6:>10.5f}'
                    .format( epoch + 1, j + 1, running_loss / (j + 1), net.test_loss, train_pct, "", temp_time))
            else:
                print('{0:<10s}\t{1:>10d}'.format( "Training", j + 1), end = "\r", flush = True)

        temp_time = time.time() - start_time
        print('{0:>10d}\t{1:>10d}\t{2:>10.5f}\t{3:>10.5f}\t{4:>10.5f}\t{5:>10s}\t{6:>10.5f}'
            .format( epoch + 1, j + 1, running_loss / (j + 1), net.test_loss, train_pct, "", temp_time))

        train_loss = running_loss / (j + 1)
        net.train_loss = train_loss
        test_loss, test_pct = testing(net, lossfunc)
        net.test_loss = test_loss

        net.epochs_trained += 1
        etime  = time.time() - start_time
        total_time += etime
        print('{0:<10s}\t{1:<10s}\t{2:>10.5f}\t{3:>10.5f}\t{4:>10.5f}\t{5:>10.5f}\t{6:>10.5f}'
            .format( "Results", "", net.train_loss, net.test_loss, train_pct, test_pct, etime))
        print_border()
        print( '{0:<10s}\t{1:>10s}\t{2:>10s}\t{3:>10s}\t{4:>10s}\t{5:>10s}\t{6:>10s}'
            .format( 'Epoch', 'Batch#', 'Loss Train' , 'Loss Test', 'pct Train', 'pct Test', 'Time') )


    print_border()
    print_header("Total Training Time :{0:1.9f}".format(total_time))
    print_border()
    net.train_loss = train_loss
    net.total_time += total_time

    save_net_info(net, optimizer, lossfunc)
    return net


def testing(net, lossfunc):
    """ Loop of test data and compute test loss"""
    test_loss = 0.0
    start_time = time.time()
    correct_count, total_count = 0, 0
    for j, test_batch in enumerate(TEST_LOADER, 0):
        inputs, labels = test_batch[0], test_batch[1]
        inputs, labels = Variable(inputs), Variable(labels)
        result = net(inputs)

        corr, total = count_correct(result.data.numpy(), labels.data.numpy())
        correct_count += corr
        total_count += total

        test_loss += lossfunc(result, labels).data[0]

        print('{0:<10s}\t{1:>10d}\t{2:>10.5f}'.format( "Testing", j + 1, net.train_loss), end = "\r", flush = True)

    test_loss = test_loss / (j + 1)
    test_pct = correct_count / total_count

    return test_loss, test_pct


def count_correct(res, lab):
    res = np.argmax(res, axis = 1)
    corr = np.sum(res == lab)
    count = len(res)
    return corr, count


def save_net_info(net, optimizer, lossfunc):
    path = os.path.realpath(PATH+"/../../")
    with open(path + "/top5_info.json",'r') as jsoninfo:
        top_info = json.load(jsoninfo)

    compare = [net.test_loss < top_loss for top_loss in top_info['top_5_test_loss']]

    print_border()
    if any(compare):
        net_name = str(uuid.uuid4()).split("-")[-1]
        rank = 6 - sum(compare)

        print_header("Neural Net ranked {0:d}".format(rank))
        print_header("Saving net, optimizer, loss function and information")

        net_results = {
            "net_name" : net_name ,
            "rank" : rank,
            "net_parameters" : str(net),
            "train_loss" : net.train_loss,
            "test_loss" : net.test_loss,
            "train_pct" : net.train_pct,
            "test_pct" : net.test_pct,
            "optimizer_name" : net.optimizer,
            "optimizer_info" : None,
            "lossfunc_name" : net.lossfunc,
            "lossfunc_info" : dict(lossfunc.state_dict())
        }
        try:
            net_results['optimizer_info'] = dict(optimizer.state_dict())["param_groups"]
            net_results['lossfunc_info'] = dict(lossfunc.state_dict())["param_groups"]
        except:
            pass

        top_info['top_5_test_loss'].insert(rank - 1, net.test_loss)
        top_info['top_5_train_loss'].insert(rank - 1, net.train_loss)
        top_info['info'].insert(rank - 1, net_results)

        top_info['top_5_test_loss'].pop(-1)
        top_info['top_5_train_loss'].pop(-1)
        old = top_info['info'].pop(-1)

        # Create dir move old
        os.mkdir( path + "/nets/" + net_name)
        movecall = "mv %s/ %s" % (path + "/nets/" + old['net_name'],
                             path + "/nets/old" )
        os.system(movecall)

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

        print_header("Files saved in folder {0:s}".format(net_name))
        print_border()

    else:
        print_header("Neural Net did not rank in top 5")
        print_border()

    return

def print_header(header):
    prl = (HLEN//2-len(header)//2) - 1
    prr = HLEN - prl - len(header) - 2
    print("#" + " "*prl + header + " "*prr + "#")
    return

def print_border():
    print("-"*HLEN)
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
    data_dir = PATH.rstrip('src/pcNN.py')

    TRAIN_LOADER, TEST_LOADER, CLASSES = build_loaders(data_dir, BATCH_SIZE, IMG_TYPE)

    net = pcNN()
    net.set_info(LOSS_NAME, OPTIM_NAME)

    OPTIMIZER = get_optimer( net.parameters() )
    net = training(net, OPTIMIZER, LOSSFUNC, NO_EPOCHS)
    #net = testing(net, LOSSFUNC)

    #save_net_info(net, optimizer, lossfunc)
    #net1, optimizer1, lossfunc1 = load_ranked_n(1)
