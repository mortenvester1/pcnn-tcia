import os
import pdb
import sys
import copy
import json
import time
import uuid
import pickle

import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

IMG_TYPES = ["adc", "bval", "ktrans"]
PATH = None
HLEN = 106

#TRAIN_LOADER = TEST_LOADER = CLASSES = None
np.random.seed(1337)
torch.manual_seed(7331)

NO_EPOCHS = 40
BATCH_SIZE = 64
LR = 0.001
MOM = 0.9
DROPOUT_RATE = 0.20
LOSSFUNC = nn.CrossEntropyLoss()
LOSS_NAME = "CrossEntropy"

OPTIM_NAME = "ADAM"
get_optimizer = lambda param: optim.Adam( param, lr = LR, weight_decay = 0.005)
get_scheduler = lambda opt: lrs.ReduceLROnPlateau(opt, 'min', factor = 0.5, patience = 3, min_lr = 0.0001, verbose = True) if opt is not None else None

class pcNN3D(nn.Module):
    def __init__(self):
        super(pcNN3D, self).__init__()
        self.epochs_trained = 0
        self.set_layers()


    def set_layers(self):
        """
        Conv/Poll -> Dropout -> BN -> Activation
        """
        # Convolutions
        self.layer1 = nn.Sequential(
            nn.Conv3d(3,4,(1,3,3)),
            nn.InstanceNorm3d(4),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(4,4,(3,3,3)),
            nn.InstanceNorm3d(4),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(4,8,(1,3,3)),
            nn.InstanceNorm3d(8),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer4 = nn.Sequential(
            nn.Conv3d(8,8,(3,3,3)),
            nn.InstanceNorm3d(8),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer5 = nn.Sequential(
            nn.MaxPool3d((1,2,2)),
            nn.InstanceNorm3d(8),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer6 = nn.Sequential(
            nn.Conv3d(8,16,(1,3,3)),
            nn.InstanceNorm3d(16),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer7 = nn.Sequential(
            nn.Conv3d(16,16,(3,3,3)),
            nn.InstanceNorm3d(16),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer8 = nn.Sequential(
            nn.Conv3d(16,32,(1,3,3)),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer9 = nn.Sequential(
            nn.Conv3d(32,32,(3,3,3)),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer10 = nn.Sequential(
            nn.Conv3d(32,64,(3,3,3)),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Dropout(p = DROPOUT_RATE)
        )

        self.layer11Dense = nn.Sequential(
            nn.Linear(512, 192),
            nn.ReLU()
        )
        self.layer12Dense = nn.Sequential(
            nn.Linear(192, 90),
            nn.ReLU()
        )
        self.layer13Dense = nn.Sequential(
            nn.Linear(90, 2),
            nn.Softmax()
        )


    def set_info(self, optimzer_name, lossfunc_name):
        self.optimizer = optimzer_name
        self.lossfunc = lossfunc_name
        self.train_loss = np.inf
        self.valid_loss = np.inf
        self.test_loss = np.inf
        self.train_pct = 0
        self.valid_pct = 0
        self.test_pct = 0
        self.total_time = 0


    def forward(self, batch):
        x = self.layer1(batch)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = x.view(-1, 512)
        x = self.layer11Dense(x)
        x = self.layer12Dense(x)
        x = self.layer13Dense(x)
        #pdb.set_trace()
        return x


class TCIADataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform

        self.h5 = None
        self.keys = None
        self.classes = None
        self.img_types = None
        self.load_file_keys()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        name, label = self.keys[idx]
        paths = ["/" + "/".join([label,it,name]) for it in self.img_types]
        arrs = [self.h5[path][:] for path in paths]
        arr = torch.FloatTensor(arrs).transpose(1,3)
        label = self.classes.index(label)
        return arr, label

    def load_file_keys(self):
        self.h5 = h5py.File(self.root, 'r')

        # Get Classes
        h5path = '/'
        self.classes = [ key for key in self.h5[h5path].keys()]

        # Get Image Types
        h5path += self.classes[0]
        self.img_types = IMG_TYPES#[ key for key in self.h5[h5path].keys()]

        # Get File names
        h5path += "/" + self.img_types[0]
        names = [ key for key in self.h5[h5path].keys()]
        self.keys = [ (name, label) for name in names for label in self.classes ]

        self.split_train_valid()

        return

    def split_train_valid(self, pct = 30):
        self.k = int(len(self.keys) * pct / 100.)
        self.keys = np.random.permutation(self.keys).tolist()
        return


def build_loaders(data_dir, batch_size):
    """get train and test data """

    print_border()
    print_header("Building dataloaders for {0} images".format(", ".join(IMG_TYPES)))

    classes = ("POS", "NEG")

    train_root = data_dir + "/traindata/volumes/train.h5"
    test_root = data_dir + "/testdata/volumes/test.h5"
    transform = None
    trainset = TCIADataset(train_root, transform)
    validset = TCIADataset(train_root, transform)
    # Split into train and validation set
    validset.keys = trainset.keys[-trainset.k:]
    trainset.keys = trainset.keys[:-trainset.k]
    #pdb.set_trace()
    testset = TCIADataset(test_root, transform)

    train_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = 0)

    valid_loader = torch.utils.data.DataLoader(validset,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = 0)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = 0)

    print_header("Dataloaders Ready!")
    print_border()

    return train_loader, valid_loader, test_loader, classes


def l2_regularization(net, loss, gamma = 0.005):
    li_reg_loss = 0
    for m in net.modules():
        if isinstance(m,nn.Linear):
            temp_loss = torch.sum(((torch.sum(((m.weight.data)**2),1))**0.5),0)
            li_reg_loss += temp_loss

    loss += Variable(gamma * li_reg_loss, requires_grad= True)
    return loss


def training(net, optimizer, lossfunc, number_of_epochs = 1, scheduler = None):
    """ Train a NN for n epochs"""
    #global TRAIN_LOADER, TEST_LOADER
    if number_of_epochs == 0:
        return np.inf, np.inf

    print_border()
    print_header("Training")
    print_border()

    #pdb.set_trace()
    print( '{0:<10s}\t{1:>10s}\t{2:>10s}\t{3:>10s}\t{4:>10s}\t{5:>10s}\t{6:>10s}'
        .format( 'Epoch', 'Batch#', 'Loss Train' , 'Loss Valid', 'pct Train', 'pct Valid', 'Time') )

    total_time = net.total_time
    for epoch in range(number_of_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct_count = 0
        total_count = 0
        net.train()
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

            if (j + 1) % 128 == 0:
                temp_time = time.time() - start_time
                print('{0:>10d}\t{1:>10d}\t{2:>10.5f}\t{3:>10s}\t{4:>10.5f}\t{5:>10s}\t{6:>10.5f}'
                    .format( epoch + 1, j + 1, running_loss / (j + 1), "", train_pct, "", temp_time))
            else:
                print('{0:<10s}\t{1:>10d}'.format( "Training", j + 1), end = "\r", flush = True)

        temp_time = time.time() - start_time
        print('{0:>10d}\t{1:>10d}\t{2:>10.5f}\t{3:>10s}\t{4:>10.5f}\t{5:>10s}\t{6:>10.5f}'
            .format( epoch + 1, j + 1, running_loss / (j + 1), "", train_pct, "", temp_time))

        train_loss = running_loss / (j + 1)
        net.train_loss = train_loss
        net.train_pct = train_pct

        valid_loss, valid_pct = validation(net, lossfunc, VALID_LOADER)
        net.valid_loss = valid_loss
        net.valid_pct = valid_pct

        net.epochs_trained += 1
        etime  = time.time() - start_time
        total_time += etime
        print('{0:<10s}\t{1:<10s}\t{2:>10.5f}\t{3:>10.5f}\t{4:>10.5f}\t{5:>10.5f}\t{6:>10.5f}'
            .format( "Results", "", net.train_loss, net.valid_loss, net.train_pct, net.valid_pct, etime))

        print_border()
        print( '{0:<10s}\t{1:>10s}\t{2:>10s}\t{3:>10s}\t{4:>10s}\t{5:>10s}\t{6:>10s}'
            .format( 'Epoch', 'Batch#', 'Loss Train' , 'Loss Valid', 'pct Train', 'pct Valid', 'Time') )

        if net.train_pct > 0.90 and net.valid_pct > 0.90:
            print_border()
            print_header("!!!Early termination!!!")
            break

        # Adjusting Learning rate
        if scheduler is not None:
            scheduler.step(valid_loss)

    print_border()
    print_header("Total Training Time :{0:1.9f}".format(total_time))
    print_border()
    net.total_time += total_time

    print_border()
    print_header("TESTING")
    test_loss, test_pct = validation(net, lossfunc, TEST_LOADER, 'testing')
    print_header("TEST LOSS {0:5.4f}, TEST PCT {1:5.4f}".format(test_loss, test_pct))
    print_border()
    net.test_loss = test_loss
    net.test_pct = test_pct

    save_net_info(net, optimizer, lossfunc)

    return net


def validation(net, lossfunc, loader, loader_type = 'validating'):
    """ Loop of test data and compute test loss"""
    net.eval()
    valid_loss = 0.0
    start_time = time.time()
    correct_count, total_count = 0, 0
    for j, test_batch in enumerate(loader, 0):
        inputs, labels = test_batch[0], test_batch[1]
        inputs, labels = Variable(inputs), Variable(labels)
        result = net(inputs)

        corr, total = count_correct(result.data.numpy(), labels.data.numpy())
        correct_count += corr
        total_count += total

        valid_loss += lossfunc(result, labels).data[0]

        print('{0:<10s}\t{1:>10d}\t{2:>10.5f}'.format( loader_type, j + 1, net.train_loss), end = "\r", flush = True)

    valid_loss = valid_loss / (j + 1)
    valid_pct = correct_count / total_count

    return valid_loss, valid_pct


def count_correct(res, lab):
    res = np.argmax(res, axis = 1)
    corr = np.sum(res == lab)
    count = len(res)
    return corr, count


def save_net_info(net, optimizer, lossfunc):
    path = os.path.realpath(PATH+"/../../")
    with open(path + "/top5_info.json",'r') as jsoninfo:
        top_info = json.load(jsoninfo)
    pdb.set_trace()
    compare = [net.valid_loss < top_loss for top_loss in top_info['top_5_valid_loss']]

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
            "valid_loss" : net.valid_loss,
            "train_pct" : net.train_pct,
            "test_pct" : net.test_pct,
            "valid_pct" : net.test_pct,
            "train_time" : net.total_time,
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

        top_info['top_5_valid_loss'].insert(rank - 1, net.valid_loss)
        top_info['top_5_train_loss'].insert(rank - 1, net.train_loss)
        top_info['info'].insert(rank - 1, net_results)

        top_info['top_5_valid_loss'].pop(-1)
        top_info['top_5_train_loss'].pop(-1)
        old = top_info['info'].pop(-1)

        # Create dir move old
        os.mkdir( path + "/nets/" + net_name)
        cpcall = "cp %s/pcNN3D.py %s" % (path + "/src", path + "/nets/" + net_name + "/")
        os.system(cpcall)

        movecall = "mv %s/ %s" % (path + "/nets/" + old['net_name'],
                             path + "/nets/old" )
        os.system(movecall)

        filename = path + "/nets/" + net_name + "/pcNN3D.pk"
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
    data_dir = PATH.rstrip('src/pcNN3D.py')

    net = pcNN3D()
    net.set_info(LOSS_NAME, OPTIM_NAME)

    TRAIN_LOADER, VALID_LOADER, TEST_LOADER, CLASSES = build_loaders(data_dir, BATCH_SIZE)

    OPTIMIZER = get_optimizer( net.parameters() )
    SCHEDULER = get_scheduler(OPTIMIZER)

    net = training(net, OPTIMIZER, LOSSFUNC, NO_EPOCHS, SCHEDULER)
    #gen = enumerate(VALID_LOADER)
    #inp = next(gen)
    #inp = Variable(inp[1][0])
    #res = net(inp)
    #net.eval()
    #res2 = net(inp)
    #save_net_info(net, optimizer, lossfunc)
    #net1, optimizer1, lossfunc1 = load_ranked_n(1)
