"""
cifar10.py

python script to be run on colab to import the cifar 10 dataset

in addition, it contains function useful to split into train+validation
sets and build pytorch dataloaders to access the data
"""

# imports
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

########################################################################
### 1 -- importing CIFAR 10 dataset
########################################################################

def import_cifar10(data_path= '/content/data',normalize = True, test = True):
    """
    downloads and returns the cifar-10 training set
    :param data_path: path indicating where the data will be stored
    :param normalize: when true, pixel are normalized and scaled to [-1,1]
    :param test: when True, the test set is downloaded as well
    :return: a tuple containing the training set, test set and a tuple
     of all the class names in order
    """

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if normalize:
        # bring pixel values to [-1,1]
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.ToTensor()

    # load train set
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform)
    if test:
        # load test set
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)
        return trainset, testset, classes

    else:
        return trainset, classes


########################################################################
### 2 -- train/validation split
########################################################################

def train_val_split(trainset, valid_ratio=0.2, seed=0,
                    batch_size=8):
    """
    split the trainset into train + validation sets and

    :param trainset: torchvision datasets that will be split in 2
    :param valid_ratio: percentage of the dataset that will be used for validation
    :return: returns a dic of pytorch dataloaders for train, validation,
    and entire train sets (fulltrain) as well as a dic of their sizes.
    """

    # Random splitting the training set into train+validation
    num_samples = len(trainset)
    indices = list(range(num_samples))
    split = int(np.floor(valid_ratio * num_samples))

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    # Building the data samplers
    from torch.utils.data.sampler import SubsetRandomSampler

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # the SubsetRandomSampler creates a Sampler object that will feed indexes
    # randomly selected among the validation indices to the dataloader.

    # building the dataloaders

    dataloaders = {'train': torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                        sampler=train_sampler, num_workers=1,
                                                        pin_memory=True),
                   'val': torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      sampler=val_sampler, num_workers=1,
                                                      pin_memory=True),
                   'fulltrain': torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                            shuffle=True, num_workers=1,
                                                            pin_memory=True) }

    dataset_sizes = {'train': (num_samples-split) ,'val':split, 'fulltrain':num_samples}

    return dataloaders,  dataset_sizes


########################################################################
### 3 -- display images from CIFAR 10
########################################################################

# source : pytorch.com cifar_10 tutorial

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


