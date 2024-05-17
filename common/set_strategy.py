import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import models, transforms
from torch.nn.functional import relu, avg_pool2d

from avalanche.logging import InteractiveLogger,CSVLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.models import  MTSlimResNet18, SlimResNet18
from avalanche.training import AGEM,Naive,GSS_greedy,ICaRL
from avalanche.benchmarks.scenarios import CLStream, NCExperience
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer,ClassBalancedBuffer
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from tegrastats_module import *
import numpy
import math
import torch
import matplotlib.pyplot as plt
import csv
import time
import os
from functools import reduce
from typing import Union


""" -----------------------------CustomResnet for AGEM_PMNIST---------------------------------------"""

class CustomMTSlimResNet18_agem_pmnist(MTSlimResNet18):
    def __init__(self, nclasses, nf=20):
        super().__init__(nclasses, nf)
        # Change the number of input channels for the first convolutional layer
        self.conv1 = nn.Conv2d(1, nf * 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, task_labels):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x)))  # Adjusted input layer
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out, task_labels)  # Adjusted output layer
        return out

""" -----------------------------------Replay_pluggin for AGEM_CIFAR--------------------------------"""

class ReplayP_agem_cifar(SupervisedPlugin):
    def __init__(self, mem_size,num_class=2,i=1):
        super().__init__()
        # self.buffer = ReservoirSamplingBuffer(max_size=mem_size)
        self.buffer = ClassBalancedBuffer(max_size=mem_size, total_num_classes=num_class*i,adaptive_size=False)


    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Use a custom dataloader to combine samples from the current data and memory buffer. """
        if len(self.buffer.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.buffer.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """ Update the buffer. """
        self.buffer.update(strategy, **kwargs)
 
"""----------------------------------Custom model for AGEM_CORE50----------------------------------""" 
class CustomMTSlimResNet18_agem_core50(MTSlimResNet18):
    def __init__(self, nclasses, nf=20):
        super().__init__(nclasses, nf)
        # Change the number of input channels for the first convolutional layer
        self.conv1 = nn.Conv2d(3, nf * 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, task_labels):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x)))  # Adjusted input layer
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 12)
        out = out.view(out.size(0), -1)
        out = self.linear(out, task_labels)  # Adjusted output layer
        return out

"""------------------------------------------Replay plugin for ICARL_PMNIST-------------------------------------------"""

class ReplayP_icarl(SupervisedPlugin):

    def __init__(self, mem_size):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        self.buffer = ReservoirSamplingBuffer(max_size=mem_size)

    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Use a custom dataloader to combine samples from the current data and memory buffer. """
        if len(self.buffer.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.buffer.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """ Update the buffer. """
        self.buffer.update(strategy, **kwargs)

"""----------------------------------------FeatureExtracter for ICARL_PMNIST------------------------------------"""

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_name='resnet18'):
        super(FeatureExtractor, self).__init__()
        
        # Load the specified pretrained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        # Add more options for other models as needed
        
        # Replace the last fully connected layer with an identity layer
        if 'resnet' in model_name:
            self.model.fc = torch.nn.Identity()
        elif 'vgg' in model_name:
            self.model.classifier[-1] = torch.nn.Identity()
        # Add similar modifications for other models if required

    def forward(self, x,dataset = 'SplitMNIST'):
        # Convert grayscale images to RGB by duplicating the single channel
        '''
        For the different types of datasets we need to adjust the number of channels
        for SplitMNIST: (1,3,1,1)
        for SplitCIFAR10: (1,1,1,1)
        for SplitCIFAR100: (1,1,1,1)
        '''
        if dataset=='SplitCIFAR10':
            x = x.repeat(1, 1, 1, 1)     
        elif dataset==('SplitMNIST' or 'SplitFMNIST'):
            x = x.repeat(1, 3, 1, 1)     
        return self.model(x)

"""----------------------------------------classifier function for ICARL_PMNIST"""

# Define classifier model
class Classifier_icarl_pmnist(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

"""--------------------------------------main strategies---------------------------------------"""

def agem_strategy(name):
    name = name.lower()
    """Parameters define here"""

    mem_size = 1000
    batch_size = 50
    train_epochs = 1

    if name == "pmnist":
        model = CustomMTSlimResNet18_agem_pmnist(nclasses=10)
        replay_plugin = ReplayPlugin(
            mem_size=1000,
            storage_policy=ReservoirSamplingBuffer(max_size=mem_size)
        )
        logger = [InteractiveLogger(),CSVLogger(log_folder = "agem/pmnist")]

    elif name == "cifar10":
        model = MTSlimResNet18(nclasses=10)
        replay_plugin = ReplayP_agem_cifar(mem_size = mem_size, num_class =10)
        logger = [InteractiveLogger(),CSVLogger(log_folder = "agem/cifar10")]

    elif name == "cifar100":
        model = MTSlimResNet18(nclasses=100)
        replay_plugin = ReplayP_agem_cifar(mem_size = mem_size, num_class = 100)
        logger = [InteractiveLogger(),CSVLogger(log_folder = "agem/cifar100")]

    elif name == "core50":
        model = CustomMTSlimResNet18_agem_core50(nclasses=50)
        replay_plugin = ReplayPlugin(
            mem_size = mem_size,
            batch_size=50,
            storage_policy=ReservoirSamplingBuffer(max_size=mem_size)
        )
        logger = [InteractiveLogger(),CSVLogger(log_folder = "agem/core50")]


    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Define logger and evaluation plugin
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        cpu_usage_metrics(experience=True),
        forgetting_metrics(experience=True, stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),    
        loggers=logger
    )

    strategy = AGEM(
        model = model,
        optimizer = optimizer,
        criterion = criterion,
        train_mb_size = batch_size,
        eval_mb_size = batch_size,
        patterns_per_exp=mem_size,
        sample_size=mem_size,
        train_epochs = train_epochs,
        plugins = [eval_plugin, replay_plugin]
    )
    print(type(strategy))

    return strategy


def icarl_strategy(name):
    name = name.lower()
    """Parameters define here"""

    mem_size = 1000
    batch_size = 50
    train_epochs = 1

    if name == "pmnist":
        feature_extractor = Icarlnet(nclasses=10)
        classfier = Classifier_icarl_pmnist(input_size=512, num_classes = 10)

        replay_plugin = ReplayP_icarl(
            mem_size=mem_size
        )
        logger = [InteractiveLogger(),CSVLogger(log_folder = "icarl/pmnist")]

    elif name == "cifar10":
        feature_extractor= IcarlNet(num_classes=10)
        classifier=nn.Linear(in_features=10, out_features=10, bias=True)
        replay_plugin = ReplayP_icarl(
            mem_size=mem_size
        )        
        logger = [InteractiveLogger(),CSVLogger(log_folder = "icarl/cifar10")]

    elif name == "cifar100":
        feature_extractor = IcarlNet(num_classes=100)
        classifier=nn.Linear(in_features=100, out_features=100, bias=True)
        replay_plugin = ReplayP_icarl(
            mem_size=mem_size
        )        
        logger = [InteractiveLogger(),CSVLogger(log_folder = "icarl/cifar100")]

    elif name == "core50":
        feature_extractor = IcarlNet(num_classes=50)
        classifier = nn.Linear(in_features=50, out_features=50, bias=True)
        replay_plugin = ReplayPlugin(
            mem_size = mem_size,
            batch_size=50,
            storage_policy=ReservoirSamplingBuffer(max_size=mem_size)
        )
        logger = [InteractiveLogger(),CSVLogger(log_folder = "icarl/core50")]


    optimizer = SGD(feature_extractor.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Define logger and evaluation plugin
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        cpu_usage_metrics(experience=True),
        forgetting_metrics(experience=True, stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),    
        loggers=logger
    )

    strategy = ICaRL(
        feature_extractor = feature_extractor,
        classifier = classifier,
        optimizer = optimizer,
        memory_size = mem_size,
        buffer_transform = None,
        fixed_memory = True,
        train_epochs = train_epochs,
        train_mb_size = batch_size,
        eval_mb_size = batch_size,
        plugins = [eval_plugin, replay_plugin]
    )

    return strategy


"""    Need to complete the GSS_greedy algorithm    """
# def gss_greedy_strategy(name):
#     if nam

