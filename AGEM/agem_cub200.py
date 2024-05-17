import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18,vgg19,resnet50
from torchvision import transforms
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitCUB200, CORe50,SplitTinyImageNet
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.models import MobilenetV1, SimpleMLP, MTSlimResNet18, SimpleCNN,SlimResNet18
from avalanche.training import MER, AGEM, Replay
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario, classification_scenario, online_scenario, NCExperience
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics

from torchvision import models

import numpy as np
import math

import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from avalanche.models import MTSlimResNet18
'''
Can able to run the model MTSlimresnet18 with splitcifar10
'''
class CustomMTSlimResNet18(MTSlimResNet18):
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
        out = avg_pool2d(out, 16)
        out = out.view(out.size(0), -1)

        # Adjusted output layer for CUB-200 (200 classes)
        out = self.linear(out,task_labels)  

        return out

transform = transforms.Compose([
    transforms.Resize(224),                # Resize to 224x224
    transforms.RandomCrop(224, padding=8), # Random crop with padding
    transforms.ToTensor(),                  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
])
# Load SplitFMNIST dataset
benchmark = SplitCUB200(n_experiences=5,seed=1234, train_transform=transform, eval_transform = transform)
# model 
model = CustomMTSlimResNet18(nclasses=200)

# Define SGD optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
# Define Criterion CrossEntrophy
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    # confusion_matrix_metrics(num_classes=10, save_image=True,
    #                          stream=True),
    # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),    
    loggers=[logger]
)

replay_plugin = ReplayPlugin(
    mem_size=1000,
    storage_policy=ReservoirSamplingBuffer(max_size=1000)
)

strategy = AGEM(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    patterns_per_exp=20,
    train_mb_size=50,
    plugins=[eval_plugin,replay_plugin]
)

# Train the model
coreset_percent = 20
for i,batch in enumerate(benchmark.train_stream):
    old_ds = benchmark.train_stream[i].dataset
    
    new_ds = old_ds.subset(range(0, len(old_ds),math.floor(100/coreset_percent)))  # select coreset randomly 
    print(len(old_ds),len(new_ds))
    stream = CLStream(name='train', exps_iter = new_ds, benchmark = benchmark)
    batch_new = NCExperience(stream,i)
    print(type(stream), type(batch_new))
    strategy.train(batch_new)    
    strategy.eval(benchmark.test_stream) 

# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD
# from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100,CORe50, SplitCUB200,CLStream51
# from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
# from avalanche.logging import InteractiveLogger
# from avalanche.models import MTSlimResNet18,SimpleMLP
# from avalanche.training import RandomExemplarsSelectionStrategy, HerdingSelectionStrategy,Naive, ICaRL,AGEM,MIR,MER,ReservoirSamplingBuffer
# from avalanche.training.plugins import EvaluationPlugin, AGEMPlugin, ReplayPlugin
# import torchvision

# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((32,32)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.485, 0.456, 0.406))
# ])

# benchmark = SplitCUB200(n_experiences=5, train_transform=transform, eval_transform=transform)

# model = MTSlimResNet18(nclasses=200)

# # Create an optimizer
# optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

# criterion = CrossEntropyLoss()

# # Create logger and evaluation plugin
# logger = InteractiveLogger()
# eval_plugin = EvaluationPlugin(
#     accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
#     loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
#     forgetting_metrics(experience=True, stream=True),
#     loggers=[logger]
# )

# replay_plugin = ReplayPlugin(
#     mem_size = 1000,
#     batch_size=50,
#     storage_policy=ReservoirSamplingBuffer(max_size=1000)
# )

# strategy = AGEM(
#     model=model,
#     optimizer=optimizer,
#     criterion=criterion,
#     patterns_per_exp=40,
#     train_mb_size=50,
#     plugins = [eval_plugin,replay_plugin]
# )

# # Train the model
# coreset_percent = 20
# for i,batch in enumerate(benchmark.train_stream):
#     old_ds = benchmark.train_stream[i].dataset
    
#     new_ds = old_ds.subset(range(0, len(old_ds),math.floor(100/coreset_percent)))  # select coreset randomly 
#     print(len(old_ds),len(new_ds))
#     stream = CLStream(name='train', exps_iter = new_ds, benchmark = benchmark)
#     batch_new = NCExperience(stream,i)
#     print(type(stream), type(batch_new))
#     strategy.train(batch_new)    
#     strategy.eval(benchmark.test_stream) 