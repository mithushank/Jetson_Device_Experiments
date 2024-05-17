import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18,vgg19,resnet50
from avalanche.benchmarks.classic import CORe50
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics,forgetting_metrics
from avalanche.logging import InteractiveLogger,CSVLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training import GSS_greedy
from avalanche.models import MTSlimResNet18,SimpleMLP
from torchvision import models
from avalanche.training import RandomExemplarsSelectionStrategy, HerdingSelectionStrategy,Naive, ICaRL,AGEM,MIR,MER,ReservoirSamplingBuffer
from tegrastats_module import *
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics

from torchvision import transforms
from torch.nn.functional import relu, avg_pool2d
import torch.nn as nn
import numpy
import math
import matplotlib.pyplot as plt
import csv
import time
import os
from functools import reduce
from typing import Union

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


benchmark = CORe50(run=1,mini=True, scenario = 'nc',object_lvl=False)

model = ResNet(num_classes=50,nf=20,num_blocks = [2, 2, 2, 2],block = BasicBlock)

# Define SGD optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
# Define loss criterion
criterion = CrossEntropyLoss()
# Define logger and evaluation plugin
logger = [InteractiveLogger(),CSVLogger(log_folder="gss_greedy/core50")]

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=logger
)

replay_plugin = ReplayPlugin(
    mem_size = 1000,
    batch_size=50,
    storage_policy=ReservoirSamplingBuffer(max_size=1000)
)

# Define GSS_greedy strategy
strategy = GSS_greedy(
    model = model,
    optimizer = optimizer,
    criterion = criterion,
    mem_size = 1000,
    train_mb_size=50,
    eval_mb_size=50,
    input_size = [3,32,32],
    plugins = [eval_plugin]
)


# # start tracking using tegrastats
interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval,output_file = 'tegrastats_output_gss_greedy_core50.txt')

# Train the model

#time start
start = time.time()

for i,batch in enumerate(benchmark.train_stream):
    strategy.train(batch)
    strategy.eval(benchmark.test_stream)
# stop the tracking
stop_tegrastats(tegrastats_process)
time.sleep(1)  # Wait a bit for tegrastats to terminate cleanly
#time end
end = time.time()

#time for training
elapse = end - start
print('time taken for the training process: ', elapse)

# Process the output file to analyze the GPU energy consumption
total_gpu_soc_power, total_cpu_cv_power, total_power = parse_tegrastats_output('tegrastats_output_gss_greedy_core50.txt', interval)

print("GPU power: " + str(total_gpu_soc_power) + " mJ")
print("CPU power: " + str(total_cpu_cv_power) + " mJ")
print("Total power: " + str(total_power) + " mJ")

data = [
    ['elapsed time', elapse],
    ['GPU power', total_gpu_soc_power],
    ['CPU power', total_cpu_cv_power],
    ['total power', total_power],
]
default_directory = "/media/microsd/stream_learning/results"  # Get the current working directory

# Specify the file name
csv_file_name = 'results_gss_greedy_core50.csv'

# Combine the directory path and file name
csv_file_path = os.path.join(default_directory, csv_file_name)

# Writing data to CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write data row by row
    for row in data:
        csv_writer.writerow(row)
print("Data has been written to", csv_file_path)
