import torch
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.optim import SGD
from torchvision import models
from torchvision.models import resnet18,vgg19,resnet50,resnet34
from torchvision import transforms
from torchvision.transforms import ToTensor,ToPILImage
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, CLStream51, SplitCIFAR100,PermutedMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin,ReplayPlugin
from avalanche.models import MobilenetV1, SimpleMLP, MTSlimResNet18, SimpleCNN,slda_resnet,SlimResNet18
from avalanche.training import MER, ICaRL,AGEM,GSS_greedy
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario, NCExperience

from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.scenarios.generic_scenario import DatasetExperience
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.training import RandomExemplarsSelectionStrategy, HerdingSelectionStrategy,FeatureBasedExemplarsSelectionStrategy
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics

import numpy
import math
import matplotlib.pyplot as plt
import csv
import time
import os
from tegrastats_module import *
from functools import reduce
from typing import Union
       
# start tracking using tegrastats
interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval)

benchmark = SplitCIFAR10(n_experiences=5,seed=1234,train_transform=transforms.ToTensor(),eval_transform=transforms.ToTensor())
# Load the pre-trained ResNet18 model
model = MTSlimResNet18(nclasses=10)



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
    confusion_matrix_metrics(num_classes=10, save_image=True,
                             stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),    
    loggers=[logger]
)



class ReplayP(SupervisedPlugin):

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
 
# Define AGEM strategy
strategy = AGEM(
    model = model,
    patterns_per_exp=2000,
    optimizer = optimizer,
    criterion= criterion,
    plugins=[eval_plugin,ReplayP(mem_size=5000)],
    train_epochs=1,
    train_mb_size=50
    
)

strategy = GSS_greedy(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    mem_size=1000,
    train_epochs=1,
    train_mb_size=50,
    plugins=[eval_plugin]
)

print('--------------------------------------------------------start of train---------------------------------------------------------')
#time start
start = time.time()
coreset_percent = 100
for i,batch in enumerate(benchmark.train_stream):
    old_ds = benchmark.train_stream[i].dataset
    
    new_ds = old_ds.subset(range(0, len(old_ds),math.floor(100/coreset_percent)))  # select coreset randomly 
    print(len(old_ds),len(new_ds))
    stream = CLStream(name='train', exps_iter = new_ds, benchmark = benchmark)
    batch_new = NCExperience(stream,i)
    print(type(stream), type(batch_new))
    strategy.train(batch_new)    
    strategy.eval(benchmark.test_stream)

#time end
end = time.time()

elapse = end - start
stop_tegrastats(tegrastats_process)
time.sleep(1)  # Wait a bit for tegrastats to terminate cleanly

# Process the output file to analyze the GPU energy consumption
total_gpu_soc_power, total_cpu_cv_power, total_power = parse_tegrastats_output('tegrastats_output.txt', interval)

print("GPU power: " + str(total_gpu_soc_power) + " mJ")
print("CPU power: " + str(total_cpu_cv_power) + " mJ")
print("Total power: " + str(total_power) + " mJ")


elapse = end - start
print('time taken for the training process: ', elapse)
# Get the default directory to store the CSV file

data = [['elapsed Time', elapse]]
default_directory = os.getcwd()  # Get the current working directory

# Specify the file name
csv_file_name = 'results.csv'

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

