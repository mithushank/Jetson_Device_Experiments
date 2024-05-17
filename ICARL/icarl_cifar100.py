import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18,vgg19,resnet50
from avalanche.benchmarks.classic import SplitFMNIST, SplitCIFAR10,SplitCIFAR100, PermutedMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics,forgetting_bwt
from avalanche.logging import InteractiveLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ICaRL
from avalanche.models import icarl_resnet,IcarlNet, NCMClassifier, MTSlimResNet18
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from tegrastats_module import *
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario, NCExperience

import time
import csv
import os
import numpy as np
import math
from torchvision import models, transforms

interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval,output_file='tegrastats_output_icarl_cifar100.txt')

# password
# Define classifier model
class Classifier(torch.nn.Module):
    def __init__(self, input_size=64, num_classes=100):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x):
        return self.fc(x)

# Load SplitFMNIST dataset
benchmark = SplitCIFAR100(n_experiences=5, seed=1234)

class icarl_model(IcarlNet):
    def forward(self, x):
        x = self.feature_extractor(x)  # Already flattened
        return x

model = icarl_model(num_classes=100)
# Define ResNet18-based feature extractor
# feature_extractor = FeatureExtractor(model_name='resnet50')
# feature_extractor_d = FeatureExtractor(model_name='resnet18')
# feature_extractor_dd = FeatureExtractor(model_name='vgg16')
# Define SGD optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Define loss criterion
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = [InteractiveLogger(),CSVLogger(log_folder = "icarl/cifar100")]
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    loggers= logger
)
class ReplayP(SupervisedPlugin):

    def __init__(self, mem_size=1000):
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
        
strategy = ICaRL(
    feature_extractor=model,
    # classifier=nn.Linear(in_features=64, out_features=100, bias=True),
    classifier = Classifier(),
    optimizer=optimizer,
    memory_size=1000,
    buffer_transform=None,
    fixed_memory=True,
    train_mb_size=50,
    eval_mb_size = 50,
    plugins=[eval_plugin,ReplayP()]
)

coreset_percent = 100
current_time = time.time()

# for i,batch in enumerate(benchmark.train_stream):
#     old_ds = benchmark.train_stream[i].dataset
    
#     new_ds = old_ds.subset(range(0, len(old_ds),math.floor(100/coreset_percent)))  # select coreset randomly 
#     print(len(old_ds),len(new_ds))
#     stream = CLStream(name='train', exps_iter = new_ds, benchmark = benchmark)
#     batch_new = NCExperience(stream,i)
#     print(type(stream), type(batch_new))
#     strategy.train(batch_new)    
#     strategy.eval(benchmark.test_stream) 

for exp in benchmark.train_stream:
    strategy.train(exp)
    strategy.eval(benchmark.test_stream)

'''
same situation as cifar10 but accuracy  is quite low. need to find solution for that 
'''
new_time = time.time()

elapsed_time = new_time - current_time
print('Time to run the program: ',elapsed_time)
# stop the tracking
stop_tegrastats(tegrastats_process)
time.sleep(1)  # Wait a bit for tegrastats to terminate cleanly

# Process the output file to analyze the GPU energy consumption
total_gpu_soc_power, total_cpu_cv_power, total_power = parse_tegrastats_output('tegrastats_output_icarl_cifar100.txt', interval)

print("GPU power: " + str(total_gpu_soc_power) + " mJ")
print("CPU power: " + str(total_cpu_cv_power) + " mJ")
print("Total power: " + str(total_power) + " mJ")

data = [
    ['elapsed time', elapsed_time],
    ['GPU power', total_gpu_soc_power],
    ['CPU power', total_cpu_cv_power],
    ['total power', total_power],
]

# Get the default directory to store the CSV file
default_directory = os.getcwd()  # Get the current working directory

# Specify the file name
csv_file_name = 'results_icarl_cifar100.csv'

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