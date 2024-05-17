import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18,vgg19,resnet50
from avalanche.benchmarks.classic import SplitFMNIST, SplitCIFAR10,SplitCIFAR100, PermutedMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics,forgetting_metrics
from avalanche.logging import InteractiveLogger,CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ICaRL
from torchvision import models
from tegrastats_module import *
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario, NCExperience
from avalanche.core import SupervisedPlugin
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario, NCExperience

import math
import time
import os
import csv
import numpy as np
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


# Define classifier model
class Classifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)
    
# start tracking using tegrastats
interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval)

benchmark = PermutedMNIST(n_experiences=5,seed=1234)

# Define ResNet18-based feature extractor
feature_extractor = FeatureExtractor(model_name='resnet18')

# Define SGD optimizer
optimizer = SGD(feature_extractor.parameters(), lr=0.01, momentum=0.9)
# Define loss criterion
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = [InteractiveLogger(),CSVLogger(log_folder = "icarl/pmnist")]
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    confusion_matrix_metrics(num_classes=10, save_image=True,
                             stream=True),
    # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),    
    loggers=logger
)

# Define ICaRL strategy
strategy = ICaRL(
    feature_extractor=feature_extractor,
    classifier=Classifier(512,10), 
    optimizer=optimizer,
    memory_size=1000,
    buffer_transform=None,
    fixed_memory=True,
    train_epochs=1,
    train_mb_size=50,
    eval_mb_size = 50,
    plugins=[eval_plugin,ReplayP(mem_size=1000)]
)


coreset_percent = 100

#start time
start = time.time()

# for i,batch in enumerate(benchmark.train_stream):
#     old_ds = benchmark.train_stream[i].dataset
    
#     new_ds = old_ds.subset(range(0, len(old_ds),math.floor(100/coreset_percent)))  # select coreset randomly 
#     print(len(old_ds),len(new_ds))
#     stream = CLStream(name='train', exps_iter = new_ds, benchmark = benchmark)
#     batch_new = NCExperience(stream,i)
#     print(type(stream), type(batch_new))
#     strategy.train(batch_new)   
 
for exp in benchmark.train_stream:
    strategy.train(exp)
    strategy.eval(benchmark.test_stream) 

#end time
end = time.time()

elapsed_time = end - start
# stop the tracking
stop_tegrastats(tegrastats_process)
time.sleep(1)  # Wait a bit for tegrastats to terminate cleanly

# Process the output file to analyze the GPU energy consumption
total_gpu_soc_power, total_cpu_cv_power, total_power = parse_tegrastats_output('tegrastats_output.txt', interval)

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


