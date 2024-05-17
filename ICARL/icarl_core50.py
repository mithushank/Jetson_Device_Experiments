import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import  CORe50
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics,forgetting_metrics
from avalanche.logging import InteractiveLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training import ICaRL
from avalanche.models import IcarlNet
from torchvision import models
from avalanche.training import  ICaRL,ReservoirSamplingBuffer
from avalanche.benchmarks.scenarios import CLStream,  NCExperience

from tegrastats_module import *
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics 

import math
import time
import os
import csv

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

benchmark = CORe50(run=1,mini=True, scenario = 'nc',object_lvl=False)
# Define ResNet18-based feature extractor
feature_extractor = FeatureExtractor(model_name='resnet18')

# Define SGD optimizer
optimizer = SGD(feature_extractor.parameters(), lr=0.01, momentum=0.9)
# Define loss criterion
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = [InteractiveLogger(),CSVLogger(log_folder = "icarl/core50")]
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

# Define ICaRL strategy
strategy = ICaRL(
    feature_extractor= IcarlNet(num_classes=50),
    classifier=nn.Linear(in_features=50, out_features=50, bias=True),
    optimizer=optimizer,
    memory_size=1000,
    buffer_transform=None,
    fixed_memory=True,
    train_epochs=1,
    train_mb_size=50,
    eval_mb_size = 50,
    plugins=[eval_plugin, replay_plugin]
)

# start tracking using tegrastats
interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval)

# Train the model

# start tracking using tegrastats
interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval,output_file='tegrastats_output_icarl_core50.txt')

# Train the model

coreset_percent = 100
#time start
start = time.time()
for i,batch in enumerate(benchmark.train_stream):
    old_ds = benchmark.train_stream[i].dataset
    new_ds = old_ds.subset(range(0, len(old_ds),math.floor(100/coreset_percent)))  # select coreset randomly
    stream = CLStream(name='train', exps_iter = new_ds, benchmark = benchmark)
    batch_new = NCExperience(stream,i)
    strategy.train(batch_new)
    strategy.eval(benchmark.test_stream)

#time end
end = time.time()

elapse = end - start

# stop the tracking
stop_tegrastats(tegrastats_process)
time.sleep(1)  # Wait a bit for tegrastats to terminate cleanly
# Process the output file to analyze the GPU energy consumption
total_gpu_soc_power, total_cpu_cv_power, total_power = parse_tegrastats_output('tegrastats_output_icarl_core50.txt', interval)

print("GPU power: " + str(total_gpu_soc_power) + " mJ")
print("CPU power: " + str(total_cpu_cv_power) + " mJ")
print("Total power: " + str(total_power) + " mJ")

print('time taken for the training process: ', elapse)
# Get the default directory to store the CSV file

data = [
    ['elapsed time', elapse],
    ['GPU power', total_gpu_soc_power],
    ['CPU power', total_cpu_cv_power],
    ['total power', total_power],
]
default_directory = os.getcwd()  # Get the current working directory

# Specify the file name
csv_file_name = 'results_icarl_core50.csv'

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