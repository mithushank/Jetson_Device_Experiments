import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from avalanche.benchmarks.classic import  PermutedMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.models import  MTSlimResNet18, SlimResNet18
from avalanche.training import AGEM
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario, NCExperience
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics

from tegrastats_module import *
import os
import csv
import time
import numpy
import math
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from avalanche.models import MTSlimResNet18

class CustomMTSlimResNet18(MTSlimResNet18):
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



interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval,output_file='tegrastats_output_agem_pmnist.txt')


transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
# Load SplitFMNIST dataset
benchmark = PermutedMNIST(n_experiences=5,seed=1234, train_transform=transform, eval_transform = transform)
# model 
model = CustomMTSlimResNet18(nclasses=10)

# Define SGD optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
# Define Criterion CrossEntrophy
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = [InteractiveLogger(), CSVLogger(log_folder = "agem/pmnist")]
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=logger
)

replay_plugin = ReplayPlugin(
    mem_size=1000,
    storage_policy=ReservoirSamplingBuffer(max_size=1000)
)

strategy = AGEM(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    patterns_per_exp=1000,
    sample_size=1000,
    train_mb_size=50,
    eval_mb_size = 50,
    plugins=[eval_plugin,replay_plugin]
)
# Train the model
# Train the model
coreset_percent = 100

#start time
start = time.time()
for i,batch in enumerate(benchmark.train_stream):
    old_ds = benchmark.train_stream[i].dataset
    
    new_ds = old_ds.subset(range(0, len(old_ds),math.floor(100/coreset_percent)))  # select coreset randomly 
    print(len(old_ds),len(new_ds))
    stream = CLStream(name='train', exps_iter = new_ds, benchmark = benchmark)
    batch_new = NCExperience(stream,i)
    print(type(stream), type(batch_new))
    strategy.train(batch_new)   
    strategy.eval(benchmark.test_stream) 

#end time
end = time.time()

elapse = end - start
print('time',elapse)
# stop the tracking
stop_tegrastats(tegrastats_process)
time.sleep(1)  # Wait a bit for tegrastats to terminate cleanly

# Process the output file to analyze the GPU energy consumption
total_gpu_soc_power, total_cpu_cv_power, total_power = parse_tegrastats_output('tegrastats_output_agem_pmnist.txt', interval)

print("GPU power: " + str(total_gpu_soc_power) + " mJ")
print("CPU power: " + str(total_cpu_cv_power) + " mJ")
print("Total power: " + str(total_power) + " mJ")

data = [
    ['elapsed time', elapse],
    ['GPU power', total_gpu_soc_power], 
    ['CPU power', total_cpu_cv_power],
    ['total power', total_power],
]

# Get the default directory to store the CSV file
default_directory = "media/microsd/experiment_working/results"  # Get the current working directory

# Specify the file name
csv_file_name = 'results_agem_pmnist.csv'

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
