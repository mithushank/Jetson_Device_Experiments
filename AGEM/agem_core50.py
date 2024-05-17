import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import ReLU, AvgPool2d
from torch.nn.functional import relu, avg_pool2d

from torch.optim import SGD
from avalanche.benchmarks.classic import CORe50
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import MTSlimResNet18,SimpleMLP
from avalanche.training import RandomExemplarsSelectionStrategy, HerdingSelectionStrategy,Naive, ICaRL,AGEM,MIR,MER,ReservoirSamplingBuffer
from avalanche.training.plugins import EvaluationPlugin, AGEMPlugin, ReplayPlugin
import torchvision
from avalanche.training import MER, ICaRL,AGEM
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario,NCExperience

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
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics \


import numpy
import math
import time
import os
import csv
from tegrastats_module import *

# start tracking using tegrastats
interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval,output_file='tegrastats_output_agem_core50.txt')

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
        out = avg_pool2d(out, 12)
        out = out.view(out.size(0), -1)
        out = self.linear(out, task_labels)  # Adjusted output layer
        return out


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.485, 0.456, 0.406))
])
benchmark = CORe50(scenario='nc')
model = CustomMTSlimResNet18(nclasses=50)

# Create an optimizer
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

criterion = CrossEntropyLoss()

# Create logger and evaluation plugin
logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    # confusion_matrix_metrics(num_classes=10, save_image=True,stream=True),
    # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),  
    loggers=[logger]
)

replay_plugin = ReplayPlugin(
    mem_size = 1000,
    batch_size=50,
    storage_policy=ReservoirSamplingBuffer(max_size=1000)
)

strategy = AGEM(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    patterns_per_exp=10,
    train_mb_size=50,
    plugins = [eval_plugin,replay_plugin]
)
# print(f"Input tensor shape: {experience.train_stream[0]['x'].shape}")

coreset_percent = 100
#time start
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


#time end
end = time.time()

elapse = end - start
stop_tegrastats(tegrastats_process)
time.sleep(1)  # Wait a bit for tegrastats to terminate cleanly

# Process the output file to analyze the GPU energy consumption
total_gpu_soc_power, total_cpu_cv_power, total_power = parse_tegrastats_output('tegrastats_output_agem_core50.txt', interval)

print("GPU power: " + str(total_gpu_soc_power) + " mJ")
print("CPU power: " + str(total_cpu_cv_power) + " mJ")
print("Total power: " + str(total_power) + " mJ")


elapse = end - start
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
csv_file_name = 'results_agem_core50.csv'

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