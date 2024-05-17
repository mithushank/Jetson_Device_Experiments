import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18,vgg19,resnet50
from avalanche.benchmarks.classic import SplitFMNIST, SplitCIFAR10,SplitCIFAR100, PermutedMNIST, SplitCUB200
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics,forgetting_bwt
from avalanche.logging import InteractiveLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ICaRL
from avalanche.models import icarl_resnet,IcarlNet, NCMClassifier
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario, classification_scenario, online_scenario, NCExperience
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from torchvision import models, transforms
import math
import time
import os
import csv
from tegrastats_module import *

# Define classifier model
class Classifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x):
        return self.fc(x)

transform = transforms.Compose([
    transforms.Resize(224),                # Resize to 224x224
    transforms.RandomCrop(224, padding=8), # Random crop with padding
    transforms.ToTensor(),                  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
])

# start tracking using tegrastats
interval = 500 # read every 500 ms
tegrastats_process = start_tegrastats(interval=interval)

# Load SplitFMNIST dataset
benchmark = SplitCUB200(n_experiences=11,seed=1234, train_transform=transform, eval_transform = transform)

model = IcarlNet(num_classes=200)
# Define ResNet18-based feature extractor
# feature_extractor = FeatureExtractor(model_name='resnet50')
# feature_extractor_d = FeatureExtractor(model_name='resnet18')
# feature_extractor_dd = FeatureExtractor(model_name='vgg16')
# Define SGD optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Define loss criterion
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = [InteractiveLogger(),CSVLogger(log_folder = "icarl/cub200")]
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
        
strategy = ICaRL(
    feature_extractor=model,
    classifier=nn.Linear(in_features=200, out_features=200, bias=True),
    optimizer=optimizer,
    memory_size=1000,
    buffer_transform=None,
    fixed_memory=True,
    train_mb_size=50,
    plugins=[eval_plugin,ReplayP(mem_size=1000)]
)

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
'''
same situation as cifar10 but accuracy  is quite low. need to find solution for that 
'''
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