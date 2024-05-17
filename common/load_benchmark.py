from avalanche.benchmarks.classic import CORe50,SplitCIFAR10,SplitCIFAR100, PermutedMNIST
from torchvision import transforms

""""
Here I choose datasets from  the avalanche and tested with AGEM strategy
Mostly working for that otherwise need top modify it
"""
def load_cifar10(strategy_name,num_experience,num_classes=10):
    strategy_name = strategy_name.lower()
    if strategy_name == "agem":
        benchmark = SplitCIFAR10(n_experiences=num_experience,seed=1234,train_transform=transforms.ToTensor(),eval_transform=transforms.ToTensor())
    elif strategy_name == "icarl":
        benchmark = SplitCIFAR10(n_experiences=num_experience,seed=1234,train_transform=transforms.ToTensor(),eval_transform=transforms.ToTensor())
    return benchmark

def load_cifar100(strategy_name,num_experience,num_classes=100):
    strategy_name = strategy_name.lower()
    if strategy_name == "agem":
        benchmark = SplitCIFAR100(n_experiences=num_experience,seed=1234,train_transform=transforms.ToTensor(),eval_transform=transforms.ToTensor())
    elif strategy_name == "icarl":
        benchmark = SplitCIFAR100(n_experiences=num_experience,seed=1234,train_transform=transforms.ToTensor(),eval_transform=transforms.ToTensor())
    return benchmark

def load_core50(strategy_name,num_experience,num_classes=50):
    benchmark = CORe50(run=1,mini=True, scenario = 'nc',object_lvl=False)
    return benchmark

def load_pmnist(strategy_name,num_experience,num_classes=10):
    if strategy_name == "agem":
        transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
        benchmark = PermutedMNIST(n_experiences=num_experience,seed=1234, train_transform=transform, eval_transform = transform)   
    elif strategy_name == "icarl":
        benchmark = PermutedMNIST(n_experiences = num_experience, seed=1234)
    return benchmark


