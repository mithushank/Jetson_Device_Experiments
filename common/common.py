from load_benchmark import *
from set_strategy import *
from tegrastats_module import *
from avalanche import *

# load dataset
def _get_data(name,strategy_name,num_classes,num_experience):
    if name == "cifar10":
        benchmark = load_cifar10(strategy_name,num_experience)
    elif name == "cifar100":
        benchmark = load_cifar100(strategy_name,num_experience)
    elif name == "pmnist":
        benchmark = load_pmnist(strategy_name,num_experience)
    elif name == "core50":
        benchmark = load_core50(strategy_name,num_experience)
    else :
        "Not included in the function"
        return None

    return benchmark

def _get_strategy(strategy_name ,name):
    strategy_name = strategy_name.lower()
    if strategy_name == "agem":
        strategy = agem_strategy(name)
    elif strategy_name == "icarl":
        strategy = icarl_strategy(name)
    elif strategy_name == "gss_greedy":
        strategy = gss_greedy_strategy(name)
    elif strategy_name == "naive":
        strategy = naive_strategy(name)
    return strategy

def train_strategy(benchmark,strategy,coreset_percent=100):
    if coreset_percent == 100:
        for experience in benchmark.train_stream:
            strategy.train(experience)
            strategy.eval(benchmark.test_stream)
    else :
        for i,batch in enumerate(benchmark.train_stream):
            old_ds = benchmark.train_stream[i].dataset
            
            new_ds = old_ds.subset(range(0, len(old_ds),math.floor(100/coreset_percent)))  # select coreset randomly 
            print(len(old_ds),len(new_ds))
            stream = CLStream(name='train', exps_iter = new_ds, benchmark = benchmark)
            batch_new = NCExperience(stream,i)
            print(type(stream), type(batch_new))
            strategy.train(batch_new)    
            strategy.eval(benchmark.test_stream) 



def store_power_evaluations(name,strategy_name,num_classes,num_experience):
    interval = 500
    output_file = "tegrastats_"+strategy_name.lower()+"_"+name.lower()+".txt"
    tegrastats_process = start_tegrastats(interval = interval, output_file = output_file)

    benchmark = _get_data(name,strategy_name,num_classes,num_experience)
    strategy = _get_strategy(strategy_name ,name)
    print(type(strategy))
    #start time
    start = time.time()

    train_strategy(benchmark = benchmark, strategy = strategy)
    end = time.time()
    # end of time
    elapse = end - start

    stop_tegrastats(tegrastats_process)
    time.sleep(1)  # Wait a bit for tegrastats to terminate cleanly
    total_gpu_soc_power, total_cpu_cv_power, total_power = parse_tegrastats_output(output_file, interval)

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
    csv_file_name = "results_"+name.lower()+"_"+strategy_name.lower()+".csv"

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

def main():
    # Ask user for input
    name = str(input("Enter dataset name (e.g., cifar10, cifar100, pmnist, core50): "))
    strategy_name = str(input("Enter strategy name (e.g., agem, icarl, gss_greedy, naive): "))
    num_experience = int(input("Enter the number of experiences: "))
    num_classes = 10  # Assuming default value of 10, you can change this if necessary
    
    # Call functions with user input
    benchmark = _get_data(strategy_name, name, num_classes, num_experience)
    strategy = _get_strategy(strategy_name, name)
    store_power_evaluations(name, strategy_name, num_classes, num_experience)


main()

    