import subprocess
import time
import os

import psutil

def start_tegrastats(interval=100, output_file='tegrastats_output.txt'):
    """Starts tegrastats with a specified interval."""
    # return subprocess.Popen(['tegrastats', '--interval', str(interval)], stdout=open(output_file, 'w'))
    p =  subprocess.Popen(['sudo', 'tegrastats', '--interval', str(interval)], stdout=open(output_file, 'w'))
    return p.pid

def stop_tegrastats(tegrastats_process):
    """Stops the tegrastats process."""
    #tegrastats_process.terminate()
    try:
        #subprocess.run(['sudo', 'kill', '-9', str(tegrastats_process)])
        #os.system("sudo kill -9 " + str(tegrastats_process))
        os.system("sudo pkill tegrastats")
        print("Subprocess terminated successfully.")
    except (psutil.NoSuchProcess, PermissionError) as e:
        print(f"Termination failed: {e}")


def parse_tegrastats_output(file_path, interval=100):
    total_gpu_soc_power = 0
    total_cpu_cv_power = 0

    with open(file_path, 'r') as file:
        for line in file:
            if 'VDD_GPU_SOC' in line:
                try:
                    gpu_soc_power = line.split('VDD_GPU_SOC ')[1].split('mW')[0]
                    #total_gpu_soc_power += int(gpu_soc_power)
                    total_gpu_soc_power += float(gpu_soc_power) * (interval / 1000.0)
                except (IndexError, ValueError):
                    print("Error parsing GPU SOC power value.")
            if 'VDD_CPU_CV' in line:
                try:
                    cpu_cv_power = line.split('VDD_CPU_CV ')[1].split('mW')[0]
                    total_cpu_cv_power += float(cpu_cv_power) * (interval / 1000.0)
                    #total_cpu_cv_power += int(cpu_cv_power)
                except (IndexError, ValueError):
                    print("Error parsing CPU CV power value.")
    #total_gpu_soc_power = total_gpu_soc_power /1000 * interval
    #total_cpu_cv_power = total_cpu_cv_power /1000 * interval
    total_power = total_gpu_soc_power + total_cpu_cv_power

    return total_gpu_soc_power, total_cpu_cv_power, total_power