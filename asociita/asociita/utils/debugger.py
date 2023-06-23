import torch
import os
import csv

def log_gpu_memory(iteration = int,
                   path: str = None):
    """Debugger function that prints out the device that we train on,
    amout of GPU memory that is currently allocated, and the amoung of 
    GPU memory that is currently cached by CUDA. Save result to a log.
    If not path is provided, it will save the log in the cwd.
    Parameters
    ----------
    nodes_data: list[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]: 
        A list containing train set and test set wrapped 
        in a hugging face arrow_dataset.Dataset containers.
    
    Returns
    -------
    int
        Returns 0 on the successful completion of the training.
    """
    device_name = torch.cuda.get_device_name(0)
    mem_aloc = round(torch.cuda.memory_allocated(0)/1024**3,4)
    mem_cached = round(torch.cuda.memory_reserved(0)/1024**3,4)
    
    print(device_name)
    print('Memory Usage:')
    print('Allocated:', mem_aloc, 'GB')
    print('Cached:   ', mem_cached, 'GB')

    if not path:
        path = os.getcwd()
    with open(os.path.join(path, 'memory_usage_logs.csv'), '+a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if iteration == 0:
            writer.writerow(['Iteration', 'Memory Allocated', 'Memory Cached'])
        writer.writerow([iteration, mem_aloc, mem_cached])
        
