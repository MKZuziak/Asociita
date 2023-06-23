"""A simple python script to monitor the memory usage of a main device.
Can be helpful for detecting 'memory leaks' - an increase amount of PyTorch
cached and reserved memory chunks that accumulate over time."""
import time
import os
import csv
import torch

if __name__ == '__main__':
    file = os.path.join(os.getcwd(), 'gpu_sensor.csv')
    with open(file, "w+", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'memory allocated', 'memory cached'])

    while True:
        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%H:%M:%S", named_tuple)
        device_name = torch.cuda.get_device_name(0)
        mem_aloc = round(torch.cuda.memory_allocated(0)/1024**3,4)
        mem_cached = round(torch.cuda.memory_reserved(0)/1024**3,4)
        with open(file, "a+", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([time_string, mem_aloc, mem_cached])
            print(device_name)
        print('Memory Usage:')
        print('Allocated:', mem_aloc, 'GB')
        print('Cached:   ', mem_cached, 'GB')
        time.sleep(2)

        