"""A simple python script to monitor the memory usage of a main device.
Can be helpful for detecting 'memory leaks' - an increase amount of PyTorch
cached and reserved memory chunks that accumulate over time."""
import time
import os
import csv
import nvidia_smi

if __name__ == '__main__':
    file = os.path.join(os.getcwd(), 'gpu_sensor.csv')
    file_highest = os.path.join(os.getcwd(), 'gpu_sensor_highest.csv')
    with open(file, "w+", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'device', 'memory usage'])
    with open(file_highest, "w+", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'device', 'memory highest usage'])

    highest_mem_used = 0
    while True:
        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%H:%M:%S", named_tuple)
        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(f"|Device {i}| Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB | gpu-util: {util.gpu:3.1%} | gpu-mem: {util.memory:3.1%} |")
            mem_used = (mem.total / 1024**2) - (mem.free/1024**2)
            if mem_used > highest_mem_used:
                highest_mem_used = mem_used
                with open(file_highest, "a+", newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([time_string, i, highest_mem_used])
                print(f"New highest value of gpu used recorded at {time_string}: {highest_mem_used} MB")
            with open(file, "a+", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([time_string, i, mem_used])
        time.sleep(2)

        