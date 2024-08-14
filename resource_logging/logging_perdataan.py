import subprocess as sp
import os
from threading import Thread , Timer
import sched, time
import psutil
from datetime import datetime
import pandas as pd
import clr # the pythonnet module.
clr.AddReference(r'OpenHardwareMonitorLib') 
# e.g. clr.AddReference(r'OpenHardwareMonitor/OpenHardwareMonitorLib'), without .dll
from OpenHardwareMonitor.Hardware import Computer

c = Computer()
c.CPUEnabled = True # get the Info about CPU
c.GPUEnabled = True # get the Info about GPU
c.Open()

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

def get_gpu_temp():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=temperature.gpu --format=csv"
    try:
        memory_temp_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_temp = [int(x.split()[0]) for i, x in enumerate(memory_temp_info)]
    return memory_temp


def get_cpu_percentage():
    return psutil.cpu_percent(interval=1)


def get_virtual_memory():
    return psutil.virtual_memory().percent, (psutil.virtual_memory().used/1000000)

def logger_every_5secs(df, csv_name):
    cpu_temp=None

    while True:
        for a in range(0, len(c.Hardware[0].Sensors)):
            # print(c.Hardware[0].Sensors[a].Identifier)
            if "/temperature" in str(c.Hardware[0].Sensors[a].Identifier):
                cpu_temp = c.Hardware[0].Sensors[a].get_Value()
                c.Hardware[0].Update()
        current_dateTime = datetime.now()
        print(current_dateTime)
        print("GPU Memory Used: "+ str(get_gpu_memory()[0]))
        print("GPU Temperature: "+ str(get_gpu_temp()[0]) +"°C")
        print("CPU Percentage Used: "+ str(get_cpu_percentage()))
        print("CPU Temperature: "+ str(cpu_temp) +"°C")
        vram_perc, vram_val = get_virtual_memory()
        print("RAM Percentage Used: "+ str(vram_perc))
        print("RAM Used: "+ str(vram_val)+"\n")
    

        new_row = {'Timestamp':current_dateTime, 
            'CPU Percentage':get_cpu_percentage(), 
            'CPU Temperature':cpu_temp, 
            'RAM Percentage':vram_perc, 
            'RAM Memory':vram_val, 
            'GPU Memory':get_gpu_memory()[0],
            'GPU Temperature': get_gpu_temp()[0]}
            
        df = pd.concat([df, pd.DataFrame.from_records([new_row])])
        df.to_csv((csv_name+'.csv'), index=False)
        # time.sleep(0)

csv_name = input("Insert the CSV Name: ")

df = pd.DataFrame(columns=['Timestamp', 'CPU Percentage','CPU Temperature', 'RAM Percentage', 'RAM Memory', 'GPU Memory', 'GPU Temperature'])
logger_every_5secs(df, csv_name)