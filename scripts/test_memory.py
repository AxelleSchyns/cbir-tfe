import time
import random
import torch
import numpy as np

print("Torch status:", torch.cuda.is_available())
print("Number of devices detected:",torch.cuda.device_count())
print("Current:", torch.cuda.current_device())

mat1 = torch.rand((25000, 50000), device= 'cuda:0')
mat2 = torch.rand((25000, 50000), device = "cuda:0")


rand_indexes_col  = []
rand_indexes_row = []
for i in range(20):
    rand_indexes_col.append(random.randint(0, 25000))
    rand_indexes_row.append(random.randint(0, 50000))
for i in range(20):
    print(mat1[rand_indexes_col[i], rand_indexes_row[i]].item(), end=" ")
print("")


mat_fin = mat1 * mat2
mat2 = mat2.to(device='cpu')

# setting device on GPU if available, else CPU
if torch.cuda.is_available():
    for device in range(torch.cuda.device_count()):
        print('Using device:', torch.cuda.get_device_name(device))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(device)/1024**3,1), 'GB')

try:
    while True:
        time.sleep(10)
        for i in range(20):
            print(mat1[rand_indexes_col[i], rand_indexes_row[i]].item(), end=" ")
        print("")
except KeyboardInterrupt:
    print('\n interrupted!')