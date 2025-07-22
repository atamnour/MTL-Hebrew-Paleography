import os
import numpy as np
import torch
# Print the current working directory
print("Current working directory:", os.getcwd())

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)