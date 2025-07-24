import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def generate_interior_data(filename, var, nx=None, nz=None, device='cpu'):
    """
    Read CSV data and return torch tensors.
    Array containing the u values reshaped to (nx, nz) if dimensions provided,
    otherwise returns the raw data as torch tensors
    """
    try:
        # Read the CSV file
        data = pd.read_csv(filename)
        
        # Verify the required columns exist
        required_cols = ['x', 'z', var]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Create coordinate tensor
        X_data_np = np.column_stack((data['x'].values, data['z'].values))
        X_data = torch.tensor(X_data_np, dtype=torch.float64, requires_grad=True, device=device)
        
        if nx is not None and nz is not None:
            # Get unique x and z values
            unique_x = np.sort(data['x'].unique())
            unique_z = np.sort(data['z'].unique())
            
            # Verify the dimensions match
            if len(unique_x) != nx or len(unique_z) != nz:
                raise ValueError(f"Data dimensions ({len(unique_x)}, {len(unique_z)}) "
                               f"don't match specified dimensions ({nx}, {nz})")
            
            # Reshape the data into a 2D grid and convert to torch tensor
            u_values = data[var].values.reshape((nx, nz))
            u_data = torch.tensor(u_values, dtype=torch.float64, device=device)
        else:
            # Return the raw data as torch tensor if dimensions aren't specified
            u_data = torch.tensor(data[var].values, dtype=torch.float32, device=device)
            
        return X_data, u_data
            
    except Exception as e:
        print(f"Error reading or processing data: {str(e)}")
        raise


def sample_interior_data3(X_data, u_data, num_samples=1000,
                         x_mean=500.0, x_std=150.0,  # increased std
                         z_decay_rate=100000,
                         device='cpu'):
    X = X_data.to(device)
    u = u_data.to(device)
    x_coords = X[:, 0:1]
    
    # Add checks
    x_weights = torch.exp(-((x_coords - x_mean)**2) / (2 * x_std**2))
    print(f"Min weight: {x_weights.min()}")
    print(f"Max weight: {x_weights.max()}")
    print(f"Sum of weights: {x_weights.sum()}")
    
    # Ensure weights are positive
    x_weights = torch.clamp(x_weights, min=1e-10)
    
    probs = x_weights / x_weights.sum()
    
    # Verify probabilities
    assert probs.sum() > 0, "Sum of probabilities must be positive"
    assert torch.all(probs >= 0), "All probabilities must be non-negative"
    
    idx = torch.multinomial(probs, num_samples, replacement=True)
    return X[idx], u[idx]
