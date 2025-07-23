import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.fftpack import fft2, fftshift

def plot_wavefield_fft(wavefield, model, title="Wavefield in Frequency Domain"):
    """
    Compute and plot the 2D FFT of a wavefield.
    
    Parameters:
    -----------
    wavefield : numpy.ndarray
        The wavefield data in time domain
    model : Model
        The devito model object
    title : str
        Title for the plot
    """
    # Get the shape of the wavefield
    if len(wavefield.shape) == 3:  # (time, x, y)
        # Take a snapshot at the middle time
        snapshot_idx = wavefield.shape[0] // 2
        snapshot = wavefield[snapshot_idx, :, :]
    else:  # Assume it's already a snapshot
        snapshot = wavefield
    
    # Compute the 2D FFT
    fft_result = fft2(snapshot)
    fft_shifted = fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    
    # Create the frequency axes
    nx, nz = snapshot.shape
    dx, dz = model.spacing
    
    # Frequency axes in cycles per meter
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    kz = np.fft.fftshift(np.fft.fftfreq(nz, dz))
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Use log scale for better visualization
    plt.imshow(magnitude, cmap='viridis', 
               norm=LogNorm(vmin=magnitude.max()/1000, vmax=magnitude.max()),
               extent=[kz.min(), kz.max(), kx.min(), kx.max()])
    
    plt.colorbar(label='Magnitude (log scale)')
    plt.title(title)
    plt.xlabel('Wavenumber kz (cycles/m)')
    plt.ylabel('Wavenumber kx (cycles/m)')
    plt.tight_layout()
    
    return plt.gcf()

def plot_wavefield_fft_snapshots(wavefield, model, time_indices=None, title_prefix="Wavefield FFT at time"):
    """
    Compute and plot the 2D FFT of a wavefield at multiple time snapshots.
    
    Parameters:
    -----------
    wavefield : numpy.ndarray
        The wavefield data in time domain with shape (time, x, y)
    model : Model
        The devito model object
    time_indices : list
        List of time indices to plot. If None, will choose evenly spaced indices.
    title_prefix : str
        Prefix for the plot titles
    """
    if len(wavefield.shape) != 3:
        raise ValueError("Expected 3D wavefield with shape (time, x, y)")
    
    nt = wavefield.shape[0]
    
    if time_indices is None:
        # Choose 4 evenly spaced time indices
        time_indices = [nt//5, 2*nt//5, 3*nt//5, 4*nt//5]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for i, t_idx in enumerate(time_indices[:4]):  # Limit to 4 plots
        snapshot = wavefield[t_idx, :, :]
        
        # Compute the 2D FFT
        fft_result = fft2(snapshot)
        fft_shifted = fftshift(fft_result)
        magnitude = np.abs(fft_shifted)
        
        # Create the frequency axes
        nx, nz = snapshot.shape
        dx, dz = model.spacing
        
        # Frequency axes in cycles per meter
        kx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
        kz = np.fft.fftshift(np.fft.fftfreq(nz, dz))
        
        # Plot on the corresponding subplot
        im = axes[i].imshow(magnitude, cmap='viridis', 
                      norm=LogNorm(vmin=magnitude.max()/1000, vmax=magnitude.max()),
                      extent=[kz.min(), kz.max(), kx.min(), kx.max()])
        
        axes[i].set_title(f"{title_prefix} {t_idx}")
        axes[i].set_xlabel('Wavenumber kz (cycles/m)')
        axes[i].set_ylabel('Wavenumber kx (cycles/m)')
        
        plt.colorbar(im, ax=axes[i], label='Magnitude (log scale)')
    
    plt.tight_layout()
    return fig