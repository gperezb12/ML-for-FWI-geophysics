import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

def export_magnitude_freq_domain(u_, z_range=None, output_file='magnitude_freq_domain.csv'):
    """
    Export the magnitude in the frequency domain as CSV.
    
    Parameters:
    -----------
    u_ : devito.Function
        The wave field data
    z_range : tuple or None
        Range in z-direction to sample (min, max). If None, use full range.
    output_file : str
        Output CSV filename
    
    Returns:
    --------
    None
    """
    # Get the shape of the wave field
    shape = u_.shape
    
    # Take the middle time step for analysis
    middle_time = shape[0] // 2
    wave_data = u_.data[middle_time]
    
    # Apply z_range if specified
    if z_range is not None:
        z_min, z_max = z_range
        # Convert to indices
        z_min_idx = int(z_min / 10)  # Assuming 10m grid spacing
        z_max_idx = int(z_max / 10)
        wave_data = wave_data[:, z_min_idx:z_max_idx]
    
    # Compute FFT and shift to center
    fft_data = fftshift(fft2(wave_data))
    magnitude = np.abs(fft_data)
    
    # Create coordinate grids
    x_coords = np.arange(magnitude.shape[0])
    z_coords = np.arange(magnitude.shape[1])
    
    # Create output data
    output_data = []
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            output_data.append([x_coords[i], z_coords[j], magnitude[i, j]])
    
    # Save to CSV
    np.savetxt(output_file, output_data, delimiter=',', header='x,z,magnitude', comments='')
    print(f"Magnitude data exported to {output_file}")

def export_phase_freq_domain(u_, z_range=None, output_file='phase_freq_domain.csv'):
    """
    Export the phase in the frequency domain as CSV.
    
    Parameters:
    -----------
    u_ : devito.Function
        The wave field data
    z_range : tuple or None
        Range in z-direction to sample (min, max). If None, use full range.
    output_file : str
        Output CSV filename
    
    Returns:
    --------
    None
    """
    # Get the shape of the wave field
    shape = u_.shape
    
    # Take the middle time step for analysis
    middle_time = shape[0] // 2
    wave_data = u_.data[middle_time]
    
    # Apply z_range if specified
    if z_range is not None:
        z_min, z_max = z_range
        # Convert to indices
        z_min_idx = int(z_min / 10)  # Assuming 10m grid spacing
        z_max_idx = int(z_max / 10)
        wave_data = wave_data[:, z_min_idx:z_max_idx]
    
    # Compute FFT and shift to center
    fft_data = fftshift(fft2(wave_data))
    phase = np.angle(fft_data)
    
    # Create coordinate grids
    x_coords = np.arange(phase.shape[0])
    z_coords = np.arange(phase.shape[1])
    
    # Create output data
    output_data = []
    for i in range(phase.shape[0]):
        for j in range(phase.shape[1]):
            output_data.append([x_coords[i], z_coords[j], phase[i, j]])
    
    # Save to CSV
    np.savetxt(output_file, output_data, delimiter=',', header='x,z,phase', comments='')
    print(f"Phase data exported to {output_file}")

def export_model_speed(model, z_range=None, output_file='model_speed.csv'):
    """
    Export the model speed v as CSV.
    
    Parameters:
    -----------
    model : Model
        The model used in the simulation
    z_range : tuple or None
        Range in z-direction to sample (min, max). If None, use full range.
    output_file : str
        Output CSV filename
    
    Returns:
    --------
    None
    """
    # Get the velocity data
    v_data = model.vp.data
    
    # Apply z_range if specified
    if z_range is not None:
        z_min, z_max = z_range
        # Convert to indices
        z_min_idx = int(z_min / 10)  # Assuming 10m grid spacing
        z_max_idx = int(z_max / 10)
        v_data = v_data[:, z_min_idx:z_max_idx]
    
    # Create coordinate grids
    x_coords = np.arange(v_data.shape[0])
    z_coords = np.arange(v_data.shape[1])
    
    # Create output data
    output_data = []
    for i in range(v_data.shape[0]):
        for j in range(v_data.shape[1]):
            output_data.append([x_coords[i], z_coords[j], v_data[i, j]])
    
    # Save to CSV
    np.savetxt(output_file, output_data, delimiter=',', header='x,z,velocity', comments='')
    print(f"Model speed data exported to {output_file}")