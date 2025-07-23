import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_wave_iterations(u_, model, iterations=None, interval=10, save_animation=False, filename='wave_propagation.gif'):
    """
    Plot several iterations of a wave field.
    
    Parameters:
    -----------
    u_ : devito.Function
        The wave field to visualize
    model : Model
        The model used in the simulation
    iterations : list or None
        List of specific iterations to plot. If None, iterations will be selected automatically
    interval : int
        If iterations is None, plot every 'interval' iterations
    save_animation : bool
        Whether to save the animation as a GIF
    filename : str
        Filename for the saved animation
    """
    # Get the shape of the wave field
    shape = u_.shape
    print(f"Wave field shape: {shape}")
    
    # If iterations not specified, create a list of iterations to plot
    if iterations is None:
        if len(shape) == 3:  # Time dimension included
            total_iterations = shape[0]
            iterations = list(range(0, total_iterations, interval))
        else:
            raise ValueError("Cannot determine iterations automatically. Please provide iterations list.")
    
    # Create a figure for static plots
    n_plots = len(iterations)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each iteration
    for i, iter_idx in enumerate(iterations):
        if i < len(axes):
            if len(shape) == 3:  # Time dimension included
                data = u_.data[iter_idx]
            else:
                raise ValueError("Expected wave field with time dimension")
            
            im = axes[i].imshow(data.T, cmap='seismic', 
                               vmin=-np.max(np.abs(data))/2, 
                               vmax=np.max(np.abs(data))/2,
                               extent=[model.origin[0], model.origin[0] + model.domain_size[0],
                                      model.origin[1], model.origin[1] + model.domain_size[1]])
            axes[i].set_title(f'Iteration {iter_idx}')
            axes[i].set_xlabel('x (m)')
            axes[i].set_ylabel('z (m)')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create animation if requested
    if save_animation:
        fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
        
        def update(frame):
            ax_anim.clear()
            if len(shape) == 3:
                data = u_.data[frame]
            else:
                raise ValueError("Expected wave field with time dimension")
            
            im = ax_anim.imshow(data.T, cmap='seismic', 
                               vmin=-np.max(np.abs(u_.data))/2, 
                               vmax=np.max(np.abs(u_.data))/2,
                               extent=[model.origin[0], model.origin[0] + model.domain_size[0],
                                      model.origin[1], model.origin[1] + model.domain_size[1]])
            ax_anim.set_title(f'Iteration {frame}')
            ax_anim.set_xlabel('x (m)')
            ax_anim.set_ylabel('z (m)')
            return [im]
        
        anim = FuncAnimation(fig_anim, update, frames=iterations, interval=200, blit=True)
        
        if save_animation:
            anim.save(filename, writer='pillow', fps=5)
            print(f"Animation saved as {filename}")
        
        plt.show()

# Example usage:
# plot_wave_iterations(u_, model, iterations=[0, 10, 20, 30, 40, 50])
# plot_wave_iterations(u_, model, interval=20, save_animation=True)