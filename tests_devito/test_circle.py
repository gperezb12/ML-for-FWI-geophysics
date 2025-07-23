from examples.seismic import demo_model, plot_velocity, plot_perturbation
from examples.seismic import AcquisitionGeometry
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define true and initial model 
shape = (101, 101)  # Number of grid points (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations

model = demo_model('circle-isotropic', vp_circle=5.0, vp_background=2.5,
                    origin=origin, shape=shape, spacing=spacing, nbl=40, radio_custom=10)
plot_velocity(model, name="circleSim/velocityModel.png")

# Define source and receivers
t0 = 0.
tn = 1000. 
f0 = 0.01 #10Hz

src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, 0] = 20.  # Depth is 20m
 
nreceivers = 101  # Number of receiver locations per shot 
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 1] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 0] = 0.


# Geometry
geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')


from examples.seismic.acoustic import AcousticWaveSolver

solver = AcousticWaveSolver(model, geometry, space_order=4)
true_d, _, _ = solver.forward(vp=model.vp)

from examples.seismic import plot_shotrecord
plot_shotrecord(true_d.data, model, t0, tn, name="circleSim/simulation.png")

# Export and plot receiver amplitudes
time_axis = np.linspace(t0, tn, true_d.shape[0])
receiver_data = pd.DataFrame(true_d.data, columns=[f'Receiver {i}' for i in range(nreceivers)])
receiver_data.insert(0, 'Time (ms)', time_axis)

# Agregar la posición de los receptores
receiver_positions = pd.DataFrame({'Receiver': [f'Receiver {i}' for i in range(nreceivers)],
                                   'Position X': rec_coordinates[:, 0],
                                   'Position Z': rec_coordinates[:, 1]})

# Guardar los datos en CSV
receiver_data.to_csv("circleSim/reciever_signals.csv", index=False)
receiver_positions.to_csv("circleSim/reciever_positions.csv", index=False)

plt.figure(figsize=(10, 6))
for i in range(nreceivers):
    if i % 10 == 0:
        plt.plot(time_axis, true_d.data[:, i], alpha=0.5, label=f'Receiver {i}' if i % 20 == 0 else "")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("Receiver Signals Over Time")
plt.legend()
plt.grid()
plt.savefig("circleSim/receiver_signals.png")

# Obtener el linspace del pulso de Ricker
ricker_time = np.linspace(t0, tn, geometry.src.data.shape[0])  # Eje de tiempo del pulso
ricker_wavelet = geometry.src.data[:, 0]  # Valores de amplitud del pulso de Ricker

print(ricker_wavelet)

# Guardar en un CSV
ricker_data = pd.DataFrame({'Time (ms)': ricker_time, 'Amplitude': ricker_wavelet})
ricker_data.to_csv("circleSim/ricker_wavelet.csv", index=False)

# Graficar el pulso de Ricker
plt.figure(figsize=(8, 5))
plt.plot(ricker_time, ricker_wavelet, label="Ricker Wavelet", color='r')
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("Ricker Wavelet")
plt.legend()
plt.grid()
plt.savefig("circleSim/ricker_wavelet.png")
