import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.io import loadmat
# Load the .mat file
data = loadmat(r'C:\Users\ZINO7M\Desktop\Respiratory modified\Nasal_264.mat')
signal = np.load(r'C:\Users\ZINO7M\Desktop\Respiratory modified\motion264.npy')
 
#data = loadmat(r'C:\Users\mun4sg\Documents\Python\MotionAlignment\Nasal_264.mat')
#signal = np.load(r'C:\Users\mun4sg\Documents\Python\MotionAlignment\motion264.npy')
flattened_signal = signal.flatten()
#
# Sample time in seconds
#sample_time = 4.5255 # Sample time in milliseconds for the full resolution FLORET
sample_time = 5.2 # Sample time for the mid resolution for FLORET
# sampling_time_s = sample_time / 1000  # convert to seconds
sampling_rate = 1 / sample_time  # in Hz
time_delay = 3 * sample_time # Sampling time is multiplied by 3 because you are skipping two points (i.e. 3 skipping 3 time intervals)
 
## Create time values based on the flattened signal length
#time_values_flattened_signal = np.arange(len(flattened_signal)) * (sample_time/1000)
# 
#for i in range(len(flattened_signal)):
#   if (i+1) % 44 == 0 :
#      sample_time = ((i+2)*sample_time)/i
#      #print(f'Time delays at i = {i}, sampling time = {sample_time:.4f} ms')
# To incoporate the time delay you'll need to build a time array and add the delay every 44 points.
time_array = []
current_time = 0
for i in range(len(flattened_signal)):
    time_array.append(current_time)
    # Every 44th sample, add the double interval
    if (i + 1) % 19 == 0:
        current_time += time_delay/1000
    else:
        current_time += sample_time/1000
 
# Convert to a numpy array
time_values_flattened_signal = np.array(time_array)      
# Access the 'dataCell' and 'timeCell' variables
data_cell = data.get('dataCell')
time_cell = data.get('timeCell')
# Extract the first array for nasal pressure and the corresponding time values
nasal_pressure_signal = data_cell[1, 0].flatten()  # Nasal pressure signal
time_values_nasal = time_cell[1, 0].flatten()  # Corresponding time values for nasal signal
print("Total number of points", len(nasal_pressure_signal))
# Extract the second array for esophageal signal and the corresponding time values
esophageal_signal = data_cell[0, 0].flatten()  # Esophageal signal
time_values_esophageal = time_cell[0, 0].flatten()  # Corresponding time values for esophageal signal
print("Total number of points", len(esophageal_signal))
# Plot all three signals on the same time axis
plt.figure(figsize=(12, 8))
# Plot the first signal (flattened_signal) with the converted time axis
plt.subplot(3, 1, 1)
plt.plot(time_values_flattened_signal, flattened_signal, label='Bellow Signal', color='green')
plt.xlim(0,400)
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.title('Bellow Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
# Plot the second signal (esophageal_signal) with its time values
plt.subplot(3, 1, 2)
plt.plot(time_values_esophageal, esophageal_signal, color='orange')
plt.xlim(0,400)
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.title('Esophageal Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
# Plot the third signal (nasal_pressure_signal) with its time values
plt.subplot(3, 1, 3)
plt.plot(time_values_nasal, nasal_pressure_signal, color='blue')
plt.xlim(0,400)
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.title('Nasal Pressure Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
# Show the plot with legends
plt.tight_layout()
plt.show()