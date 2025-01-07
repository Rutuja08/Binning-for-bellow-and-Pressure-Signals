import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from scipy.signal import correlate
import scipy.signal

# Load the .mat file and the .npy file
data = loadmat(r'C:\Users\ZINO7M\Desktop\Respiratory modified\Nasal_264.mat')
signal = np.load(r'C:\Users\ZINO7M\Desktop\Respiratory modified\motion264.npy')

flattened_signal = signal.flatten()

# Generate time values for the bellow signal with mid-resolution skipping logic
noProj = len(flattened_signal)  # Number of projections (length of the bellow signal)
TR = 5.1589  / 1000              # TR value in seconds
time_array = []                # Initialize empty list for time values

# Generate time array with skipping logic
i = 0
j = 0
while True:
    i += 1
    if (i % 20 == 0) or (i % 21 == 0):  # Skip 2 bits after every 19th sample
        continue
    else:
        j += 1
        time_array.append(i * TR)  # Append current time based on TR
        if j == noProj:  # Stop if we reach the required number of projections
            break

time_values_flattened_signal = np.array(time_array)

# Access the 'dataCell' and 'timeCell' variables
data_cell = data.get('dataCell')
time_cell = data.get('timeCell')

# Extract nasal pressure and esophageal signals
nasal_pressure_signal = data_cell[1, 0].flatten()  # Nasal pressure signal
time_values_nasal = time_cell[1, 0].flatten()      # Time values for nasal signal

esophageal_signal = data_cell[0, 0].flatten()      # Esophageal signal
time_values_esophageal = time_cell[0, 0].flatten()  # Time values for esophageal signal

# Align the esophageal signal with the flattened bellow signal
array1_col = time_values_flattened_signal.reshape(-1, 1)
array2_col = time_values_esophageal.reshape(-1, 1)

nbrs = NearestNeighbors(n_neighbors=1).fit(array2_col)
distances, indices = nbrs.kneighbors(array1_col)
indices = indices.flatten()

# Downsample the esophageal signal
esophageal_signal_downsampled = esophageal_signal[indices]

# Align the nasal pressure signal with the flattened bellow signal
array3_col = time_values_nasal.reshape(-1, 1)

nbrs_nasal = NearestNeighbors(n_neighbors=1).fit(array3_col)
distances_nasal, indices_nasal = nbrs_nasal.kneighbors(array1_col)
indices_nasal = indices_nasal.flatten()

# Downsample the nasal pressure signal
nasal_pressure_signal_downsampled = nasal_pressure_signal[indices_nasal]

# Normalize the signals
def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

normalized_flattened_signal = normalize(flattened_signal)
normalized_esophageal_signal = normalize(esophageal_signal_downsampled)
normalized_nasal_pressure_signal = normalize(nasal_pressure_signal_downsampled)

# Fine-tune alignment using cross-correlation
def fine_tune_alignment(reference_signal, target_signal):
    corr = correlate(reference_signal, target_signal, mode='full')
    lags = np.arange(-len(reference_signal) + 1, len(target_signal))
    optimal_lag = lags[np.argmax(corr)]
    fine_tuned_signal = np.roll(target_signal, optimal_lag)
    return fine_tuned_signal, optimal_lag

fine_tuned_esophageal_signal, lag_esophageal = fine_tune_alignment(
    normalized_flattened_signal, normalized_esophageal_signal
)
fine_tuned_nasal_pressure_signal, lag_nasal = fine_tune_alignment(
    normalized_flattened_signal, normalized_nasal_pressure_signal
)

# Plot individual signals
plt.figure(figsize=(12, 8))

# Plot the bellow signal
plt.subplot(3, 1, 1)
plt.plot(time_values_flattened_signal, flattened_signal, label='Bellow Signal', color='green')
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.title('Bellow Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# Plot the esophageal signal
plt.subplot(3, 1, 2)
time_values_esophageal_downsampled = time_values_esophageal[indices]
plt.plot(
    time_values_flattened_signal,
    fine_tuned_esophageal_signal,
    color='orange',
    label='Esophageal Signal (Fine-Tuned)',
)
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.title('Esophageal Signal (Fine-Tuned)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# Plot the nasal pressure signal
plt.subplot(3, 1, 3)
time_values_nasal_downsampled = time_values_nasal[indices_nasal]
plt.plot(
    time_values_flattened_signal,
    fine_tuned_nasal_pressure_signal,
    color='blue',
    label='Nasal Pressure Signal (Fine-Tuned)',
)
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.title('Nasal Pressure Signal (Fine-Tuned)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()



# Normalize all signals consistently
normalized_esophageal_signal_final = normalize(fine_tuned_esophageal_signal)
normalized_nasal_pressure_signal_final = normalize(fine_tuned_nasal_pressure_signal)

normalized_flattened_signal = normalize(normalized_flattened_signal)
# Determine the scaling factor based on the maximum values of each signal
scaling_factor = np.max(normalized_flattened_signal) / np.max(normalized_esophageal_signal_final)
scaling_factor *= 2
# Scale down the bellow signal by the scaling factor
scaled_bellow_signal = normalized_flattened_signal / scaling_factor


# Overlapping plot of all normalized signals
plt.figure(figsize=(12, 8))
plt.plot(
    time_values_flattened_signal,
    scaled_bellow_signal,
    label='Bellow Signal (Normalized)',
    color='green',
)
plt.plot(
    time_values_flattened_signal,
    normalized_esophageal_signal_final,
    label='Esophageal Signal (Normalized & Fine-Tuned)',
    color='orange',
)
# plt.plot(
#     time_values_flattened_signal,
#     normalized_nasal_pressure_signal_final,
#     label='Nasal Pressure Signal (Normalized & Fine-Tuned)',
#     color='blue',
# )

plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.title('Overlapping Signals (Normalized & Fine-Tuned)')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

def calculate_peak_to_peak_duration(signal):
    # Find peaks
    peaks, _ = scipy.signal.find_peaks(signal)
    
    # Calculate peak-to-peak durations
    peak_times = peaks / sample_rate  # Convert peak indices to time
    peak_durations = np.diff(peak_times)  # Calculate differences between consecutive peaks
    
    return peak_times, peak_durations

# Example signals (replace these with your actual signals)
motion_signal = np.random.rand(1000)  # Replace with actual motion signal
nasal_signal = np.random.rand(1000)   # Replace with actual nasal signal
esophageal_signal = np.random.rand(1000)  # Replace with actual esophageal signal

# Sample rate (adjust as necessary)
sample_rate = 40000

# Calculate for each signal in the range of indices 50 to 100
motion_signal_range = motion_signal[50:71]
nasal_signal_range = nasal_signal[50:71]
esophageal_signal_range = esophageal_signal[50:71]

motion_peak_times, motion_peak_durations = calculate_peak_to_peak_duration(motion_signal_range)
nasal_peak_times, nasal_peak_durations = calculate_peak_to_peak_duration(nasal_signal_range)
esophageal_peak_times, esophageal_peak_durations = calculate_peak_to_peak_duration(esophageal_signal_range)

# Print results
print("Motion Peak Times (50-100):", motion_peak_times)
print("Motion Peak Durations (50-100):", motion_peak_durations)

print("Nasal Peak Times (50-100):", nasal_peak_times)
print("Nasal Peak Durations (50-100):", nasal_peak_durations)

print("Esophageal Peak Times (50-100):", esophageal_peak_times)
print("Esophageal Peak Durations (50-100):", esophageal_peak_durations)
