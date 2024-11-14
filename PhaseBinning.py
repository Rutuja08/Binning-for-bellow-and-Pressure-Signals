import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
import random

# Load and flatten the signal
signal = np.load(r'C:\Users\ZINO7M\Desktop\Respiratory modified\motionTest.npy')
flattened_signal = signal.flatten()

# Apply median filter to smooth the signal
window_size = 20  # Choose an appropriate window size
smoothed_signal = np.convolve(flattened_signal, np.ones(window_size) / window_size, mode='same')

# Compute the analytic signal and phase angle
analytic_signal = hilbert(smoothed_signal)
phase_angle = np.angle(analytic_signal)

# Find peaks and troughs
peaks, _ = find_peaks(smoothed_signal)
troughs, _ = find_peaks(-smoothed_signal)

# Ensure peaks and troughs are properly ordered
if len(peaks) > 0 and len(troughs) > 0:
    if peaks[0] > troughs[0]:
        peaks = peaks[1:]
    if len(peaks) > len(troughs):
        peaks = peaks[:-1]
    elif len(troughs) > len(peaks):
        troughs = troughs[:-1]

# Define the number of phase bins and colors
num_bins = 10
colors = plt.cm.tab20(np.linspace(0, 1, num_bins))  # Use the color map for bins
basic_colors = colors[:num_bins]  # Use the first set of colors for bins

# Prepare counters for phase bin positions
bin_positions = {f'Bin {i}': [] for i in range(num_bins)}

# Plot original signal
plt.figure(figsize=(14, 15))
plt.subplot(4, 1, 1)
plt.plot(flattened_signal, color='blue')
plt.xlim(0, 4000)
plt.title('Original Signal')
plt.ylabel('Amplitude')
plt.legend()

# Plot smoothed signal
plt.subplot(4, 1, 2)
plt.plot(smoothed_signal, color='orange')
plt.xlim(0, 4000)
plt.title('Smoothed Signal')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 3)
for i in range(len(peaks) - 1):
    cycle_start = peaks[i]
    cycle_end = peaks[i + 1] if i < len(peaks) - 1 else len(smoothed_signal)
    if cycle_end <= cycle_start:
        continue
    cycle_segment = smoothed_signal[cycle_start:cycle_end]
    phase_segment = phase_angle[cycle_start:cycle_end]
    
    # Define phase bin edges
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    
    # Digitize phase segment
    bin_indices = np.digitize(phase_segment, bin_edges) - 1
    
    # Collect position (index) for each bin
    for j in range(num_bins):
        bin_positions[f'Bin {j}'].extend(np.where(bin_indices == j)[0] + cycle_start)  # Saving positions
    
    # Plot binned data for each cycle
    for j in range(num_bins):
        plt.plot(np.arange(len(cycle_segment))[bin_indices == j] + cycle_start,
                 cycle_segment[bin_indices == j], '.', color=basic_colors[j])

plt.xlim(0, 4000)
plt.title('Phase Binned Signal')
plt.ylabel('Amplitude')
plt.legend(loc='best', fontsize='small')

# Box plot for the number of points per bin
plt.subplot(4, 1, 4)
data_for_boxplot = [bin_positions[f'Bin {i}'] for i in range(num_bins)]
boxplot = plt.boxplot(data_for_boxplot, labels=[f'Bin {i}' for i in range(num_bins)], patch_artist=True)

# Set colors for the box plot
for i, patch in enumerate(boxplot['boxes']):
    patch.set_facecolor(basic_colors[i])  # Set the face color for the box
    patch.set_edgecolor('black')  # Optional: Set the edge color for the box

# Annotate the number of points per bin on the box plot
for i, data in enumerate(data_for_boxplot):
    count = len(data)
    # Display the count above the max value in the bin
    plt.text(i + 1, np.max(data) if len(data) > 0 else 0, f'n={count}', ha='center', va='bottom', fontsize=8, color= 'black')

plt.title('Distribution of Positions per Bin')
plt.xlabel('Bins')
plt.ylabel('Positions')
plt.xticks(rotation=90)
plt.grid(True)

plt.tight_layout()
plt.show()

# Save the bin positions (indices) instead of amplitudes
np.save('binned_positions.npy', bin_positions)

desired_bin_size = 10000

# Load the bin positions from your saved file
loaded_bin_positions = np.load('binned_positions.npy', allow_pickle=True).item()

# Adjust bins to have exactly 10,000 elements
adjusted_bin_positions = {}

for bin_name, positions in loaded_bin_positions.items():
    current_size = len(positions)
    
    if current_size > desired_bin_size:
        # Downsample if there are more than 10,000 elements
        adjusted_bin_positions[bin_name] = random.sample(positions, desired_bin_size)
    elif current_size < desired_bin_size:
        # Upsample by repeating some elements if there are fewer than 10,000 elements
        adjusted_bin_positions[bin_name] = random.choices(positions, k=desired_bin_size)
    else:
        # If already 10,000, just keep the bin as is
        adjusted_bin_positions[bin_name] = positions

# Optionally save the adjusted bins if needed
np.save('adjusted_binned_positions.npy', adjusted_bin_positions)

# Check the size of Bin 0 after adjustment
#print(f"Size of Bin 0 after adjustment: {len(adjusted_bin_positions['Bin 0'])}")
