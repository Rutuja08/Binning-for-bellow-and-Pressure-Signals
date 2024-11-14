import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.utils import resample

# Load and flatten the signal
signal = np.load('motionTest.npy')
flattened_signal = signal.flatten()

# Apply median filter to smooth the signal
window_size = 20  # Choose an appropriate window size
smoothed_signal = np.convolve(flattened_signal, np.ones(window_size) / window_size, mode='same')
print("length of the signal", len(smoothed_signal))
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

# Define the number of bins and colors
num_bins = 6  # Increased number of bins
colors = plt.cm.tab20(np.linspace(0, 1, num_bins * 2))
target_bin_size = 10000  # Target of 10,000 points per bin

plt.figure(figsize=(14, 12))

# Plot original signal
plt.subplot(4, 1, 1)
plt.plot(flattened_signal, color='blue')
plt.xlim(0, 4000)
plt.title('Original Signal')
plt.ylabel('Amplitude')

# Plot smoothed signal
plt.subplot(4, 1, 2)
plt.plot(smoothed_signal, color='orange')
plt.xlim(0, 4000)
plt.title('Smoothed Signal')
plt.ylabel('Amplitude')

# Prepare counters for bin indices (not amplitudes)
bin_indices = {f'Exhalation Bin {i}': [] for i in range(1, num_bins - 1)}
bin_indices.update({f'Inhalation Bin {i}': [] for i in range(1, num_bins - 1)})

# Plot each cycle with binned data for inhalation and exhalation
plt.subplot(4, 1, 3)
combined_max_bin_indices = []
combined_0th_bin_indices = []

for i in range(len(peaks) - 1):
    cycle_start = peaks[i]
    cycle_end = peaks[i + 1] if i < len(peaks) - 1 else len(smoothed_signal)
    
    if cycle_end <= cycle_start:
        continue
    
    cycle_segment = smoothed_signal[cycle_start:cycle_end]
    exhalation_segment = cycle_segment[:len(cycle_segment) // 2]
    inhalation_segment = cycle_segment[len(cycle_segment) // 2:]

    combined_segment = np.concatenate((exhalation_segment, inhalation_segment))
    bin_edges = np.linspace(np.min(combined_segment), np.max(combined_segment), num_bins + 1)

    # Digitize both segments
    exhalation_bin_indices_array = np.digitize(exhalation_segment, bin_edges) - 1
    inhalation_bin_indices_array = np.digitize(inhalation_segment, bin_edges) - 1

    # Store positions (indices) in each bin
    for j in range(1, num_bins - 1):  # Exclude 0 and (num_bins - 1)
        bin_indices[f'Exhalation Bin {j}'].extend(np.where(exhalation_bin_indices_array == j)[0] + cycle_start)
        bin_indices[f'Inhalation Bin {j}'].extend(np.where(inhalation_bin_indices_array == j)[0] + cycle_start + len(exhalation_segment))

    # Extract 0th and max bin indices for exhalation and inhalation
    combined_0th_bin_indices.extend(np.where(exhalation_bin_indices_array == 0)[0] + cycle_start)
    combined_0th_bin_indices.extend(np.where(inhalation_bin_indices_array == 0)[0] + cycle_start + len(exhalation_segment))
    combined_max_bin_indices.extend(np.where(exhalation_bin_indices_array == (num_bins - 1))[0] + cycle_start)
    combined_max_bin_indices.extend(np.where(inhalation_bin_indices_array == (num_bins - 1))[0] + cycle_start + len(exhalation_segment))

    # Plot binned data for each cycle, excluding 0th and max bins
    for j in range(1, num_bins - 1):
        plt.plot(np.arange(len(exhalation_segment))[exhalation_bin_indices_array == j] + cycle_start,
                 exhalation_segment[exhalation_bin_indices_array == j], '.', color=colors[j - 1])
        plt.plot(np.arange(len(inhalation_segment))[inhalation_bin_indices_array == j] + cycle_start + len(exhalation_segment),
                 inhalation_segment[inhalation_bin_indices_array == j], '.', color=colors[num_bins + j - 1])

# Limit 0th and max bin sizes to 10,000 points each
if len(combined_0th_bin_indices) > target_bin_size:
    excess_0th_bin_points = combined_0th_bin_indices[target_bin_size:]
    combined_0th_bin_indices = combined_0th_bin_indices[:target_bin_size]
else:
    excess_0th_bin_points = []

if len(combined_max_bin_indices) > target_bin_size:
    excess_max_bin_points = combined_max_bin_indices[target_bin_size:]
    combined_max_bin_indices = combined_max_bin_indices[:target_bin_size]
else:
    excess_max_bin_points = []

# Redistribute excess points to center bins
num_center_bins = num_bins - 2
total_excess_points = len(excess_0th_bin_points) + len(excess_max_bin_points)
points_per_center_bin, remaining_points = divmod(total_excess_points, num_center_bins)

for j in range(1, num_bins - 1):
    # Assign points from 0th and max bins to center bins
    start_idx = (j - 1) * points_per_center_bin + min(j - 1, remaining_points)
    end_idx = j * points_per_center_bin + min(j, remaining_points)
    
    bin_indices[f'Exhalation Bin {j}'].extend(excess_0th_bin_points[start_idx:end_idx])
    bin_indices[f'Inhalation Bin {j}'].extend(excess_max_bin_points[start_idx:end_idx])

# Resampling to ensure exactly 10,000 points per bin for center bins
for key in bin_indices:
    if len(bin_indices[key]) > 0:
        bin_indices[key] = resample(bin_indices[key], n_samples=target_bin_size, random_state=0)

# Plot combined 0th and max bin elements using scatter for consistency
plt.scatter(combined_0th_bin_indices, smoothed_signal[combined_0th_bin_indices], color='black', s=8)
plt.scatter(combined_max_bin_indices, smoothed_signal[combined_max_bin_indices], color='green', linewidth=3, s=2)
plt.xlim(0, 4000)
plt.title('Binned Signal by Position')
plt.ylabel('Amplitude')

# Prepare data for the box plot
plot_data = []
tick_labels = []

# Add combined bins (0th and max) to plot data
plot_data.append(combined_0th_bin_indices)
tick_labels.append('Combined 0th Bin')

# Add individual bins for exhalation and inhalation
for j in range(1, num_bins - 1):  # Exclude 0 and (num_bins - 1)
    exhalation_bin_elements = bin_indices[f'Exhalation Bin {j}']
    plot_data.append(exhalation_bin_elements)
    tick_labels.append(f'Ex Bin {j}')
    
    inhalation_bin_elements = bin_indices[f'Inhalation Bin {j}']
    plot_data.append(inhalation_bin_elements)
    tick_labels.append(f'In Bin {j}')

# Add combined max bin to plot data
plot_data.append(combined_max_bin_indices)
tick_labels.append('Combined Max Bin')

# Create the box plot
plt.subplot(4, 1, 4)
bp = plt.boxplot(plot_data, patch_artist=True)

# Customize colors if needed
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Print number of points per bin and display above each box in box plot
for i, data in enumerate(plot_data, start=1):
    count = len(data)
    print(f"Bin {i}: {count} points")
    plt.text(i, max(data) if len(data) > 0 else 0, f'{count}', horizontalalignment='center')

plt.xticks(range(1, len(tick_labels) + 1), tick_labels, rotation=45)
plt.title('Box Plot of Binned Indices (with Point Counts)')
plt.ylabel('Index Positions')

plt.tight_layout()
plt.show()

# Create a final array to store all indices with their corresponding bin labels
final_binned_indices = {
    'Combined_0th_Bin': np.array(combined_0th_bin_indices),
    'Combined_Max_Bin': np.array(combined_max_bin_indices)
}

# Add indices from each exhalation and inhalation bin
for j in range(1, num_bins - 1):  # Exclude 0 and (num_bins - 1)
    final_binned_indices[f'Exhalation_Bin_{j}'] = np.array(bin_indices[f'Exhalation Bin {j}'])
    final_binned_indices[f'Inhalation_Bin_{j}'] = np.array(bin_indices[f'Inhalation Bin {j}'])

# Optionally, you can convert the dictionary to a structured array if needed
# final_binned_array = np.array(list(final_binned_indices.items()), dtype=[('bin', 'U30'), ('indices', 'O')])

# Save the final binned indices to an NPY file
np.save('final_binned_indices.npy', final_binned_indices)

# Print final binned indices
print("Final Binned Indices:")
for bin_label, indices in final_binned_indices.items():
    print(f"{bin_label}: {len(indices)} points")
