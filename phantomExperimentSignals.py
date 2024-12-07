import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Load the .mat file more efficiently by only loading the data we need
try:
    # Load data with squeeze_me=True to remove unnecessary dimensions
    mat_data = loadmat('bellows_data.mat', squeeze_me=True)
    volume_data = loadmat('volume_data.mat', squeeze_me=True)
    
    # Create figure before data processing
    plt.figure(figsize=(10, 5))
    
    # Convert to numpy array
    bellow_data = mat_data['bellow']
    
    # Timing parameters for bellows data
    sample_time_bellows = 4.5195  # Sample time in milliseconds for mid resolution FLORET
    sampling_rate_bellows = 1 / sample_time_bellows  # in Hz
    time_delay = 3 * sample_time_bellows
    
    # Create time array for bellows data with special timing
    time_array = []
    current_time = 0
    for i in range(len(bellow_data)):
        time_array.append(current_time)
        if (i + 1) % 19 == 0:
            current_time += time_delay / 1000
        else:
            current_time += sample_time_bellows / 1000
    
    time_values = np.array(time_array)
    
    # Use the full volume data
    volume_data = volume_data['volume_data']
    
    # Set volume data sampling rate to 40000 Hz
    sample_time_volume = 1 / 40000 * 1000  # Convert to milliseconds
    volume_time = np.arange(len(volume_data)) * (sample_time_volume / 1000)  # Convert to seconds
    
    # Detrend volume data
    slope, intercept = np.polyfit(volume_time, volume_data, 1)
    trend_line = slope * volume_time + intercept
    detrended_volume = volume_data - trend_line
    
    # Plot the full bellows data
    plt.subplot(2, 1, 1)
    plt.plot(time_values, bellow_data, label='Signal Data')
    plt.xlim(0, 400)
    plt.title('Signal Plot')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, linewidth=1.5)
    
    # Set more granular x-ticks for the first subplot
    #plt.xticks(np.arange(0, 26, 1))  # Adjust the step as needed for granularity
    
    plt.subplot(2, 1, 2)
    # Plot volume data with time in seconds
    plt.plot(volume_time, detrended_volume, label='Detrended Volume')
    plt.xlim(0, 400)
    plt.title('Volume Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, linewidth=1.5)
    
    # Set more granular x-ticks for the second subplot
    #plt.xticks(np.arange(0, 26, 1))  # Adjust the step as needed for granularity
    
    plt.show()

except Exception as e:
    print(f"Error loading or plotting data: {e}")

