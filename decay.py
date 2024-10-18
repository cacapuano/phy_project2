# %% Cell Monte Carlo simulation on tritium decay
import numpy as np
import matplotlib.pyplot as plt

# Constants
T_HALF = 12.32  # Half-life of tritium in years
LAMBDA = np.log(2) / T_HALF  # Decay constant

# Simulation parameters
initial_nuclei = 1000  # Initial number of tritium nuclei
time_end = 50  # Total time to simulate in years
num_trials = 1000  # Number of Monte Carlo trials

def monte_carlo_tritium_decay(initial_nuclei, time_end, num_trials):
    decay_times = []

    for _ in range(num_trials):
        N = initial_nuclei
        time = 0

        while N > 0 and time < time_end:
            # Random time until next decay
            time_until_decay = np.random.exponential(1 / LAMBDA)
            time += time_until_decay

            # Determine if decay occurs
            if np.random.rand() < 1 - np.exp(-LAMBDA * time_until_decay):
                N -= 1

        decay_times.append(time)

    return decay_times

# Run the simulation
decay_times = monte_carlo_tritium_decay(initial_nuclei, time_end, num_trials)

# Plot results
plt.hist(decay_times, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Distribution of Decay Times for Tritium')
plt.xlabel('Time (years)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Print statistics
mean_decay_time = np.mean(decay_times)
std_decay_time = np.std(decay_times)
print(f'Mean decay time: {mean_decay_time:.2f} years')
print(f'Standard deviation: {std_decay_time:.2f} years')
