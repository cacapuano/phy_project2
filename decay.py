# Monte Carlo simulation on tritium decay
import numpy as np
import matplotlib.pyplot as plt

# Constants
tou = 12.32
Lambda = np.log(2) / tou

# Simulation parameters
initial_particles = 1000
end_time = 100
sampling = 1000 

def monte_carlo_tritium_decay(initial_particles, end_time, sampling):
    decay_times = []

    for _ in range(sampling):
        N = initial_particles
        time = 0

        while N > 0 and time < end_time:
            # Random time until next decay
            time_before_decay = np.random.exponential(1 / Lambda)
            time += time_before_decay

            # if decay happens
            if np.random.rand() < 1 - np.exp(-Lambda * time_until_decay):
                N -= 1

        decay_times.append(time)

    return decay_times

decay_times = monte_carlo_tritium_decay(initial_particles, end_time, sampling)

plt.hist(decay_times, bins=30, density=True, alpha=0.5, color='lightblue')
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
