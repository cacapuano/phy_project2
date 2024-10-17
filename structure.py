
# %% Cell: Monte Carlo simulation for soving ODE
import numpy as np
import matplotlib.pyplot as plt

# Constants
k = 0.1    # Decay constant
sigma = 0.2  # Noise magnitude
y0 = 100    # Initial value
time_end = 50  # Total time
num_trials = 1000  # Number of Monte Carlo trials
dt = 0.1  # Time step

# Function to simulate the ODE using Monte Carlo method
def monte_carlo_ode(y0, k, sigma, time_end, num_trials, dt):
    num_steps = int(time_end / dt)
    results = np.zeros((num_trials, num_steps))

    for trial in range(num_trials):
        y = y0
        for i in range(num_steps):
            # Update y according to the ODE with stochastic noise
            noise = np.random.normal(0, sigma)
            y = y - k * y * dt + noise
            results[trial, i] = max(y, 0)  # Ensure y does not go negative

    return results

# Run the simulation
results = monte_carlo_ode(y0, k, sigma, time_end, num_trials, dt)

# Calculate mean and standard deviation
mean_results = np.mean(results, axis=0)
std_results = np.std(results, axis=0)

# Time vector
time_points = np.linspace(0, time_end, int(time_end / dt))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time_points, mean_results, label='Mean of Trials', color='blue')
plt.fill_between(time_points, mean_results - std_results, mean_results + std_results,
                 color='blue', alpha=0.2, label='Standard Deviation')
plt.title('Monte Carlo Simulation of Stochastic ODE')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# Print final mean and std deviation
print(f'Final Mean Value: {mean_results[-1]}')
print(f'Final Standard Deviation: {std_results[-1]}')
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
