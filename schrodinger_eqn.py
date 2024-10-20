import numpy as np
import matplotlib.pyplot as plt

class QuantumSystem:
    def __init__(self, x_min, x_max, num_points, bohr_radius, mass, potential=0):
        self.x_values = np.linspace(x_min, x_max, num_points)  # Range of x values
        self.dx = self.x_values[1] - self.x_values[0]  # Space step
        self.bohr_radius = bohr_radius
        self.mass = mass
        self.hbar = 1.0545718e-34  # Reduced Planck's constant in J·s
        self.V = np.full_like(self.x_values, potential, dtype=np.complex128)  # Potential energy as complex array
        self.psi = self.wave_function(self.x_values).astype(np.complex128)  # Initial wave function as complex

    def wave_function(self, x):
        sqrt_term = 1 / np.sqrt(np.pi)
        r = np.abs(x)  # Using absolute value for radial distance
        return sqrt_term * (1 / self.bohr_radius)**(3/2) * np.exp(-r / self.bohr_radius)

    def normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)  # Normalization factor
        self.psi /= norm  # Normalize the wave function

    def second_derivative(self, psi):
        # Approximate the second derivative using central differences
        d2_psi = np.zeros_like(psi, dtype=np.complex128)
        d2_psi[1:-1] = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / (self.dx**2)
        return d2_psi

    def schrodinger_step(self):
        # Calculate the second derivative
        d2_psi = self.second_derivative(self.psi)

        # Calculate k values for the Runge-Kutta method
        k1 = -1j * (-(self.hbar**2 / (2 * self.mass)) * d2_psi + self.V * self.psi)
        k2 = -1j * (-(self.hbar**2 / (2 * self.mass)) * self.second_derivative(self.psi + self.dt / 2 * k1) + self.V * (self.psi + self.dt / 2 * k1))
        k3 = -1j * (-(self.hbar**2 / (2 * self.mass)) * self.second_derivative(self.psi + self.dt / 2 * k2) + self.V * (self.psi + self.dt / 2 * k2))
        k4 = -1j * (-(self.hbar**2 / (2 * self.mass)) * self.second_derivative(self.psi + self.dt * k3) + self.V * (self.psi + self.dt * k3))

        # Update the wave function
        self.psi += (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, num_steps, dt):
        self.dt = dt  # Set the time step
        self.normalize()  # Normalize the initial wave function

        for _ in range(num_steps):
            self.schrodinger_step()  # Update wave function

        self.normalize()  # Normalize the final wave function

    def probability_density(self):
        return np.abs(self.psi)**2

    def plot_probability_density(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.probability_density(), label='Probability Density')
        plt.title('Normalized Probability Density as a Function of x')
        plt.xlabel('Position (m)')
        plt.ylabel('Probability Density (|ψ|²)')
        plt.legend()
        plt.grid()
        plt.show()
    def monte_carlo_simulation(self, num_samples, potential_range, num_steps, dt):
        all_probabilities = []
        
        for _ in range(num_samples):
            # Random potential value within the specified range
            random_potential = np.random.uniform(*potential_range)
            self.V = np.full_like(self.x_values, random_potential, dtype=np.complex128)  # Update potential
            
            # Reset wave function and simulate
            self.psi = self.wave_function(self.x_values).astype(np.complex128)
            self.simulate(num_steps, dt)

            # Store the probability density
            all_probabilities.append(self.probability_density())

        return np.array(all_probabilities)

# Constants
bohr_radius = 5.29177e-11  # Bohr radius in meters
mass = 9.10938356e-31  # Mass of electron in kg

# Create a quantum system and run the simulation
quantum_system = QuantumSystem(x_min=-5 * bohr_radius, x_max=5 * bohr_radius, num_points=1000, bohr_radius=bohr_radius, mass=mass)
quantum_system.simulate(num_steps=1000, dt=1e-18)

# Plot the probability density
quantum_system.plot_probability_density()
