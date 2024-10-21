import numpy as np
import matplotlib.pyplot as plt

class QuantumSystem:
    def __init__(self, x_initial, x_final, num_points, bohr_radius, mass, potential=0): #init function for classes
        self.x_values = np.linspace(x_initial, x_final, num_points)  # range of x values
        self.dx = self.x_values[1] - self.x_values[0]  # step in space
        self.bohr_radius = bohr_radius
        self.mass = mass
        self.hbar = 1.0545718e-34  # Reduced Planck's constant in J·s
        self.V = np.full_like(self.x_values, potential, dtype=np.complex128)  # potential energy
        self.wave = self.wave_function(self.x_values).astype(np.complex128)  # initial wavefunction that will be later updated using runga-kutta

    def wave_function(self, x): #wave function of tritium
        sqrt_term = 1 / np.sqrt(np.pi)
        r = np.abs(x)  # radial distance
        return sqrt_term * (1 / self.bohr_radius)**(3/2) * np.exp(-r / self.bohr_radius)

    def normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.wave)**2) * self.dx) 
        self.wave /= norm  # Normalize the wave function

    def second_derivative(self, wave): # approximating the second derivative 
        d2_wave = np.zeros_like(wave, dtype=np.complex128)
        d2_wave[1:-1] = (wave[2:] - 2 * wave[1:-1] + wave[:-2]) / (self.dx**2)
        return d2_wave

    def runga_kutta(self):
        # calculates the second derivative
        d2_wave = self.second_derivative(self.wave)

        # calculate the coefficents for runga-kutta
        k1 = -1j * (-(self.hbar**2 / (2 * self.mass)) * d2_wave + self.V * self.wave)
        k2 = -1j * (-(self.hbar**2 / (2 * self.mass)) * self.second_derivative(self.wave + self.dt / 2 * k1) + self.V * (self.wave + self.dt / 2 * k1))
        k3 = -1j * (-(self.hbar**2 / (2 * self.mass)) * self.second_derivative(self.wave + self.dt / 2 * k2) + self.V * (self.wave + self.dt / 2 * k2))
        k4 = -1j * (-(self.hbar**2 / (2 * self.mass)) * self.second_derivative(self.wave + self.dt * k3) + self.V * (self.wave + self.dt * k3))

        # update wavefunction
        self.wave += (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, num_steps, dt):
        self.dt = dt  # time step
        self.normalize()  # normalize the initial wave function

        for _ in range(num_steps):
            self.runga_kutta()  # update wave function

        self.normalize()  # Normalize the final wave function

    def probability_density(self):
        return np.abs(self.wave)**2

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
            # random potential value within the specified range
            random_potential = np.random.uniform(*potential_range)
            self.V = np.full_like(self.x_values, random_potential, dtype=np.complex128)  # Update potential
            
            # simulate wavefunction
            self.wave = self.wave_function(self.x_values).astype(np.complex128)
            self.simulate(num_steps, dt)

            # Store the probability density
            all_probabilities.append(self.probability_density())

        return np.array(all_probabilities)

bohr_radius = 5.29177e-11  # Bohr radius in meters
#mass = 9.10938356e-31  # Mass of electron in kg
mass = 3 # mass of tritium

# run the simulation
quantum_system = QuantumSystem(x_initial=-5 * bohr_radius, x_final=5 * bohr_radius, num_points=1000, bohr_radius=bohr_radius, mass=mass)
quantum_system.simulate(num_steps=1000, dt=1e-18)

# Plot the probability density
quantum_system.plot_probability_density()
