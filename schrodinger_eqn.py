import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Planck's constant (J·s)
m = 3.0     # Electron mass (kg)
bohr_radius = 5.29e-11  # Bohr radius (m)

class Potential:
    def __init__(self, width=1):
        self.width = width

    def value(self, x):
        if 0 < x < self.width:
            return 0
        else:
            return 10000000  # Infinite potential outside the well

class WaveFunctionSolver:
    def __init__(self, potential):
        self.potential = potential
        self.bohr_radius = bohr_radius

    def derivatives(self, x, y, E):
        psi1 = y[0]
        psi2 = y[1]
        V = self.potential.value(x)
        dpsi1_dx = psi2
        dpsi2_dx = (2 * m / hbar**2) * (V - E) * psi1
        return np.array([dpsi1_dx, dpsi2_dx])

    def runge_kutta(self, E, y0, x0, x_end, dx):
        num_steps = int((x_end - x0) / dx) + 1
        x_values = np.linspace(x0, x_end, num_steps)
        y_values = np.zeros((num_steps, len(y0)))
        y_values[0] = y0

        for i in range(num_steps - 1):
            x = x_values[i]
            y = y_values[i]

            k1 = dx * self.derivatives(x, y, E)
            k2 = dx * self.derivatives(x + dx / 2, y + k1 / 2, E)
            k3 = self.derivatives(x + dx / 2, y + k2 / 2, E) * dx
            k4 = self.derivatives(x + dx, y + k3, E) * dx

            y_values[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return x_values, y_values

    def wave_function(self, x):
        sqrt_term = 1 / np.sqrt(np.pi)
        r = np.abs(x)  # Radial distance
        return sqrt_term * (1 / self.bohr_radius)**(3/2) * np.exp(-r / self.bohr_radius)

class MonteCarloSimulation:
    def __init__(self, solver, num_samples=1000, x0=0, x_end=1, dx=0.01, E=2):
        self.solver = solver
        self.num_samples = num_samples
        self.x0 = x0
        self.x_end = x_end
        self.dx = dx
        self.E = E

    def run(self):
        plt.figure(figsize=(10, 6))
        total_density = np.zeros(int((self.x_end - self.x0) / self.dx) + 1)

        for _ in range(self.num_samples):
            x_values, solution = self.solver.runge_kutta(self.E, np.array([1, 0]), self.x0, self.x_end, self.dx)

            psi = solution[:, 0]
            norm = np.sqrt(np.trapz(np.abs(psi)**2, x_values))
            psi /= norm
            total_density += np.abs(psi)**2

        average_density = total_density / self.num_samples
        plt.plot(x_values, average_density, color='blue', label='Normalized Wave Function Probability Density')

        # Calculate the real wave function for tritium
        x_tritium = np.linspace(-5 * self.solver.bohr_radius, 5 * self.solver.bohr_radius, 1000)
        psi_tritium = self.solver.wave_function(x_tritium)

        # Normalize tritium wave function
        norm_tritium = np.sqrt(np.trapz(np.abs(psi_tritium)**2, x_tritium))
        psi_tritium /= norm_tritium

        plt.plot(x_tritium, np.abs(psi_tritium)**2, color='red', label='Tritium Wave Function')

        # Calculate error
        # Use the same range for comparison
        x_common = np.linspace(self.x0, self.x_end, len(average_density))
        average_density_common = np.interp(x_common, x_values, average_density)

        error = np.abs(average_density_common - np.abs(psi_tritium[:len(average_density_common)])**2)
        print("Maximum error between simulated and real wave functions:", np.max(error))

        plt.title('Wave Function Probability Densities')
        plt.xlabel('Position (m)')
        plt.ylabel('Probability Density')
        plt.grid()
        plt.legend()
        plt.show()

# Create instances and run the simulation
potential = Potential()
solver = WaveFunctionSolver(potential)
simulation = MonteCarloSimulation(solver)
simulation.run()
