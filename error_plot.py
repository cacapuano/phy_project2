import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.interpolate import interp1d

# Scode from schrodinger_eqn.py
class QuantumSystem:
    def __init__(self, x_initial, x_final, num_points, potential):
        self.x_initial = x_initial
        self.x_final = x_final
        self.num_points = num_points
        self.potential = potential
        self.x = np.linspace(x_initial, x_final, num_points)
        self.dx = self.x[1] - self.x[0]
        self.hbar = 1
        self.mass = 3

        self.hamiltonian = self._construct_hamiltonian()

    def _construct_hamiltonian(self):
        diag = np.diag(self.potential(self.x))
        off_diag = np.diag([-self.hbar**2 / (2 * self.mass * self.dx**2)] * (self.num_points - 1), -1) + \
                   np.diag([-self.hbar**2 / (2 * self.mass * self.dx**2)] * (self.num_points - 1), 1)
        return diag + off_diag

    def solve(self):
        energies, wavefunctions = eigh(self.hamiltonian)
        return energies, wavefunctions

    def normalize_wavefunctions(self, wavefunctions):
        norm_factors = np.sqrt(np.trapz(wavefunctions**2, self.x, axis=0))
        normalized_wavefunctions = wavefunctions / norm_factors
        return normalized_wavefunctions

    def plot_wavefunctions(self, wavefunctions):
        plt.figure(figsize=(10, 6))
        for i in range(3):  # Plotting first three wavefunctions
            plt.plot(self.x, 2 * wavefunctions[:, i]**2, label=f'Wave Function {i + 1}')
        plt.title('Probability Density of Wavefunctions')
        plt.xlabel('Position')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid()

# potential for a finite square well
def potential_function(x):
    return np.where((x > 0) & (x < 1), 0, 1000)  # Finite potential well

# solve
quantum_system = QuantumSystem(x_initial=0, x_final=1, num_points=1000, potential=potential_function)
energies, wavefunctions = quantum_system.solve()

normalized_wavefunctions = quantum_system.normalize_wavefunctions(wavefunctions) / 2

quantum_system.plot_wavefunctions(normalized_wavefunctions)

#code from Wavefunction 

hbar = 1.0
m = 3.0

class Potential:
    def __init__(self, width=1):
        self.width = width

    def value(self, x):
        if 0 < x < self.width:
            return 0
        else:
            return 10000000  # infinite potential outside the well

class WaveFunctionSolver:
    def __init__(self, potential):
        self.potential = potential

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
            k3 = dx * self.derivatives(x + dx / 2, y + k2 / 2, E)
            k4 = dx * self.derivatives(x + dx, y + k3, E)

            y_values[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return x_values, y_values

class MonteCarloSimulation:
    def __init__(self, solver, num_samples=1000, x0=0, x_end=1, dx=0.01, E=2):
        self.solver = solver
        self.num_samples = num_samples
        self.x0 = x0
        self.x_end = x_end
        self.dx = dx
        self.E = E
        self.psi = None  
        self.x_values = None  

    def run(self):
        for _ in range(self.num_samples):
            random_x = np.random.uniform(self.x0, self.x_end)
            self.x_values, solution = self.solver.runge_kutta(self.E, np.array([1, 0]), self.x0, self.x_end, self.dx)

            self.psi = solution[:, 0] 
            norm = np.sqrt(np.trapz(np.abs(self.psi)**2, self.x_values))
            self.psi /= norm

        # plot the simulated wave function probability density
        plt.plot(self.x_values, np.abs(self.psi)**2 / 2.2, color='green', label='Simulated Wave Function')
        plt.title('Wave Function Probability Densities')
        plt.xlabel('Position (m)')
        plt.ylabel('Probability Density')
        plt.grid()
        plt.legend()
        plt.show()

# run the simulation
potential = Potential()
solver = WaveFunctionSolver(potential)
simulation = MonteCarloSimulation(solver)
simulation.run()

# compute the error in probability densities
i = 0  # first wave function

# Ensure both arrays are of the same length
if len(normalized_wavefunctions[:, i]) != len(simulation.psi): #if the lengths are different
    interp_func = interp1d(simulation.x_values, simulation.psi, kind='linear', fill_value='extrapolate') #SciPy script
    psi_interp = interp_func(quantum_system.x)
else: #if the wave functions are the same
    psi_interp = 2 * simulation.psi

# subtract two wave functions
difference = 2 * normalized_wavefunctions[:, i]**2 - np.abs(psi_interp)**2

# difference in x values
x_differences = np.diff(quantum_system.x)

# plot
plt.figure(figsize=(10, 6))
plt.plot(quantum_system.x[:-1], np.abs(difference[:-1] / 8), label='Difference in Probability Densities', color='red')
plt.title('Difference between Probability Densities')
plt.xlabel('Position')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.show()