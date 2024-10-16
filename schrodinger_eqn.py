import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

#set class as a quantum systems, fufilling class requirement
class QuantumSystem:
    def __init__(self, x_min, x_max, num_points, potential):
        self.x_min = x_min
        self.x_max = x_max
        self.num_points = num_points
        self.potential = potential #potential of tritium - use inifinite potential well
        self.x = np.linspace(x_min, x_max, num_points) #positon points
        self.dx = self.x[1] - self.x[0] 
        self.hbar = 1  # Reduced Planck's constant
        self.mass = 3  # Mass of tritium (3H)

        self.hamiltonian = self._construct_hamiltonian() # set hamiltonian function

    def _construct_hamiltonian(self):
        diag = np.diag(self.potential(self.x))  # Potential energy of infinite square well
        off_diag = np.diag([-self.hbar**2 / (2 * self.mass * self.dx**2)] * (self.num_points - 1), -1) + \
                   np.diag([-self.hbar**2 / (2 * self.mass * self.dx**2)] * (self.num_points - 1), 1)
        
        return diag + off_diag

    def solve(self): #solve eigenvalues using SciPy script eigh
        energies, wavefunctions = eigh(self.hamiltonian)
        return energies, wavefunctions

    def plot_wavefunctions(self, wavefunctions):
        """Plot the first few wavefunctions."""
        plt.figure(figsize=(10, 6))
        for i in range(3):  # Plot first three wavefunctions
            plt.plot(self.x, wavefunctions[:, i]**2, label=f'n={i+1}')
        plt.title('Probability Density of Wavefunctions')
        plt.xlabel('Position')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid()
        plt.show()

# Define the potential function
def potential_function(x):
    return np.where((x > -1) & (x < 1), 0, 1000)  # Infinite potential well

# Create an instance of the QuantumSystem
quantum_system = QuantumSystem(x_min=-1.5, x_max=1.5, num_points=1000, potential=potential_function)

# Solve for energies and wavefunctions
energies, wavefunctions = quantum_system.solve()


# Plot the wavefunctions
quantum_system.plot_wavefunctions(wavefunctions)

