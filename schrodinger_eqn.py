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
        self.hbar = 1  # value we're using for a reduced planck's constant
        self.mass = 3  # Mass of tritium (3H)

        self.hamiltonian = self._construct_hamiltonian() # set hamiltonian function

    def _construct_hamiltonian(self):
        #schrodinger eqn with the mass of tritium, x points, and potential energy
        potential = np.diag(self.potential(self.x))  # Potential energy of finite square well
        
        kinetic_energy = (-self.hbar**2 / (2 * self.mass)) * runga_kutta_wavefunction
        
        
        off_diag = np.diag([-self.hbar**2 / (2 * self.mass * self.dx**2)] * (self.num_points - 1), -1) + \
                   np.diag([-self.hbar**2 / (2 * self.mass * self.dx**2)] * (self.num_points - 1), 1) 
        return potential + off_diag

    def solve(self): #solve eigenvalues using SciPy script eigh
        energies, wavefunctions = eigh(self.hamiltonian)
        return energies, wavefunctions

    def plot_wavefunctions(self, wavefunctions):
        plt.figure(figsize=(10, 6))
        for i in range(3):  # plotting first three wavefunctions
            plt.plot(self.x, wavefunctions[:, i]**2, label=f'n={i+1}')
        plt.title('Probability Density of Wavefunctions')
        plt.xlabel('Position')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid()
        plt.show()

# potential for a finite square well

# note for me - cannot use infinite for this function, using 1000 for max of potential well makes it act like an infinite square well
def potential_function(x):
    return np.where((x > -1) & (x < 1), 0, 1000)  # finite potential well

#solve using classes
quantum_system = QuantumSystem(x_min=-1.5, x_max=1.5, num_points=1000, potential=potential_function)
energies, wavefunctions = quantum_system.solve()


# plot the wavefunctions
quantum_system.plot_wavefunctions(wavefunctions)

