#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# constants
hbar = 1.0  # Planck's constant (J·s)
m = 3.0     # Electron mass (kg)

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

    def runge_kutta(self, E, y0, x0, x_end, dx): #solving second ODE
        num_steps = int((x_end - x0) / dx) + 1
        x_values = np.linspace(x0, x_end, num_steps)
        y_values = np.zeros((num_steps, len(y0)))
        y_values[0] = y0

        for i in range(num_steps - 1):
            x = x_values[i]
            y = y_values[i]

            #runga kutta coefficients

            k1 = dx * self.derivatives(x, y, E)
            k2 = dx * self.derivatives(x + dx / 2, y + k1 / 2, E)
            k3 = self.derivatives(x + dx / 2, y + k2 / 2, E) * dx
            k4 = self.derivatives(x + dx, y + k3, E) * dx

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

    def run(self):
        plt.figure(figsize=(10, 6))

        for _ in range(self.num_samples):
            # generate a random x value within the specified range
            random_x = np.random.uniform(self.x0, self.x_end)
            x_values, solution = self.solver.runge_kutta(self.E, np.array([1, 0]), self.x0, self.x_end, self.dx)

            # find the corresponding psi value for the random x
            psi = solution[:, 0]
            norm = np.sqrt(np.trapz(np.abs(psi)**2, x_values))
            psi /= norm

        plt.plot(x_values, np.abs(psi)**2 / 2.2, color='blue', alpha=0.1)

        plt.title('Monte Carlo Simulation of Wave Function Probability Densities')
        plt.xlabel('Position (m)')
        plt.ylabel('Probability Density')
        plt.grid()
        plt.savefig('wavefunction_tritium.jpg')
        plt.show()

