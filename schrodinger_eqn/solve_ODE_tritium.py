#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from schrodinger_eqn import QuantumSystem
from schrodinger_eqn import Potential
from schrodinger_eqn import WaveFunctionSolver
from schrodinger_eqn import MonteCarloSimulation

# potential for a finite square well
def potential_function(x):
    return np.where((x > 0) & (x < 1), 0, 1000)  # finite potential well

# create instances and run the simulation
potential = Potential()
solver = WaveFunctionSolver(potential)
simulation = MonteCarloSimulation(solver)
simulation.run()
