#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from wavefunction import Potential
from wavefunction import WaveFunctionSolver
from wavefunction import MonteCarloSimulation

# Create instances and run the simulation
potential = Potential()
solver = WaveFunctionSolver(potential)
simulation = MonteCarloSimulation(solver)
simulation.run()
