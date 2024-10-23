#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='Monte Carlo simulation for tritum wavefunction',
    version='0.1',
    packages=find_packages(),
    description='Use Monte Carlo simulation to get tritum wavefunction',
    author='Caroline and Jasmine',
    author_email='cecapuan@syr.edu, yhuan223@syr.edu',
    install_requires=['numpy', 'matplotlib'],
)

