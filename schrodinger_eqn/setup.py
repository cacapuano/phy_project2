#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='ODE solutino for tritum wavefunction',
    version='0.1',
    packages=find_packages(),
    description='solve schrodinger eqn to get tritum wavefunction',
    author='Caroline and Jasmine',
    author_email='cecapuan@syr.edu, yhuan223@syr.edu',
    install_requires=['numpy', 'matplotlib', 'scipy'],
)

