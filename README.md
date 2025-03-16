# Dynamic Voltage Scaling (DVS) Simulation & Optimization

This repository contains two Python scripts for simulating and optimizing a Dynamic Voltage Scaling (DVS) server. The simulation models job arrivals, state transitions (OFF, SETUP, BUSY, SCALED, IDLE), and tracks performance metrics such as response times and energy consumption.

## Overview

### DVS Simulation (`DVS.py`)
- Uses **SimPy** for event-driven simulation.
- Models job processing with different states: `OFF`, `SETUP`, `BUSY`, `SCALED`, and `IDLE`.
- Tracks **response times**, **energy consumption**, and **server state durations**.

### Optimization Model (`DVS-GA-BA.py`)
- Uses **Genetic Algorithm (GA)** and **Bayesian Optimization**.
- Optimizes DVS server parameters for improved performance.
- Evaluates **energy efficiency** and **processing speed trade-offs**.

## Installation

Ensure you have Python 3 installed, then install the required dependencies:

```bash
pip install simpy numpy scipy scikit-learn
