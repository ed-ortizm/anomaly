# Anomaly Detection in Galaxy Spectra Using Reconstruction-Based Scores

## Overview

This repository contains tools to compute anomaly scores for galaxy spectra based on reconstruction errors from a pre-trained **Variational AutoEncoder (VAE)**. The VAE model, a type of generative model, takes an input spectrum and reconstructs it. The anomaly score is determined by calculating the reconstruction error between the original spectrum and its reconstructed output, providing a measure of how unusual or "anomalous" each spectrum is. This approach helps identify spectra that differ significantly from the majority, which may indicate rare or novel astronomical objects or phenomena.

## Features

- **Multiple Anomaly Scoring Metrics**: Computes anomaly scores based on reconstruction error. Includes various scoring metrics such as:
    - **Mean Squared Error (MSE)**
    - **Maximum Absolute Error**
    - **Lp Metric** (generalized distance metric)

- **Modular Scoring System**: Provides a set of configurable scoring functions for detailed analysis.

- **Flexible Analysis and Plotting Tools**: Functions to analyze and visualize anomaly scores, enabling insight into patterns in anomalous spectra.

- **Data and Score Exploration**: Tools for exploring and analyzing reconstruction scores and anomaly distribution.

## Getting Started

### Prerequisites

- Python
- Dependencies for running VAE-based reconstruction and scoring (listed in `requirements.txt`)

### Installation

Clone the repository:

```bash
git clone https://github.com/ed-ortizm/anomaly.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. **Generate Reconstruction Scores**: Use the VAE to reconstruct input spectra and calculate the reconstruction error. These errors are used as anomaly scores.
2. **Analyze Anomalies**: Use the analysis and plotting scripts provided to visualize and interpret the distribution of anomaly scores.

### Repository Structure

```bash
├── anomalies              # Notebooks for specific anomaly analyses
├── data                   # Data files including spectra and line wavelengths
│   ├── line_wavelengths.dat
│   └── spectra
├── score                  # Configuration and scripts for calculating reconstruction-based scores
├── src                    # Source code for anomaly detection module
│   └── anomaly
│       ├── analysis.py        # Analysis tools for anomaly scoring
│       ├── constants.py       # Constants for model configurations
│       ├── distance.py        # Functions for distance calculations
│       ├── metrics.py         # Metrics for scoring anomalies
│       ├── parallelScore.py   # Parallelized scoring functionality
│       ├── plot.py            # Plotting functions for visualizations
│       ├── reconstruction.py  # VAE reconstruction handling
│       └── utils.py           # Utility functions
├── README.md              # This README file
├── setup.py               # Setup script for package installation
```

## Contact

For any questions or feedback, please reach out to:

- Edgar Ortiz (ed.ortizm@gmail.com)
- Mederic Boquien (mederic.boquien@oca.eu)

We welcome contributions and suggestions to improve this repository!
