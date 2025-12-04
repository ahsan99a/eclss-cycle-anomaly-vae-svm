**ECLSS Synthetic Dataset Generator**

A physics-informed dataset for anomaly detection in spacecraft Environmental Control & Life Support Systems (ECLSS)

This repository provides a fully reproducible pipeline for generating, validating, and analyzing a synthetic ECLSS dataset.
It was created for research on fault detection, VAE/SVM anomaly detection, and health monitoring of spacecraft life-support systems.

🚀 What’s Included
1. Dataset Generator (src/generate_dataset.py)

Produces clean, labeled time-series cycles for:

O₂ concentration

CO₂ concentration

Cabin pressure

Includes 6 system fault modes (CO₂ leak, valve stiction, vacuum anomaly, etc.)

Includes 5 sensor fault types (noise, drift, freeze, spikes, etc.)

Automatically runs:

Physical sanity checks

Temporal coherence validation

PCA-based class separability

Overlay visualization plots

Summary report generation

2. Extra Validation Script (src/extra_validation.py)

Optional statistical diagnostics:

Per-class summary statistics

KS-tests on nominal cycles

Cross-correlation analysis

Approximate SNR computation

These tests are exploratory only (no pass/fail), meant for deeper understanding and reporting.

3. Ready-to-Use Dataset (data/eclss_synthetic_dataset_full/)

Includes:

cycles_3d.npy (N × T × 3)

labels_system.npy

labels_sensor.npy

metadata.csv

Visualization plots

Summary markdown report
