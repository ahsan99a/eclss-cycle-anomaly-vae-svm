# ECLSS Synthetic Dataset Summary

## Dataset Statistics

- Total Samples: 180
- Timesteps per Sample: 1000
- Sensors: 3 (O₂, CO₂, Pressure)
- Sampling Rate: 2.0 Hz
- Cycle Duration: 8.3 minutes

## System Class Distribution

| Class ID | Class Name | Count | Percentage |
|----------|------------|-------|------------|
| 0 | Nominal | 30 | 16.7% |
| 1 | CO2_Leak | 30 | 16.7% |
| 2 | Valve_Stiction | 30 | 16.7% |
| 3 | Vacuum_Anomaly | 30 | 16.7% |
| 4 | CDRA_Degradation | 30 | 16.7% |
| 5 | OGA_Degradation | 30 | 16.7% |

## Sensor Fault Distribution

| Fault ID | Fault Name | Count | Percentage |
|----------|------------|-------|------------|
| 0 | None | 153 | 85.0% |
| 1 | Bias_Drift | 5 | 2.8% |
| 2 | High_Noise | 4 | 2.2% |
| 3 | Partial_Freeze | 8 | 4.4% |
| 4 | Spike_Outliers | 10 | 5.6% |

## Safety Flags Triggered

- flag_CO2_warn: 28 samples (15.6%)
- flag_CO2_crit: 5 samples (2.8%)
- flag_O2_warn_low: 18 samples (10.0%)
- flag_O2_crit_low: 0 samples (0.0%)
- flag_P_warn: 0 samples (0.0%)
- flag_P_crit: 0 samples (0.0%)


## Sensor Value Ranges

| Sensor | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| O₂ (%) | 16.578 | 23.858 | 20.534 | 0.953 |
| CO₂ (%) | 0.000 | 3.305 | 0.356 | 0.145 |
| Pressure (psi) | 11.883 | 17.997 | 14.690 | 0.254 |

## Configuration (JSON)

{
  "n_samples_per_system_class": 30,
  "n_timesteps": 1000,
  "sampling_rate_hz": 2.0,
  "O2_nominal": 20.9,
  "CO2_nominal": 0.3,
  "pressure_nominal_psi": 14.7,
  "O2_amp": 0.3,
  "CO2_amp": 0.1,
  "P_amp": 0.3,
  "O2_noise_std": 0.05,
  "CO2_noise_std": 0.02,
  "P_noise_std": 0.05,
  "enable_drift": true,
  "drift_std_O2": 0.01,
  "drift_std_CO2": 0.003,
  "drift_std_P": 0.01,
  "freq_jitter_range": [
    0.95,
    1.05
  ],
  "co2_leak_magnitude_range": [
    0.05,
    0.4
  ],
  "valve_stiction_alpha_range": [
    0.85,
    0.99
  ],
  "vacuum_drop_range": [
    0.5,
    2.0
  ],
  "cdra_offset_range": [
    0.05,
    0.5
  ],
  "oga_offset_range": [
    0.5,
    4.0
  ],
  "enable_sensor_faults": true,
  "p_sensor_fault": 0.15,
  "bias_drift_max": 0.5,
  "high_noise_factor": 4.0,
  "freeze_min_fraction": 0.2,
  "freeze_max_fraction": 0.6,
  "n_spikes_min": 5,
  "n_spikes_max": 30,
  "spike_magnitude_range": [
    0.5,
    3.0
  ],
  "O2_min": 0.0,
  "O2_max": 100.0,
  "CO2_min": 0.0,
  "CO2_max": 5.0,
  "P_min": 0.0,
  "P_max": 20.0,
  "CO2_warn": 0.7,
  "CO2_crit": 1.0,
  "O2_warn_low": 19.0,
  "O2_crit_low": 16.0,
  "P_warn_low": 14.0,
  "P_warn_high": 15.4,
  "P_crit_low": 13.5,
  "P_crit_high": 15.8,
  "random_seed": 42
}

---

*Generated on 2025-11-24 19:27:49*
