# ECLSS Synthetic Dataset Summary

## Dataset Statistics

- Total Samples: 720
- Timesteps per Sample: 1000
- Sensors: 3 (O₂, CO₂, Pressure)
- Sampling Rate: 2.0 Hz
- Cycle Duration: 8.3 minutes

## System Class Distribution

| Class ID | Class Name | Count | Percentage |
|----------|------------|-------|------------|
| 0 | Nominal | 120 | 16.7% |
| 1 | CO2_Leak | 120 | 16.7% |
| 2 | Valve_Stiction | 120 | 16.7% |
| 3 | Vacuum_Anomaly | 120 | 16.7% |
| 4 | CDRA_Degradation | 120 | 16.7% |
| 5 | OGA_Degradation | 120 | 16.7% |

## Sensor Fault Distribution

| Fault ID | Fault Name | Count | Percentage |
|----------|------------|-------|------------|
| 0 | None | 617 | 85.7% |
| 1 | Bias_Drift | 23 | 3.2% |
| 2 | High_Noise | 30 | 4.2% |
| 3 | Partial_Freeze | 20 | 2.8% |
| 4 | Spike_Outliers | 30 | 4.2% |

## Safety Flags Triggered

- flag_CO2_warn: 97 samples (13.5%)
- flag_CO2_crit: 16 samples (2.2%)
- flag_O2_warn_low: 75 samples (10.4%)
- flag_O2_crit_low: 0 samples (0.0%)
- flag_P_warn: 0 samples (0.0%)
- flag_P_crit: 0 samples (0.0%)


## Sensor Value Ranges

| Sensor | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| O₂ (%) | 16.062 | 23.964 | 20.528 | 0.963 |
| CO₂ (%) | 0.000 | 3.818 | 0.354 | 0.148 |
| Pressure (psi) | 11.764 | 18.052 | 14.689 | 0.255 |

## Configuration (JSON)

{
  "n_samples_per_system_class": 120,
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

*Generated on 2025-11-28 11:22:49*
