
# F1 Tire Model Prebuilder (v1)

A Bayesian tire performance modeling system that processes real Formula 1 telemetry data to build predictive tire degradation models for race strategy simulation.

## Overview

This system processes real F1 2024 race data to build Bayesian tire performance models using:
- **Real F1 Telemetry**: 2024 season data via FastF1 API
- **Bayesian Inference**: MCMC sampling for degradation parameter estimation
- **Fuel Correction**: Weight-adjusted lap time calculations
- **Production Ready**: Prebuilt models for fast deployment

## Features

### Data Processing
- Automated F1 2024 telemetry download and processing
- Fuel-corrected lap time calculations accounting for weight reduction
- Comprehensive data filtering and validation
- Support for all 24 circuits in the 2025 F1 calendar

### Bayesian Tire Modeling
- **Model Structure**: Linear degradation with uncertainty quantification
- **Statistical Framework**: `laptime ~ Normal(α + β × stint_lap, σ)`
- **Inference Method**: NUTS sampling via NumPyro
- **Prior Specifications**: Domain-informed priors for tire performance parameters

### Model Persistence
- Pickle-based model serialization for production deployment
- Comprehensive metadata storage including data quality metrics
- Circuit-specific model files for modular loading
- Summary reporting for model validation

## Installation

```bash
# Clone repository
git clone <[repository-url](https://github.com/j5t3313/tireModel)>
cd f1-tire-model-v1

# Install dependencies
pip install -r requirements.txt

# Create FastF1 cache directory
mkdir .f1_cache
```

## Usage

### Build All Models
```python
from tireModel import F1ModelPrebuilder

# Initialize prebuilder
prebuilder = F1ModelPrebuilder(output_dir="prebuilt_models")

# Build models for all circuits
prebuilder.prebuild_all_models()
```

### Command Line Usage
```bash
# Run the prebuilding process
python tireModel.py
```

### Load Prebuilt Models
```python
import pickle
from pathlib import Path

# Load specific circuit model
models_dir = Path("prebuilt_models")
with open(models_dir / "monaco_models.pkl", 'rb') as f:
    monaco_data = pickle.load(f)

# Access Bayesian models
soft_model = monaco_data['models']['SOFT']
samples = soft_model['samples']

# Get model parameters
alpha_samples = samples['alpha']  # Intercept samples
beta_samples = samples['beta']    # Degradation rate samples
sigma_samples = samples['sigma']  # Noise samples
```

## Model Structure

### Bayesian Framework
- **Likelihood**: Normal distribution for lap times
- **Parameters**: 
  - `α` (alpha): Baseline lap time intercept
  - `β` (beta): Linear degradation rate per lap
  - `σ` (sigma): Model uncertainty/noise
- **Priors**:
  - `α ~ Normal(80, 5)`: Baseline around 80 seconds
  - `β ~ Normal(0.03, 0.02)`: Small positive degradation
  - `σ ~ HalfNormal(0.5)`: Reasonable lap time variance

### Sampling Configuration
- **Warmup**: 500 samples for chain initialization
- **Posterior**: 1000 samples for inference
- **Algorithm**: NUTS (No-U-Turn Sampler)
- **Convergence**: Automatic step size adaptation

## Output Structure

Each circuit generates a pickle file containing:
```
{circuit_name}_models.pkl:
├── circuit_name: str
├── models: dict
│   ├── SOFT: model_data
│   ├── MEDIUM: model_data  
│   └── HARD: model_data
├── processed_data_summary: dict
│   ├── total_laps: int
│   ├── compounds: compound_counts
│   ├── drivers: driver_count
│   └── stint_range: [min_stint, max_stint]
└── created_timestamp: ISO_timestamp
```

### Model Data Structure
```python
model_data = {
    'samples': {
        'alpha': np.array,    # Posterior samples for intercept
        'beta': np.array,     # Posterior samples for degradation
        'sigma': np.array     # Posterior samples for noise
    },
    'n_observations': int,    # Number of data points used
    'circuit': str,           # Circuit name
    'compound': str,          # Tire compound
    'x_range': [min, max],    # Stint lap range in data
    'y_range': [min, max]     # Lap time range in data
}
```

## Data Processing Pipeline

### 1. Data Acquisition
- Downloads 2024 F1 race data via FastF1
- Extracts lap-by-lap telemetry including times, compounds, stint information
- Filters for valid racing laps (60-300 second range)

### 2. Feature Engineering
- **Stint Lap Calculation**: Position within each tire stint
- **Fuel Correction**: Accounts for decreasing car weight during race
- **Data Validation**: Removes outliers and non-racing conditions

### 3. Model Building
- Groups data by tire compound (SOFT, MEDIUM, HARD)
- Fits Bayesian linear regression to stint lap vs. lap time relationship
- Requires minimum 15 observations per compound for model stability

## Quality Assurance

### Data Quality Checks
- Minimum sample size requirements (15+ laps per compound)
- Lap time range validation (60-300 seconds)
- Stint length filtering (≤50 laps)
- Compound validation (SOFT/MEDIUM/HARD only)

### Model Validation
- Convergence monitoring through MCMC diagnostics
- Prior-posterior comparison for parameter reasonableness
- Data range tracking for model applicability assessment

## Performance Characteristics

### Computational Requirements
- **Build Time**: ~5-10 minutes per circuit
- **Memory Usage**: ~500MB-1GB during processing
- **Storage**: ~10-50MB per circuit model file
- **Dependencies**: JAX-compatible system for MCMC

### Model Accuracy
- Captures tire degradation trends from real F1 data
- Quantifies prediction uncertainty through posterior distributions
- Accounts for fuel weight effects on lap time performance

## Limitations

- **Data Dependency**: Requires 2024 F1 data availability via FastF1
- **Weather Independence**: Models dry conditions only
- **Driver Neutrality**: Does not account for driver-specific performance
- **Track Conditions**: Assumes consistent track surface and conditions

## Circuit Coverage

Supports all 24 circuits from the 2025 F1 calendar:
- Australia, China, Japan, Bahrain, Saudi Arabia, Miami
- Imola, Monaco, Spain, Canada, Austria, Britain
- Belgium, Hungary, Netherlands, Italy, Azerbaijan
- Singapore, United States, Mexico, Brazil, Las Vegas, Qatar, Abu Dhabi

## Integration

This model builder is designed for integration with F1 strategy simulation systems:
- **Streamlit Apps**: Fast model loading for interactive applications
- **Strategy Simulators**: Probabilistic tire performance predictions
- **Monte Carlo Analysis**: Uncertainty propagation in race simulations

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- FastF1 project for F1 data access
- NumPyro team for Bayesian inference framework
- JAX team for high-performance computing backend
- Formula 1 for providing telemetry data access
