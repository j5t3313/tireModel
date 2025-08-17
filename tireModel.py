#!/usr/bin/env python3

import pickle
import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# FastF1 and Bayesian modeling imports
try:
    import fastf1
    import jax.numpy as jnp
    import jax.random as random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False
    exit(1)

class F1ModelPrebuilder:
    def __init__(self, output_dir="prebuilt_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache directory for FastF1
        cache_dir = '.f1_cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        fastf1.Cache.enable_cache(cache_dir)
        
        # 2025 F1 Calendar 
        self.circuit_data = [
            ('Australia', {'laps': 58, 'distance_km': 5.278, 'gp_name': 'Australian Grand Prix'}),
            ('China', {'laps': 56, 'distance_km': 5.451, 'gp_name': 'Chinese Grand Prix'}),
            ('Japan', {'laps': 53, 'distance_km': 5.807, 'gp_name': 'Japanese Grand Prix'}),
            ('Bahrain', {'laps': 57, 'distance_km': 5.412, 'gp_name': 'Bahrain Grand Prix'}),
            ('Saudi Arabia', {'laps': 50, 'distance_km': 6.174, 'gp_name': 'Saudi Arabian Grand Prix'}),
            ('Miami', {'laps': 57, 'distance_km': 5.41, 'gp_name': 'Miami Grand Prix'}),
            ('Imola', {'laps': 63, 'distance_km': 4.909, 'gp_name': 'Emilia Romagna Grand Prix'}),
            ('Monaco', {'laps': 78, 'distance_km': 3.337, 'gp_name': 'Monaco Grand Prix'}),
            ('Spain', {'laps': 66, 'distance_km': 4.655, 'gp_name': 'Spanish Grand Prix'}),
            ('Canada', {'laps': 70, 'distance_km': 4.361, 'gp_name': 'Canadian Grand Prix'}),
            ('Austria', {'laps': 71, 'distance_km': 4.318, 'gp_name': 'Austrian Grand Prix'}),
            ('Britain', {'laps': 52, 'distance_km': 5.891, 'gp_name': 'British Grand Prix'}),
            ('Belgium', {'laps': 44, 'distance_km': 7.004, 'gp_name': 'Belgian Grand Prix'}),
            ('Hungary', {'laps': 70, 'distance_km': 4.381, 'gp_name': 'Hungarian Grand Prix'}),
            ('Netherlands', {'laps': 72, 'distance_km': 4.259, 'gp_name': 'Dutch Grand Prix'}),
            ('Italy', {'laps': 53, 'distance_km': 5.793, 'gp_name': 'Italian Grand Prix'}),
            ('Azerbaijan', {'laps': 51, 'distance_km': 6.003, 'gp_name': 'Azerbaijan Grand Prix'}),
            ('Singapore', {'laps': 62, 'distance_km': 4.940, 'gp_name': 'Singapore Grand Prix'}),
            ('United States', {'laps': 56, 'distance_km': 5.513, 'gp_name': 'United States Grand Prix'}),
            ('Mexico', {'laps': 71, 'distance_km': 4.304, 'gp_name': 'Mexico City Grand Prix'}),
            ('Brazil', {'laps': 71, 'distance_km': 4.309, 'gp_name': 'São Paulo Grand Prix'}),
            ('Las Vegas', {'laps': 50, 'distance_km': 6.201, 'gp_name': 'Las Vegas Grand Prix'}),
            ('Qatar', {'laps': 57, 'distance_km': 5.380, 'gp_name': 'Qatar Grand Prix'}),
            ('Abu Dhabi', {'laps': 58, 'distance_km': 5.281, 'gp_name': 'Abu Dhabi Grand Prix'})
        ]
        
        self.circuits = {name: data for name, data in self.circuit_data}
        
    def calculate_fuel_corrected_laptime(self, raw_laptime, lap_number, total_laps, 
                                       fuel_consumption_per_lap, weight_effect=0.03):
        """Fuel corrected laptime calculation"""
        remaining_laps = total_laps - lap_number
        fuel_correction = remaining_laps * fuel_consumption_per_lap * weight_effect
        return raw_laptime - fuel_correction
    
    def load_and_process_f1_data(self, circuit_name):
        """Load and process F1 data for a circuit"""
        print(f"  Loading 2024 data for {circuit_name}...")
        
        circuit_info = self.circuits.get(circuit_name)
        if not circuit_info:
            return None
            
        try:
            # Load the specific GP's 2024 race data
            session = fastf1.get_session(2024, circuit_info['gp_name'], 'R')
            session.load()
            
            # Process laps 
            laps = session.laps
            stints = laps[["Driver", "Stint", "Compound", "LapNumber", "LapTime"]].copy()
            stints["LapTime_s"] = stints["LapTime"].dt.total_seconds()
            stints.dropna(subset=["LapTime_s"], inplace=True)
            stints["StintLap"] = stints.groupby(["Driver", "Stint"]).cumcount() + 1
            
            # Filter valid data
            stints = stints[
                (stints["LapTime_s"] > 60) &
                (stints["LapTime_s"] < 300) &
                (stints["StintLap"] <= 50) &
                (stints["Compound"].isin(['SOFT', 'MEDIUM', 'HARD']))
            ]
            
            if len(stints) == 0:
                print(f"    No valid data found for {circuit_name}")
                return None
            
            # Apply fuel correction
            circuit_laps = circuit_info['laps']
            fuel_per_lap = 105.0 / circuit_laps # 110 kg/ num laps w/ 5kg reserve
            weight_effect = 0.03
            
            stints['LapTime_FC'] = stints.apply(
                lambda row: self.calculate_fuel_corrected_laptime(
                    row['LapTime_s'], row['LapNumber'], circuit_laps, fuel_per_lap, weight_effect
                ), axis=1
            )
            
            print(f"    Processed {len(stints)} laps")
            return stints
            
        except Exception as e:
            print(f"    Error loading data for {circuit_info['gp_name']}: {str(e)}")
            return None
    
    def build_bayesian_tire_model(self, compound_data, circuit_name, compound):
        """Build Bayesian tire model from compound data"""
        print(f"    Building {compound} model...")
        
        if len(compound_data) < 15:
            print(f"      Insufficient data for {compound} ({len(compound_data)} laps)")
            return None
        
        try:
            x_data = jnp.array(compound_data['StintLap'].values, dtype=jnp.float32)
            y_data = jnp.array(compound_data['LapTime_FC'].values, dtype=jnp.float32)
            
            def model(x, y=None):
                alpha = numpyro.sample("alpha", dist.Normal(80, 5))
                beta = numpyro.sample("beta", dist.Normal(0.03, 0.02))
                sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
                mu = alpha + beta * x
                numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

            kernel = NUTS(model)
            mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, progress_bar=False)
            mcmc.run(random.PRNGKey(0), x_data, y_data)
            
            # Extract samples for serialization
            samples = mcmc.get_samples()
            samples_dict = {
                'alpha': np.array(samples['alpha']),
                'beta': np.array(samples['beta']),
                'sigma': np.array(samples['sigma'])
            }
            
            model_data = {
                'samples': samples_dict,
                'n_observations': len(compound_data),
                'circuit': circuit_name,
                'compound': compound,
                'x_range': [float(x_data.min()), float(x_data.max())],
                'y_range': [float(y_data.min()), float(y_data.max())]
            }
            
            print(f"      Built {compound} model with {len(compound_data)} observations")
            return model_data
            
        except Exception as e:
            print(f"      Error building {compound} model: {str(e)}")
            return None
    
    def save_circuit_models(self, circuit_name, models, processed_data):
        """Save models and data for a circuit"""
        circuit_file = self.output_dir / f"{circuit_name.lower().replace(' ', '_')}_models.pkl"
        
        circuit_data = {
            'circuit_name': circuit_name,
            'models': models,
            'processed_data_summary': {
                'total_laps': len(processed_data),
                'compounds': processed_data['Compound'].value_counts().to_dict(),
                'drivers': processed_data['Driver'].nunique(),
                'stint_range': [int(processed_data['StintLap'].min()), int(processed_data['StintLap'].max())]
            },
            'created_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(circuit_file, 'wb') as f:
            pickle.dump(circuit_data, f)
        
        print(f"  Saved models to {circuit_file}")
    
    def prebuild_all_models(self):
        """Prebuild models for all circuits"""
        print("Starting F1 tire model prebuilding process...")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        success_count = 0
        total_models = 0
        
        for circuit_name, circuit_info in self.circuits.items():
            print(f"\nProcessing {circuit_name} ({circuit_info['gp_name']})...")
            
            # Load and process data
            processed_data = self.load_and_process_f1_data(circuit_name)
            if processed_data is None:
                print(f"  Skipping {circuit_name} - no data available")
                continue
            
            # Build models for each compound
            models = {}
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                compound_data = processed_data[processed_data['Compound'] == compound]
                if len(compound_data) > 0:
                    model = self.build_bayesian_tire_model(compound_data, circuit_name, compound)
                    if model is not None:
                        models[compound] = model
                        total_models += 1
            
            if models:
                self.save_circuit_models(circuit_name, models, processed_data)
                success_count += 1
                print(f"  Successfully built {len(models)} models for {circuit_name}")
            else:
                print(f"  No models built for {circuit_name}")
        
        print(f"\n Prebuilding complete!")
        print(f"Successfully processed {success_count}/{len(self.circuits)} circuits")
        print(f"Built {total_models} total tire models")
        print(f"Models saved in: {self.output_dir.absolute()}")
        
        # Create a summary file
        self.create_summary_file(success_count, total_models)
    
    def create_summary_file(self, success_count, total_models):
        """Create a summary of all prebuilt models"""
        summary = {
            'prebuilding_timestamp': pd.Timestamp.now().isoformat(),
            'circuits_processed': success_count,
            'total_circuits': len(self.circuits),
            'total_models': total_models,
            'model_files': []
        }
        
        # List all model files
        for model_file in self.output_dir.glob("*_models.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                summary['model_files'].append({
                    'filename': model_file.name,
                    'circuit': data['circuit_name'],
                    'compounds': list(data['models'].keys()),
                    'total_laps': data['processed_data_summary']['total_laps']
                })
            except:
                pass
        
        summary_file = self.output_dir / "models_summary.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    if not DEPENDENCIES_AVAILABLE:
        print("Required dependencies not available!")
        print("Please install: pip install fastf1 jax numpyro")
        exit(1)
    
    prebuilder = F1ModelPrebuilder()
    prebuilder.prebuild_all_models()