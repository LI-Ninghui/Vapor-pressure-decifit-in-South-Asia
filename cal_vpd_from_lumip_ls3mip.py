# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 10:44:08 2026

@author: drnin
"""


import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

#%%

def calculate_vpd(tas: xr.DataArray, huss: xr.DataArray, ps: xr.DataArray) -> xr.DataArray:
    """
    Calculate Vapor Pressure Deficit (VPD) using temperature, specific humidity, and surface pressure.
    
    Parameters:
        tas (xr.DataArray): Near-surface air temperature (units: K)
        huss (xr.DataArray): Near-surface specific humidity (units: kg/kg)
        ps (xr.DataArray): Surface air pressure (units: Pa)
    
    Returns:
        xr.DataArray: Vapor Pressure Deficit (units: kPa)
    
    Calculation Logic:
        1. Saturation vapor pressure (e_s) via Tetens' formula (requires °C)
        2. Actual vapor pressure (e_a) using specific humidity and pressure
        3. VPD = e_s - e_a
    """
    # Convert units: Kelvin → Celsius, Pascals → kilopascals
    tas_c = tas - 273.15  # Temperature in °C
    ps_kpa = ps / 1000    # Pressure in kPa

    # Calculate saturation vapor pressure (e_s) using Tetens' formula
    # e_s = 0.61078 * np.exp((17.27 * tas_c) / (tas_c + 237.3))  # Units: kPa
    # e_s = 0.611 * np.exp((17.5 * tas_c) / (tas_c + 237.3))  # Units: kPa
    e_s = 0.611 * np.exp((17.5 * tas_c) / (tas_c + 240.978))

    # Calculate actual vapor pressure (e_a) using specific humidity and pressure
    # Formula: e_a = (huss * P) / (0.622 + huss)
    e_a = (huss * ps_kpa) / (0.622 + huss)  # Units: kPa

    # Compute VPD (saturation deficit)
    vpd = e_s - e_a

    # Add metadata following CF conventions
    vpd.attrs = {
        "long_name": "Vapor Pressure Deficit",
        # "standard_name": "vapor_pressure_deficit",
        "units": "kPa",
        # "calculation_method": "Tetens' formula (e_s) + specific humidity-pressure formula (e_a)",
        # "input_variables": "tas (air_temperature), huss (specific_humidity), ps (surface_pressure)"
    }

    return vpd

#%%
def process_forcing(forcing_name: str, data_dir: str | Path, models: list[str]) -> xr.Dataset:
    """
    Process a single forcing experiment: merge variables (tas/huss/ps) → calculate VPD for all models.
    
    Parameters:
        forcing_name (str): Name of the forcing experiment (e.g., "hist-aer", "hist-GHG")
        data_dir (str | Path): Directory containing input NetCDF files
        models (list[str]): List of model names to process (e.g., ["ACCESS-ESM1-5", ...])
    
    Returns:
        xr.Dataset: VPD dataset for all models under the specified forcing
    """
    # Convert to Path object for OS-agnostic path handling
    data_dir = Path(data_dir)
    vpd_data = {}  # Store VPD DataArrays for each model

    # Define file paths for tas, huss, ps (follow your naming convention)
    file_paths = {
        "tas": data_dir / f"lumip_single_{forcing_name}_tas.nc",
        "huss": data_dir / f"lumip_single_{forcing_name}_huss.nc",
        "ps": data_dir / f"lumip_single_{forcing_name}_ps.nc"
    }

    # Validate all required files exist
    for var, path in file_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file for variable {var}: {path}")

    # Open input datasets (lazy loading, no data loaded into memory yet)
    print(f"Processing forcing experiment: {forcing_name}")
    ds_tas = xr.open_dataset(file_paths["tas"])
    ds_huss = xr.open_dataset(file_paths["huss"])
    ds_ps = xr.open_dataset(file_paths["ps"])

    # Calculate VPD for each model individually
    for model in models:
        print(f"  Calculating VPD for model: {model}")
        
        # Extract model-specific data from each variable dataset
        # Assumption: Each model is stored as a separate data variable in the NetCDF
        tas = ds_tas[model]  # Dimensions: [time_{model}, lat, lon]
        huss = ds_huss[model]
        ps = ds_ps[model]

        # Get the model-specific time dimension name (e.g., "time_ACCESS-ESM1-5")
        time_dim = tas.dims[0]

        # Align dimensions across variables (critical for calculation)
        # Rename time dimension to match tas and interpolate to tas's grid
        huss_aligned = huss.rename({huss.dims[0]: time_dim}).interp_like(tas)
        ps_aligned = ps.rename({ps.dims[0]: time_dim}).interp_like(tas)

        # Calculate VPD using the aligned variables
        vpd = calculate_vpd(tas=tas, huss=huss_aligned, ps=ps_aligned)
        
        # Store VPD with model name as the data variable key
        vpd_data[model] = vpd

    # Compile all model VPDs into a single Dataset
    # Reuse coordinates (lat, lon, and model-specific times) from input datasets
    ds_vpd = xr.Dataset(
                        data_vars=vpd_data,
                        coords={
                            "lat": ds_tas.lat,  # Reuse latitude coordinate
                            "lon": ds_tas.lon,  # Reuse longitude coordinate
                            # Collect model-specific time coordinates from tas dataset
                            **{ds_tas[model].dims[0]: ds_tas[model][ds_tas[model].dims[0]] for model in models}
                        },
                        attrs={
                            "experiment_id": forcing_name,
                            "variable": "vpd",
                            "title": f"Multi-model VPD Dataset for {forcing_name} Forcing Experiment",
                            # "models_included": ", ".join(models),
                            # "creation_time": pd.Timestamp.now().isoformat(),
                            # "processing_script": "vpd_calculation.py",
                            # "reference": "Tetens, O. (1930). Über einige meteorologische Begriffe. Zeitschrift für Geophysik, 6(2), 297-309."
                        }
                        )
    

    # ------------------------------
    # Compute multi-model mean
    # ------------------------------
    # List of model DataArrays
    vpd_list = [ds_vpd[model] for model in models]
    
    # Initialize sum and count arrays
    vpd_sum = None
    vpd_count = 0
    
    for da in vpd_list:
        # Convert to float32 to save memory (optional)
        da = da.values
        
        # Mask NaNs
        da_masked = da
        
        if vpd_sum is None:
            vpd_sum = da_masked
            vpd_count += 1 
        else:
            vpd_sum = vpd_sum + da_masked
            vpd_count += 1
    
    # Compute mean safely
    vpd_mean = vpd_sum / vpd_count
    
    # Assign to dataset
    ds_vpd["vpd_mean"] = xr.DataArray(
                                    vpd_mean.data,               # the actual array
                                    dims=ds_vpd[models[0]].dims,          # explicitly specify dims
                                    coords={
                                        "time": ds_vpd[models[0]].coords[ds_vpd[models[0]].dims[0]],
                                        "lat": ds_vpd[models[0]].lat,
                                        "lon": ds_vpd[models[0]].lon
                                    },
                                    attrs={
                                        "long_name": "Multi-model mean Vapor Pressure Deficit",
                                        "units": "kPa"
                                    }
                                )

    # Save VPD dataset to NetCDF with compression (reduces file size)
    output_path = data_dir / f"lumip_single_{forcing_name}_vpd.nc"
    encoding = {model: {"zlib": True, "complevel": 3} for model in models}  # zlib compression (level 3/9)
    ds_vpd.to_netcdf(output_path, encoding=encoding)
    print(f"  Successfully saved VPD dataset: {output_path}\n")

    # Close input datasets to free memory
    ds_tas.close()
    ds_huss.close()
    ds_ps.close()

    return ds_vpd

#%%
# -----------------------------------------------------------------------------
# Main execution workflow
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration parameters (update these according to your data)
    DATA_DIR = "E:/LUMIP_DAMIP_CMIP/mergerd_data_irri"  # Replace with your actual data path
    
    FORCING_LIST = ["land-crop-noIrrig", "land-hist"]  # All forcing experiments "hist-nolu", 
    
    MODELS = ["CESM2"] # Replace with your full model list
    

    # Batch process all forcing experiments
    for forcing in FORCING_LIST:
        vpd = process_forcing(
                            forcing_name=forcing,
                            data_dir=DATA_DIR,
                            models=MODELS
                            )
        
        
        
        
        