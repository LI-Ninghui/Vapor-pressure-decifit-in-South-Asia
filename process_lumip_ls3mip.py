# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 10:23:46 2026

@author: drnin
"""

import os
import warnings
import numpy as np
import xarray as xr

import pandas as pd
from xarray.coding.times import CFDatetimeCoder


# ============================================================
# Helper Functions
# Filter r1i1p1f1 member files (supports single-year & multi-year files)
# ============================================================
def find_all(model, exp, root, member="r1i1p1f1"):
    """
    Locate all .nc files for the specified model, experiment, and ensemble member.
    Supports both single-year files (2015.nc, 2016.nc, ...) and multi-year segments.
    
    Parameters:
        model (str): Name of the CMIP6 model
        exp (str): Name of the ScenarioMIP experiment (e.g., ssp126)
        root (str): Root directory of the ScenarioMIP tas data
        member (str): Target ensemble member (default: r1i1p1f1)
    
    Returns:
        list: Sorted list of full paths to the matching .nc files (chronological order)
    """
    paths = []
    target_dir = os.path.join(root, exp, model)
    # Traverse all subdirectories to find target .nc files
    for r, _, files in os.walk(target_dir):
        for f in files:
            # Filter files containing target model, experiment and member (avoid irrelevant files)
            if (f.endswith(".nc") and 
                model in f and 
                exp in f and 
                member in f):
                paths.append(os.path.join(r, f))
    # Sort paths to ensure chronological order (single-year files will be loaded 2015 → 2099)
    return sorted(paths)

#%%

# ============================================================
# Suppress Redundant SerializationWarning (Time Decoding)
# ============================================================
warnings.filterwarnings(
    "ignore",
    category=xr.SerializationWarning,
    message="Unable to decode time axis into full numpy.datetime64 objects"
)

# ============================================================
# Configuration Settings
# ============================================================
root = r"E:\LUMIP_LS3MIP\tas"

models = ["CESM2"]

experiments = ["land-crop-noIrrig", "land-hist"] #, "ssp245", "ssp370", "ssp585"

TARGET_START_YEAR = 1960  # Integer for robust year comparison
TARGET_END_YEAR = 2014

# ERA5-like target grid (adjust the resolution as needed)
lon_new = np.arange(50, 120+1.25, 1.25)
lat_new = np.arange(60, -10-1.25, -1.25)

# Output container to store processed tas data (key: experiment, subkey: model)
mip_tas = {exp: {} for exp in experiments}

# ============================================================
# Main Processing Loop
# Merge single-year/multi-year files for single member (r1i1p1f1) and regrid
# ============================================================
for model in models:
    for exp in experiments:

        # Find all r1i1p1f1 files for current model and experiment
        paths = find_all(model, exp, root, member="r1i1p1f1")

        if not paths:
            print(f"⚠ No files found: {model} {exp}")
            continue

        print(f"📂 {model} — {exp} | total files (single-year/multi-year): {len(paths)}")

        ds_list = []

        for p in paths:
            
            # Open netCDF file with explicit cftime support (for long time ranges)
            ds = xr.open_dataset(
                p,
                use_cftime=True
            )

            # Time subsetting to target period (reduce redundant data before concatenation)
            ds = ds.sel(time=slice(str(TARGET_START_YEAR), str(TARGET_END_YEAR)))

            # Skip files with no valid time data in the target period
            if ds.time.size == 0:
                print(f"⚠ rejected {os.path.basename(p)} (no valid time in {TARGET_START_YEAR}-{TARGET_END_YEAR})")
                ds.close()
                continue

            # Validate single-year/multi-year files (reject only files completely outside target)
            file_start_year = int(ds.time.dt.year.min())
            file_end_year = int(ds.time.dt.year.max())
            
            if (file_end_year < TARGET_START_YEAR) or (file_start_year > TARGET_END_YEAR):
                print(f"⚠ rejected {os.path.basename(p)} (file time range: {file_start_year}-{file_end_year}, outside target)")
                ds.close()
                continue
            
            # Optional: Info log for single-year files (for transparency)
            if file_start_year == file_end_year:
                print(f" processing single-year file: {os.path.basename(p)} (year: {file_start_year})")

            # Normalize coordinate names (unify 'longitude'/'latitude' to 'lon'/'lat')
            if "longitude" in ds:
                ds = ds.rename({"longitude": "lon"})
            if "latitude" in ds:
                ds = ds.rename({"latitude": "lat"})

            ds_list.append(ds)

        if not ds_list:
            continue

        # ----------------------------------------------------
        # Core Fix: Concatenate without relying on squeeze() (supports 1+ valid files)
        # ----------------------------------------------------
        # Option 1: Concatenate directly along time (more logical for time-series data)
        # Use "time" as the concatenation dim (xarray auto-aligns time coordinates)
        ds_merged = xr.concat(
            ds_list,
            dim="time",  # Concatenate along time dimension (no temporary "file" dim)
            join="inner",
            compat="broadcast_equals"  # Loose attribute validation for single-year files
        )

        # Critical: Sort by time to ensure continuous 2015 → 2099 time series
        # Fixes any out-of-order single-year files
        ds_merged = ds_merged.sortby("time")

        # ----------------------------------------------------
        # Select 'tas' variable and target geographic region
        # ----------------------------------------------------
        tas = (
            ds_merged["tas"]
            .sel(
                time=slice(str(TARGET_START_YEAR), str(TARGET_END_YEAR)),
                lat=slice(-10, 60),
                lon=slice(50, 120)
            )
            .sortby("lat", ascending=False)  # Sort latitude from north to south
        )

        # ----------------------------------------------------
        # Regrid to ERA5-like target grid (linear interpolation)
        # ----------------------------------------------------
        tas_reg = tas.interp(
            lat=lat_new,
            lon=lon_new,
            method="linear"
        )

        # Store the processed and regridded tas data
        mip_tas[exp][model] = tas_reg

        print(f" processed, merged (continuous time series) and regridded successfully")

        # Close all open datasets to release memory (critical for 85+ single-year files)
        for ds in ds_list:
            ds.close()
        ds_merged.close()

print(f" ScenarioMIP tas data processing complete (supports single-year files {TARGET_START_YEAR}-{TARGET_END_YEAR})")


#%%

# Your original configuration (unchanged)
models = ["CESM2"]

experiments = ["land-crop-noIrrig", "land-hist"] #, "ssp245", "ssp370", "ssp585"] #, "ssp245", "ssp370", "ssp585"

ref_model = "CESM2"

# Create save directory if it does not exist (avoid FileNotFoundError)
save_dir = r"E:\LUMIP_DAMIP_CMIP\mergerd_data_irri"
os.makedirs(save_dir, exist_ok=True)

# Main loop to merge and save data (fixed for old xarray versions: DataArray without drop_coords)
for experiment in experiments:
    # ======================
    # Step 1: Validate reference model existence (avoid KeyError)
    # ======================
    if experiment not in mip_tas or ref_model not in mip_tas[experiment]:
        print(f" Skip {experiment}: Reference model {ref_model} or experiment not found in mip_tas")
        continue
    
    # Extract reference lon/lat (your original logic, retained)
    lon_ds = mip_tas[experiment][ref_model].lon
    lat_ds = mip_tas[experiment][ref_model].lat
    
    # ======================
    # Step 2: Create unified time axis (your original 1960-2014 monthly)
    # ======================
    time_common = pd.date_range(start="1960-01-01", end="2014-12-31", freq="MS")

    # ======================
    # Step 3: Prepare multi-model data variables (fixed for old xarray versions)
    # ======================
    data_vars = {}
    
    for model in models:
        if model not in mip_tas[experiment]:
            print(f" Skip {model} in {experiment}: No processed data available")
            continue

        # Copy DataArray to avoid modifying original mip_tas
        da = mip_tas[experiment][model].copy()
        
        # ======================
        # Core Fix: Remove unwanted coordinates for DataArray (compatible with all xarray versions)
        # Use reset_coords() instead of drop_coords() (drop_coords() is not available for DataArray in old versions)
        # ======================
        aux_coords_to_drop = ["height", "bounds", "time_bnds"]
        for v in aux_coords_to_drop:
            if v in da.coords:  # Check if the coordinate exists in the DataArray
                # reset_coords(): Move coordinate to data variable (drop=False) or delete it (drop=True)
                da = da.reset_coords(names=[v], drop=True)

        # ======================
        # Fix 2: Robustly rename time dimension (avoid relying on da.dims[0])
        # ======================
        # Find time dimension (handle case-insensitive, more robust than da.dims[0])
        time_dim = None
        for dim in da.dims:
            if "time" in dim.lower():
                time_dim = dim
                break
        # Default to "time" if no time dimension found (prevent error)
        time_dim = time_dim or "time"
        
        if time_dim != "time":
            da = da.rename({time_dim: "time"})
        
        # ======================
        # Fix 3: Friendly time size validation (replace raw assert with error handling)
        # ======================
        try:
            assert da.sizes["time"] == len(time_common), \
                f"Time size mismatch: {model} has {da.sizes['time']} steps, expected {len(time_common)}"
        except (AssertionError, KeyError) as e:
            print(f" Skip {model} in {experiment}: {str(e)}")
            continue
        
        # Align to unified time axis (your original logic, retained)
        da = da.assign_coords(time=time_common)
        
        # Add DataArray to data variables (model name as variable key)
        data_vars[model] = da
    
    # ======================
    # Step 4: Skip if no valid data variables
    # ======================
    if not data_vars:
        print(f" Skip {experiment}: No valid model data to merge")
        continue
    
    # ======================
    # Step 5: Create unified Dataset (revised attrs)
    # ======================
    ds_tas = xr.Dataset(
                        data_vars=data_vars,
                        coords={
                            "time": time_common,
                            "lat": lat_ds,
                            "lon": lon_ds
                              },
                        attrs={
                            "experiment": experiment,
                            "variable": "tas",  # Fixed typo (from "ps" to "tas")
                            "time_range": "1960-01-01 to 2014-12-31",
                            "time_freq": "Monthly (MS)",
                            "reference_model": ref_model,
                            "processed_models": list(data_vars.keys()),
                            "creation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                       )  

    # ======================
    # Step 6: Verify and save to NetCDF
    # ======================
    # Print Dataset info for validation
    print(f"\n Unified Dataset for {experiment}:")
    print(ds_tas)
    
    # Define save path
    save_filename = f"lumip_single_{experiment}_tas.nc"
    save_path = os.path.join(save_dir, save_filename)
    
    # Save to NetCDF (add optional compression to reduce file size)
    try:
        ds_tas.to_netcdf(
            save_path,
            engine="netcdf4",
            encoding={model: {"zlib": True, "complevel": 4} for model in data_vars.keys()}
        )
        print(f"📁 Successfully saved to: {save_path}\n")
    except Exception as e:
        print(f" Failed to save {experiment}: {str(e)}\n")
    
    # Release memory
    del ds_tas
    
    
#%%
# ============================================================
# Configuration Settings
# ============================================================
# root = r"E:\ScenarioMIP\huss"
root = r"E:\LUMIP_LS3MIP\huss"

models = ["CESM2"]

experiments = ["land-crop-noIrrig", "land-hist"] #, "ssp245", "ssp370", "ssp585"] #, "ssp245", "ssp370", "ssp585"

TARGET_START_YEAR = 1960  # Integer for robust year comparison
TARGET_END_YEAR = 2014

# ERA5-like target grid (adjust the resolution as needed)
lon_new = np.arange(50, 120+1.25, 1.25)
lat_new = np.arange(60, -10-1.25, -1.25)

# Output container to store processed huss data (key: experiment, subkey: model)
mip_huss = {exp: {} for exp in experiments}

# ============================================================
# Main Processing Loop
# Merge single-year/multi-year files for single member (r1i1p1f1) and regrid
# ============================================================
for model in models:
    for exp in experiments:

        # Find all r1i1p1f1 files for current model and experiment
        paths = find_all(model, exp, root, member="r1i1p1f1")

        if not paths:
            print(f"⚠ No files found: {model} {exp}")
            continue

        print(f"📂 {model} — {exp} | total files (single-year/multi-year): {len(paths)}")

        ds_list = []

        for p in paths:
            
            # Open netCDF file with explicit cftime support (for long time ranges)
            ds = xr.open_dataset(
                p,
                # decode_times=CFDatetimeCoder(use_cftime=True),
                use_cftime=True
            )

            # Time subsetting to target period (reduce redundant data before concatenation)
            ds = ds.sel(time=slice(str(TARGET_START_YEAR), str(TARGET_END_YEAR)))

            # Skip files with no valid time data in the target period
            if ds.time.size == 0:
                print(f"⚠ rejected {os.path.basename(p)} (no valid time in {TARGET_START_YEAR}-{TARGET_END_YEAR})")
                ds.close()
                continue

            # Validate single-year/multi-year files (reject only files completely outside target)
            file_start_year = int(ds.time.dt.year.min())
            file_end_year = int(ds.time.dt.year.max())
            
            if (file_end_year < TARGET_START_YEAR) or (file_start_year > TARGET_END_YEAR):
                print(f"⚠ rejected {os.path.basename(p)} (file time range: {file_start_year}-{file_end_year}, outside target)")
                ds.close()
                continue
            
            # Optional: Info log for single-year files (for transparency)
            if file_start_year == file_end_year:
                print(f" processing single-year file: {os.path.basename(p)} (year: {file_start_year})")

            # Normalize coordinate names (unify 'longitude'/'latitude' to 'lon'/'lat')
            if "longitude" in ds:
                ds = ds.rename({"longitude": "lon"})
            if "latitude" in ds:
                ds = ds.rename({"latitude": "lat"})

            ds_list.append(ds)

        if not ds_list:
            continue

        # ----------------------------------------------------
        # Core Fix: Concatenate without relying on squeeze() (supports 1+ valid files)
        # ----------------------------------------------------
        # Option 1: Concatenate directly along time (more logical for time-series data)
        # Use "time" as the concatenation dim (xarray auto-aligns time coordinates)
        ds_merged = xr.concat(
            ds_list,
            dim="time",  # Concatenate along time dimension (no temporary "file" dim)
            join="inner",
            compat="broadcast_equals"  # Loose attribute validation for single-year files
        )

        # Critical: Sort by time to ensure continuous 2015 → 2099 time series
        # Fixes any out-of-order single-year files
        ds_merged = ds_merged.sortby("time")

        # ----------------------------------------------------
        # Select 'tas' variable and target geographic region
        # ----------------------------------------------------
        huss = (
            ds_merged["huss"]
            .sel(
                time=slice(str(TARGET_START_YEAR), str(TARGET_END_YEAR)),
                lat=slice(-10, 60),
                lon=slice(50, 120)
            )
            .sortby("lat", ascending=False)  # Sort latitude from north to south
        )

        # ----------------------------------------------------
        # Regrid to ERA5-like target grid (linear interpolation)
        # ----------------------------------------------------
        huss_reg = huss.interp(
            lat=lat_new,
            lon=lon_new,
            method="linear"
        )

        # Store the processed and regridded huss data
        mip_huss[exp][model] = huss_reg

        print(f" processed, merged (continuous time series) and regridded successfully")

        # Close all open datasets to release memory (critical for 85+ single-year files)
        for ds in ds_list:
            ds.close()
        ds_merged.close()

print(f" ScenarioMIP tas data processing complete (supports single-year files {TARGET_START_YEAR}-{TARGET_END_YEAR})")


#%%

# Your original configuration (unchanged)
models = ["CESM2"]

experiments = ["land-crop-noIrrig", "land-hist"] #, "ssp245", "ssp370", "ssp585"] #, "ssp245", "ssp370", "ssp585"

ref_model = "CESM2"

# Create save directory if it does not exist (avoid FileNotFoundError)
save_dir = r"E:\LUMIP_DAMIP_CMIP\mergerd_data_irri"
os.makedirs(save_dir, exist_ok=True)

# Main loop to merge and save data (fixed for old xarray versions: DataArray without drop_coords)
for experiment in experiments:
    # ======================
    # Step 1: Validate reference model existence (avoid KeyError)
    # ======================
    if experiment not in mip_huss or ref_model not in mip_huss[experiment]:
        print(f" Skip {experiment}: Reference model {ref_model} or experiment not found in mip_huss")
        continue
    
    # Extract reference lon/lat (your original logic, retained)
    lon_ds = mip_huss[experiment][ref_model].lon
    lat_ds = mip_huss[experiment][ref_model].lat
    
    # ======================
    # Step 2: Create unified time axis (your original 1960-2014 monthly)
    # ======================
    time_common = pd.date_range(start="1960-01-01", end="2014-12-31", freq="MS")

    # ======================
    # Step 3: Prepare multi-model data variables (fixed for old xarray versions)
    # ======================
    data_vars = {}
    
    for model in models:
        if model not in mip_huss[experiment]:
            print(f" Skip {model} in {experiment}: No processed data available")
            continue

        # Copy DataArray to avoid modifying original mip_huss
        da = mip_huss[experiment][model].copy()
        
        # ======================
        # Core Fix: Remove unwanted coordinates for DataArray (compatible with all xarray versions)
        # Use reset_coords() instead of drop_coords() (drop_coords() is not available for DataArray in old versions)
        # ======================
        aux_coords_to_drop = ["height", "bounds", "time_bnds"]
        for v in aux_coords_to_drop:
            if v in da.coords:  # Check if the coordinate exists in the DataArray
                # reset_coords(): Move coordinate to data variable (drop=False) or delete it (drop=True)
                da = da.reset_coords(names=[v], drop=True)

        # ======================
        # Fix 2: Robustly rename time dimension (avoid relying on da.dims[0])
        # ======================
        # Find time dimension (handle case-insensitive, more robust than da.dims[0])
        time_dim = None
        for dim in da.dims:
            if "time" in dim.lower():
                time_dim = dim
                break
        # Default to "time" if no time dimension found (prevent error)
        time_dim = time_dim or "time"
        
        if time_dim != "time":
            da = da.rename({time_dim: "time"})
        
        # ======================
        # Fix 3: Friendly time size validation (replace raw assert with error handling)
        # ======================
        try:
            assert da.sizes["time"] == len(time_common), \
                f"Time size mismatch: {model} has {da.sizes['time']} steps, expected {len(time_common)}"
        except (AssertionError, KeyError) as e:
            print(f" Skip {model} in {experiment}: {str(e)}")
            continue
        
        # Align to unified time axis (your original logic, retained)
        da = da.assign_coords(time=time_common)
        
        # Add DataArray to data variables (model name as variable key)
        data_vars[model] = da
    
    # ======================
    # Step 4: Skip if no valid data variables
    # ======================
    if not data_vars:
        print(f" Skip {experiment}: No valid model data to merge")
        continue
    
    # ======================
    # Step 5: Create unified Dahusset (revised attrs)
    # ======================
    ds_huss = xr.Dataset(
                        data_vars=data_vars,
                        coords={
                            "time": time_common,
                            "lat": lat_ds,
                            "lon": lon_ds
                              },
                        attrs={
                            "experiment": experiment,
                            "variable": "huss",  # Fixed typo (from "ps" to "huss")
                            "time_range": "1960-01-01 to 2014-12-31",
                            "time_freq": "Monthly (MS)",
                            "reference_model": ref_model,
                            "processed_models": list(data_vars.keys()),
                            "creation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                       )  

    # ======================
    # Step 6: Verify and save to NetCDF
    # ======================
    # Print Dataset info for validation
    print(f"\n Unified Dahusset for {experiment}:")
    print(ds_huss)
    
    # Define save path
    save_filename = f"lumip_single_{experiment}_huss.nc"
    save_path = os.path.join(save_dir, save_filename)
    
    # Save to NetCDF (add optional compression to reduce file size)
    try:
        ds_huss.to_netcdf(
            save_path,
            engine="netcdf4",
            encoding={model: {"zlib": True, "complevel": 4} for model in data_vars.keys()}
        )
        print(f"📁 Successfully saved to: {save_path}\n")
    except Exception as e:
        print(f" Failed to save {experiment}: {str(e)}\n")
    
    # Release memory
    del ds_huss
    
#%%
# ============================================================
# Configuration Settings
# ============================================================
# root = r"E:\ScenarioMIP\huss"
root = r"E:\LUMIP_LS3MIP\ps"

models = ["CESM2"]

experiments = ["land-crop-noIrrig", "land-hist"] #, "ssp245", "ssp370", "ssp585"] #, "ssp245", "ssp370", "ssp585"

TARGET_START_YEAR = 1960  # Integer for robust year comparison
TARGET_END_YEAR = 2014

# ERA5-like target grid (adjust the resolution as needed)
lon_new = np.arange(50, 120+1.25, 1.25)
lat_new = np.arange(60, -10-1.25, -1.25)

# Output container to store processed ps data (key: experiment, subkey: model)
mip_ps = {exp: {} for exp in experiments}

# ============================================================
# Main Processing Loop
# Merge single-year/multi-year files for single member (r1i1p1f1) and regrid
# ============================================================
for model in models:
    for exp in experiments:

        # Find all r1i1p1f1 files for current model and experiment
        paths = find_all(model, exp, root, member="r1i1p1f1")

        if not paths:
            print(f"⚠ No files found: {model} {exp}")
            continue

        print(f"📂 {model} — {exp} | total files (single-year/multi-year): {len(paths)}")

        ds_list = []

        for p in paths:
            
            # Open netCDF file with explicit cftime support (for long time ranges)
            ds = xr.open_dataset(
                p,
                # decode_times=CFDatetimeCoder(use_cftime=True),
                use_cftime=True
            )

            # Time subsetting to target period (reduce redundant data before concatenation)
            ds = ds.sel(time=slice(str(TARGET_START_YEAR), str(TARGET_END_YEAR)))

            # Skip files with no valid time data in the target period
            if ds.time.size == 0:
                print(f"⚠ rejected {os.path.basename(p)} (no valid time in {TARGET_START_YEAR}-{TARGET_END_YEAR})")
                ds.close()
                continue

            # Validate single-year/multi-year files (reject only files completely outside target)
            file_start_year = int(ds.time.dt.year.min())
            file_end_year = int(ds.time.dt.year.max())
            
            if (file_end_year < TARGET_START_YEAR) or (file_start_year > TARGET_END_YEAR):
                print(f"⚠ rejected {os.path.basename(p)} (file time range: {file_start_year}-{file_end_year}, outside target)")
                ds.close()
                continue
            
            # Optional: Info log for single-year files (for transparency)
            if file_start_year == file_end_year:
                print(f" processing single-year file: {os.path.basename(p)} (year: {file_start_year})")

            # Normalize coordinate names (unify 'longitude'/'latitude' to 'lon'/'lat')
            if "longitude" in ds:
                ds = ds.rename({"longitude": "lon"})
            if "latitude" in ds:
                ds = ds.rename({"latitude": "lat"})

            ds_list.append(ds)

        if not ds_list:
            continue

        # ----------------------------------------------------
        # Core Fix: Concatenate without relying on squeeze() (supports 1+ valid files)
        # ----------------------------------------------------
        # Option 1: Concatenate directly along time (more logical for time-series data)
        # Use "time" as the concatenation dim (xarray auto-aligns time coordinates)
        ds_merged = xr.concat(
            ds_list,
            dim="time",  # Concatenate along time dimension (no temporary "file" dim)
            join="inner",
            compat="broadcast_equals"  # Loose attribute validation for single-year files
        )

        # Critical: Sort by time to ensure continuous 2015 → 2099 time series
        # Fixes any out-of-order single-year files
        ds_merged = ds_merged.sortby("time")

        # ----------------------------------------------------
        # Select 'tas' variable and target geographic region
        # ----------------------------------------------------
        ps = (
            ds_merged["ps"]
            .sel(
                time=slice(str(TARGET_START_YEAR), str(TARGET_END_YEAR)),
                lat=slice(-10, 60),
                lon=slice(50, 120)
            )
            .sortby("lat", ascending=False)  # Sort latitude from north to south
        )

        # ----------------------------------------------------
        # Regrid to ERA5-like target grid (linear interpolation)
        # ----------------------------------------------------
        ps_reg = ps.interp(
            lat=lat_new,
            lon=lon_new,
            method="linear"
        )

        # Store the processed and regridded huss data
        mip_ps[exp][model] = ps_reg

        print(f" processed, merged (continuous time series) and regridded successfully")

        # Close all open datasets to release memory (critical for 85+ single-year files)
        for ds in ds_list:
            ds.close()
        ds_merged.close()

print(f" ScenarioMIP tas data processing complete (supports single-year files {TARGET_START_YEAR}-{TARGET_END_YEAR})")


#%%

# Your original configuration (unchanged)
models = ["CESM2"]

experiments = ["land-crop-noIrrig", "land-hist"] #, "ssp245", "ssp370", "ssp585"] #, "ssp245", "ssp370", "ssp585"

ref_model = "CESM2"

# Create save directory if it does not exist (avoid FileNotFoundError)
save_dir = r"E:\LUMIP_DAMIP_CMIP\mergerd_data_irri"
os.makedirs(save_dir, exist_ok=True)

# Main loop to merge and save data (fixed for old xarray versions: DataArray without drop_coords)
for experiment in experiments:
    # ======================
    # Step 1: Validate reference model existence (avoid KeyError)
    # ======================
    if experiment not in mip_ps or ref_model not in mip_ps[experiment]:
        print(f" Skip {experiment}: Reference model {ref_model} or experiment not found in mip_ps")
        continue
    
    # Extract reference lon/lat (your original logic, retained)
    lon_ds = mip_ps[experiment][ref_model].lon
    lat_ds = mip_ps[experiment][ref_model].lat
    
    # ======================
    # Step 2: Create unified time axis (your original 1960-2014 monthly)
    # ======================
    time_common = pd.date_range(start="1960-01-01", end="2014-12-31", freq="MS")

    # ======================
    # Step 3: Prepare multi-model data variables (fixed for old xarray versions)
    # ======================
    data_vars = {}
    
    for model in models:
        if model not in mip_ps[experiment]:
            print(f" Skip {model} in {experiment}: No processed data available")
            continue

        # Copy DataArray to avoid modifying original mip_ps
        da = mip_ps[experiment][model].copy()
        
        # ======================
        # Core Fix: Remove unwanted coordinates for DataArray (compatible with all xarray versions)
        # Use reset_coords() instead of drop_coords() (drop_coords() is not available for DataArray in old versions)
        # ======================
        aux_coords_to_drop = ["height", "bounds", "time_bnds"]
        for v in aux_coords_to_drop:
            if v in da.coords:  # Check if the coordinate exists in the DataArray
                # reset_coords(): Move coordinate to data variable (drop=False) or delete it (drop=True)
                da = da.reset_coords(names=[v], drop=True)

        # ======================
        # Fix 2: Robustly rename time dimension (avoid relying on da.dims[0])
        # ======================
        # Find time dimension (handle case-insensitive, more robust than da.dims[0])
        time_dim = None
        for dim in da.dims:
            if "time" in dim.lower():
                time_dim = dim
                break
        # Default to "time" if no time dimension found (prevent error)
        time_dim = time_dim or "time"
        
        if time_dim != "time":
            da = da.rename({time_dim: "time"})
        
        # ======================
        # Fix 3: Friendly time size validation (replace raw assert with error handling)
        # ======================
        try:
            assert da.sizes["time"] == len(time_common), \
                f"Time size mismatch: {model} has {da.sizes['time']} steps, expected {len(time_common)}"
        except (AssertionError, KeyError) as e:
            print(f" Skip {model} in {experiment}: {str(e)}")
            continue
        
        # Align to unified time axis (your original logic, retained)
        da = da.assign_coords(time=time_common)
        
        # Add DataArray to data variables (model name as variable key)
        data_vars[model] = da
    
    # ======================
    # Step 4: Skip if no valid data variables
    # ======================
    if not data_vars:
        print(f" Skip {experiment}: No valid model data to merge")
        continue
    
    # ======================
    # Step 5: Create unified Dapset (revised attrs)
    # ======================
    ds_ps = xr.Dataset(
                        data_vars=data_vars,
                        coords={
                            "time": time_common,
                            "lat": lat_ds,
                            "lon": lon_ds
                              },
                        attrs={
                            "experiment": experiment,
                            "variable": "ps",  # Fixed typo (from "ps" to "ps")
                            "time_range": "1960-01-01 to 2014-12-31",
                            "time_freq": "Monthly (MS)",
                            "reference_model": ref_model,
                            "processed_models": list(data_vars.keys()),
                            "creation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                       )  

    # ======================
    # Step 6: Verify and save to NetCDF
    # ======================
    # Print Dataset info for validation
    print(f"\n Unified Dapset for {experiment}:")
    print(ds_ps)
    
    # Define save path
    save_filename = f"lumip_single_{experiment}_ps.nc"
    save_path = os.path.join(save_dir, save_filename)
    
    # Save to NetCDF (add optional compression to reduce file size)
    try:
        ds_ps.to_netcdf(
            save_path,
            engine="netcdf4",
            encoding={model: {"zlib": True, "complevel": 4} for model in data_vars.keys()}
        )
        print(f"📁 Successfully saved to: {save_path}\n")
    except Exception as e:
        print(f" Failed to save {experiment}: {str(e)}\n")
    
    # Release memory
    del ds_ps