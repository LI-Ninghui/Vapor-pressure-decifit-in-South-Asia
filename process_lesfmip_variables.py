# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 21:30:37 2025

@author: drnin
"""

import os
import re
import warnings
import numpy as np
import xarray as xr

import pandas as pd

from xarray.coding.times import CFDatetimeCoder

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

# ============================================================
#  Helper functions
# ============================================================

def find_all(name1, name2, root):
    """
    Recursively find files containing both name1 and name2.
    """
    result = []
    for r, _, files in os.walk(root):
        for f in files:
            if (name1 in f) and (name2 in f) and f.endswith(".nc"):
                result.append(os.path.join(r, f))
    return sorted(result)


def extract_member(fname):
    """
    Extract CMIP ensemble member label (r*i*p*f*) from filename.
    """
    m = re.search(r"(r\d+i\d+p\d+f\d+)", fname)
    return m.group(1) if m else "unknown"


def preprocess_ds(ds):
    """
    Minimal, safe preprocessing for ensemble concatenation.
    DO NOT force dimension swaps.
    """

    # Drop auxiliary lon/lat variables ONLY if they are not dimensions
    drop_vars = []
    for v in ["lon_2", "lat_2"]:
        if v in ds.variables and v not in ds.dims:
            drop_vars.append(v)

    if drop_vars:
        ds = ds.drop_vars(drop_vars)

    return ds


#%%
# ============================================================
#  Configuration
# ============================================================

root = r"E:\LUMIP_DAMIP_CMIP\from_jw\tas"

models = [
        "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
        "CESM2",  "CNRM-CM6-1",
        "E3SM-2-0", "FGOALS-g3", "GISS-E2-1-G",
        "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MIROC6", 
        "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM"
        ] # "CMCC-CM2-SR5" is lacking

# models = [
#          "HadGEM3-GC31-LL",
#         ]

experiments = [
            "historical",
            "hist-aer",
            "hist-GHG",
            "hist-nat",
            "hist-noLu",
             ]

TARGET_START = "1960"
TARGET_END   = "2014"

# target grid (1.25° × 1.25°)
# lon_new = np.arange(-180, 180, 1.25)
# lat_new = np.arange(90, -90 - 1.25, -1.25)
lon_new = np.arange(50, 120+1.25, 1.25)
lat_new = np.arange(60, -10-1.25, -1.25)

# output container
mip_tas = {exp: {} for exp in experiments}


# ============================================================
#  Main processing loop
# ============================================================

for model in models:
    for exp in experiments:

        paths = find_all(model, exp, os.path.join(root, exp+"/"+model))

        if len(paths) == 0:
            print(f"⚠ No files found for {model} {exp}")
            continue

        print(f"📂 {model} — {exp} | Ensemble members: {len(paths)}")

        # ----------------------------------------------------
        # Open each ensemble member separately
        # ----------------------------------------------------
        ds_list = []
        members = []

        for p in paths:
            time_coder = CFDatetimeCoder(use_cftime=True)

            ds = xr.open_dataset(
                                p,
                                decode_times=time_coder,
                                # chunks={"time": 12}
                                )
            
            # Enforce the desired period
            ds = ds.sel(time=slice(TARGET_START, TARGET_END))
            
            if "lon_2" in ds.dims:
                ds = ds.drop_vars("lon")
                ds = ds.rename({"lon_2": "lon"})
        
            # get first and last available years
            t0 = ds.time.dt.year.min().item()
            t1 = ds.time.dt.year.max().item()
            
            # strict coverage check
            if (t0 > int(TARGET_START)) or (t1 < int(TARGET_END)):
                print(
                    f"⚠ {p} rejected: covers {t0}–{t1}, "
                    f"needs {TARGET_START}–{TARGET_END}"
                    )
                continue
            
            ds_list.append(ds)
            members.append(extract_member(p))

        # ----------------------------------------------------
        # Concatenate along ensemble dimension
        # ----------------------------------------------------
        
        ds_ens = xr.concat(ds_list, 
                           dim="ensemble", 
                           join="inner",
                           # coords="minimal",
                           # compat="override"
                           )
        
        ds_ens = ds_ens.assign_coords(ensemble=members)
        
        # ----------------------------------------------------
        # Select variable and time period
        # ----------------------------------------------------
        # tas = fix_duplicate_lonlat(ds_ens["tas"])
        
        tas = ds_ens["tas"].sel(time=slice(TARGET_START, TARGET_END),
                                lat=slice(-10, 60),
                                lon=slice(50, 120))
        
        tas = tas.sortby("lat", ascending=False)
        
        # lon = tas.lon
        # shift = (lon >= 180).sum().item()
        # tas = tas.roll(lon=-shift, roll_coords=True)
        # tas = tas.assign_coords(lon=((tas.lon + 180) % 360) - 180)
        # tas = tas.sortby("lon")
        
        # ----------------------------------------------------
        # Ensemble mean
        # ----------------------------------------------------
        tas_ensmean = tas.mean(dim="ensemble", skipna=True)

        # ----------------------------------------------------
        # Regrid to 1.25°
        # ----------------------------------------------------
        tas_1p25 = tas_ensmean.interp(
                                    lat=lat_new,
                                    lon=lon_new,
                                    method="linear",
                                    kwargs={"fill_value": "extrapolate"}
                                    )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------
        mip_tas[exp][model] = tas_1p25

        print(f"   ✓ Ensemble mean computed & regridded")

        # Optional: close datasets to free memory
        for ds in ds_list:
            ds.close()


# ============================================================
#  Example: Save one result
# ============================================================

# Example save (optional)
# mip_tas["hist-GHG"]["CESM2"].to_netcdf(
#     "tas_CESM2_hist-GHG_ensmean_1p25deg.nc"
# )

print("✅ All processing complete.")



experiments = ["historical", "hist-aer", "hist-GHG", "hist-nat", "hist-noLu"]

models = [
        "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
        "CESM2", "CNRM-CM6-1",
        "E3SM-2-0", "FGOALS-g3", "GISS-E2-1-G",
        "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MIROC6", 
        "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM"
        ]

# models = [
#         "ACCESS-ESM1-5"
#         ]

ref_model = "BCC-CSM2-MR"

for experiment in experiments:
    # 2. 提取参考经纬度（不变）
    lon_ds = mip_tas[experiment][ref_model].lon
    lat_ds = mip_tas[experiment][ref_model].lat
    
    # Determine common time range (intersection or target period)
    # Here we force all models to 1960-2014
    time_common = pd.date_range(start="1960-01-01", end="2014-12-31", freq="MS")

    data_vars = {}
    
    for model in models:
        if model not in mip_tas[experiment]:
            continue

        da = mip_tas[experiment][model].copy()
        
        # DROP conflicting auxiliary variables
        for v in ["height", "bounds", "time_bnds"]:
            if v in da.coords or v in da.coords:
                da = da.drop_vars(v)

        
        # Rename time dimension to "time"
        time_dim = da.dims[0]  # usually "time" already
        if time_dim != "time":
            da = da.rename({time_dim: "time"})
            
        assert da.sizes["time"] == len(time_common)
        da = da.assign_coords(time=time_common)
        
        data_vars[model] = da
        
    # Create unified Dataset
    ds_tas = xr.Dataset(
                        data_vars=data_vars,
                        coords={
                            "time": time_common,
                            "lat": lat_ds,
                            "lon": lon_ds
                              },
                        attrs={
                            "experiment": experiment,
                            "variable": "ps",
                            "models": models
                            }
                    )

    
    # 查看结果（确认维度无冲突）
    print(ds_tas)
    
    ds_tas.to_netcdf("E:/LUMIP_DAMIP_CMIP/merged_data/" + "mip_ensemble_" + experiment + "_tas.nc", engine="netcdf4") 
    
    del ds_tas


#%%
# ============================================================
#  huss
# ============================================================

root = r"E:\LUMIP_DAMIP_CMIP\from_jw\huss"

models = [
        "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
        "CESM2", "CNRM-CM6-1",
        "E3SM-2-0", "FGOALS-g3", "GISS-E2-1-G",
        "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MIROC6", 
        "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM"
        ]

# models = [
#          "GISS-E2-1-G",
#         ]

experiments = [
            "historical",
            "hist-aer",
            "hist-GHG",
            "hist-nat",
            "hist-noLu",
            ]

TARGET_START = "1960"
TARGET_END   = "2014"

# target grid (1.25° × 1.25°)
# lon_new = np.arange(-180, 180, 1.25)
# lat_new = np.arange(90, -90 - 1.25, -1.25)
lon_new = np.arange(50, 120+1.25, 1.25)
lat_new = np.arange(60, -10-1.25, -1.25)

# output container
mip_huss = {exp: {} for exp in experiments}


# ============================================================
#  Main processing loop
# ============================================================

for model in models:
    for exp in experiments:

        paths = find_all(model, exp, os.path.join(root, exp+"/"+model))

        if len(paths) == 0:
            print(f"⚠ No files found for {model} {exp}")
            continue

        print(f"📂 {model} — {exp} | Ensemble members: {len(paths)}")

        # ----------------------------------------------------
        # Open each ensemble member separately
        # ----------------------------------------------------
        ds_list = []
        members = []

        for p in paths:
            time_coder = CFDatetimeCoder(use_cftime=True)

            ds = xr.open_dataset(
                                p,
                                decode_times=time_coder,
                                # chunks={"time": 12}
                                )
            
            # Enforce the desired period
            ds = ds.sel(time=slice(TARGET_START, TARGET_END))
            
            
            if "lon_2" in ds.dims:
                ds = ds.drop_vars("lon")
                ds = ds.rename({"lon_2": "lon"})
            
            # get first and last available years
            t0 = ds.time.dt.year.min().item()
            t1 = ds.time.dt.year.max().item()
            
            # strict coverage check
            if (t0 > int(TARGET_START)) or (t1 < int(TARGET_END)):
                print(
                    f"⚠ {p} rejected: covers {t0}–{t1}, "
                    f"needs {TARGET_START}–{TARGET_END}"
                    )
                continue
            
            ds_list.append(ds)
            members.append(extract_member(p))

        # ----------------------------------------------------
        # Concatenate along ensemble dimension
        # ----------------------------------------------------
        ds_ens = xr.concat(ds_list, dim="ensemble", join="inner")
        ds_ens = ds_ens.assign_coords(ensemble=members)

        # ----------------------------------------------------
        # Select variable and time period
        # ---------------------------------------------------- 
        huss = ds_ens["huss"].sel(time=slice(TARGET_START, TARGET_END),
                                  lat=slice(-10, 60),
                                  lon=slice(50, 120))
        
        huss = huss.sortby("lat", ascending=False)
        
        # lon = huss.lon
        # shift = (lon >= 180).sum().item()
        # huss = huss.roll(lon=-shift, roll_coords=True)
        # huss = huss.assign_coords(lon=((huss.lon + 180) % 360) - 180)
        # huss = huss.sortby("lon")
        
        # ----------------------------------------------------
        # Ensemble mean
        # ----------------------------------------------------
        huss_ensmean = huss.mean(dim="ensemble", skipna=True)

        # ----------------------------------------------------
        # Regrid to 1.25°
        # ----------------------------------------------------
        huss_1p25 = huss_ensmean.interp(
                                    lat=lat_new,
                                    lon=lon_new,
                                    method="linear",
                                    kwargs={"fill_value": "extrapolate"}
                                    )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------
        mip_huss[exp][model] = huss_1p25

        print(f"   ✓ Ensemble mean computed & regridded")

        # Optional: close dahussets to free memory
        for ds in ds_list:
            ds.close()


# ============================================================
#  Example: Save one result
# ============================================================

# Example save (optional)
# mip_huss["hist-GHG"]["CESM2"].to_netcdf(
#     "huss_CESM2_hist-GHG_ensmean_1p25deg.nc"
# )

print("✅ All processing complete.")



experiments = ["historical", "hist-aer", "hist-GHG", "hist-nat", "hist-noLu"]

models = [
        "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
        "CESM2", "CNRM-CM6-1",
        "E3SM-2-0", "FGOALS-g3", "GISS-E2-1-G",
        "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MIROC6", 
        "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM"
        ]

# models = [
#         "ACCESS-ESM1-5"
#         ]

ref_model = "BCC-CSM2-MR"

for experiment in experiments:
    # 2. 提取参考经纬度（不变）
    lon_ds = mip_huss[experiment][ref_model].lon
    lat_ds = mip_huss[experiment][ref_model].lat
    
    # Determine common time range (intersection or target period)
    # Here we force all models to 1960-2014
    time_common = pd.date_range(start="1960-01-01", end="2014-12-31", freq="MS")

    data_vars = {}
    
    for model in models:
        if model not in mip_huss[experiment]:
            continue

        da = mip_huss[experiment][model].copy()
        
        # DROP conflicting auxiliary variables
        for v in ["height", "bounds", "time_bnds"]:
            if v in da.coords or v in da.coords:
                da = da.drop_vars(v)

        
        # Rename time dimension to "time"
        time_dim = da.dims[0]  # usually "time" already
        if time_dim != "time":
            da = da.rename({time_dim: "time"})
            
        assert da.sizes["time"] == len(time_common)
        da = da.assign_coords(time=time_common)
        
        data_vars[model] = da
        
    # Create unified Dataset
    ds_huss = xr.Dataset(
                        data_vars=data_vars,
                        coords={
                            "time": time_common,
                            "lat": lat_ds,
                            "lon": lon_ds
                              },
                        attrs={
                            "experiment": experiment,
                            "variable": "ps",
                            "models": models
                            }
                    )

    
    # 查看结果（确认维度无冲突）
    print(ds_huss)
    
    ds_huss.to_netcdf("E:/LUMIP_DAMIP_CMIP/merged_data/" + "mip_ensemble_" + experiment + "_huss.nc", engine="netcdf4") 
    
    del ds_huss

#%%
# ============================================================
#  ps
# ============================================================

root = r"E:\LUMIP_DAMIP_CMIP\from_jw\ps"

models = [
        "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
        "CESM2", "CNRM-CM6-1",
        "E3SM-2-0", "FGOALS-g3", "GISS-E2-1-G",
        "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MIROC6", 
        "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM"
        ]

# models = [
#          "GISS-E2-1-G",
#         ]

experiments = [
                "historical",
                "hist-aer",
                "hist-GHG",
                "hist-nat",
                "hist-noLu",
                ]

TARGET_START = "1960"
TARGET_END   = "2014"

# target grid (1.25° × 1.25°)
# lon_new = np.arange(-180, 180, 1.25)
# lat_new = np.arange(90, -90 - 1.25, -1.25)
lon_new = np.arange(50, 120+1.25, 1.25)
lat_new = np.arange(60, -10-1.25, -1.25)

# output container
mip_ps = {exp: {} for exp in experiments}


# ============================================================
#  Main processing loop
# ============================================================

for model in models:
    for exp in experiments:

        paths = find_all(model, exp, os.path.join(root, exp+"/"+model))

        if len(paths) == 0:
            print(f"⚠ No files found for {model} {exp}")
            continue

        print(f"📂 {model} — {exp} | Ensemble members: {len(paths)}")

        # ----------------------------------------------------
        # Open each ensemble member separately
        # ----------------------------------------------------
        ds_list = []
        members = []

        for p in paths:
            time_coder = CFDatetimeCoder(use_cftime=True)

            ds = xr.open_dataset(
                                p,
                                decode_times=time_coder,
                                # chunks={"time": 12}
                                )
            
            # Enforce the desired period
            ds = ds.sel(time=slice(TARGET_START, TARGET_END))
            
            if "lon_2" in ds.dims:
                ds = ds.drop_vars("lon")
                ds = ds.rename({"lon_2": "lon"})
            
            # get first and last available years
            t0 = ds.time.dt.year.min().item()
            t1 = ds.time.dt.year.max().item()
            
            # strict coverage check
            if (t0 > int(TARGET_START)) or (t1 < int(TARGET_END)):
                print(
                    f"⚠ {p} rejected: covers {t0}–{t1}, "
                    f"needs {TARGET_START}–{TARGET_END}"
                    )
                continue
            
            ds_list.append(ds)
            members.append(extract_member(p))

        # ----------------------------------------------------
        # Concatenate along ensemble dimension
        # ----------------------------------------------------
        ds_ens = xr.concat(ds_list, dim="ensemble", join="inner")
        ds_ens = ds_ens.assign_coords(ensemble=members)

        # ----------------------------------------------------
        # Select variable and time period
        # ----------------------------------------------------
        ps = ds_ens["ps"].sel(time=slice(TARGET_START, TARGET_END),
                              lat=slice(-10, 60),
                              lon=slice(50, 120))
        
        ps = ps.sortby("lat", ascending=False)
        
        # lon = ps.lon
        # shift = (lon >= 180).sum().item()
        # ps = ps.roll(lon=-shift, roll_coords=True)
        # ps = ps.assign_coords(lon=((ps.lon + 180) % 360) - 180)
        # ps = ps.sortby("lon")
        
        # ----------------------------------------------------
        # Ensemble mean
        # ----------------------------------------------------
        ps_ensmean = ps.mean(dim="ensemble", skipna=True)

        # ----------------------------------------------------
        # Regrid to 1.25°
        # ----------------------------------------------------
        ps_1p25 = ps_ensmean.interp(
                                    lat=lat_new,
                                    lon=lon_new,
                                    method="linear",
                                    kwargs={"fill_value": "extrapolate"}
                                    )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------
        mip_ps[exp][model] = ps_1p25

        print(f"   ✓ Ensemble mean computed & regridded")

        # Optional: close dapsets to free memory
        for ds in ds_list:
            ds.close()


# ============================================================
#  Example: Save one result
# ============================================================

# Example save (optional)
# mip_huss["hist-GHG"]["CESM2"].to_netcdf(
#     "huss_CESM2_hist-GHG_ensmean_1p25deg.nc"
# )

print("✅ All processing complete.")



experiments = ["historical", "hist-aer", "hist-GHG", "hist-nat", "hist-noLu"]

models = [
        "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
        "CESM2", "CNRM-CM6-1",
        "E3SM-2-0", "FGOALS-g3", "GISS-E2-1-G",
        "HadGEM3-GC31-LL", "IPSL-CM6A-LR", "MIROC6", 
        "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM"
        ]

# models = [
#         "ACCESS-ESM1-5"
#         ]

ref_model = "BCC-CSM2-MR"

for experiment in experiments:
    # 2. 提取参考经纬度（不变）
    lon_ds = mip_ps[experiment][ref_model].lon
    lat_ds = mip_ps[experiment][ref_model].lat
    
    # Determine common time range (intersection or target period)
    # Here we force all models to 1960-2014
    time_common = pd.date_range(start="1960-01-01", end="2014-12-31", freq="MS")

    data_vars = {}
    
    for model in models:
        if model not in mip_ps[experiment]:
            continue

        da = mip_ps[experiment][model].copy()
        
        # DROP conflicting auxiliary variables
        for v in ["height", "bounds", "time_bnds"]:
            if v in da.coords or v in da.coords:
                da = da.drop_vars(v)

        
        # Rename time dimension to "time"
        time_dim = da.dims[0]  # usually "time" already
        if time_dim != "time":
            da = da.rename({time_dim: "time"})
            
        assert da.sizes["time"] == len(time_common)
        da = da.assign_coords(time=time_common)
        
        data_vars[model] = da
        
    # Create unified Dataset
    ds_ps = xr.Dataset(
                        data_vars=data_vars,
                        coords={
                            "time": time_common,
                            "lat": lat_ds,
                            "lon": lon_ds
                              },
                        attrs={
                            "experiment": experiment,
                            "variable": "ps",
                            "models": models
                            }
                    )

    
    # 查看结果（确认维度无冲突）
    print(ds_ps)
    
    ds_ps.to_netcdf("E:/LUMIP_DAMIP_CMIP/merged_data/" + "mip_ensemble_" + experiment + "_ps.nc", engine="netcdf4") 
    
    del ds_ps