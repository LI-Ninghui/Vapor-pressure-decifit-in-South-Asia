# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 13:08:06 2026

@author: drnin
"""

import numpy as np 
import xarray as xr
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import cftime

from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def es_from_dewpoint(Td):
    """Saturation vapor pressure (hPa) from dew point (°C)."""
    return 6.112 * np.exp((17.67 * Td) / (Td + 243.5))

def specific_humidity_from_dewpoint(Td, p):
    """
    Td: dew point temperature (°C)
    p: air pressure (hPa)
    Returns specific humidity (kg/kg)
    """
    es = es_from_dewpoint(Td)
    q = 0.622 * es / (p - 0.378 * es)
    return q


# ---------------------- 全局配置（可按需调整）----------------------
# 比湿无需单位转换（原始单位：kg kg⁻¹）
CONVERT_TO_C = False  # 禁用温度单位转换（比湿无需）
# 绘图保存目录（修改为比湿专属目录，避免与温度图混淆）
PLOT_SAVE_DIR = Path("E:/LUMIP_DAMIP_CMIP/plots/annual_vpd_models")
PLOT_SAVE_DIR.mkdir(exist_ok=True, parents=True)  # 自动创建目录
# 空间平均范围（保持原区域：76-87°E, 20-28°N，修正lat顺序为从小到大）
REGION = {"lon": (50, 100), "lat": (40, 0)}  # 修正：lat必须从小到大（20→28）
REGION_NAME = "Gangetic Plain"  # 自定义区域名称
# 滑动平均窗口（5年，保持不变）
RUNNING_MEAN_WINDOW = 5
# 标准差阴影配置（保持与多模式均值曲线颜色一致）
STD_ALPHA = 0.2  # 阴影透明度
STD_COLOR = "black"  # 与多模式均值曲线颜色一致（黑色）

# ---------------------- 辅助函数 ----------------------
def linear_trend(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, p_value

def cftime_to_num(time_vals):
    """兼容cftime时间轴的转换函数（此处年数据可能无需，但保留兼容性）"""
    if isinstance(time_vals[0], cftime.datetime):
        return mdates.date2num(time_vals)
    return time_vals  # 年数据直接返回年份

def calculate_running_mean(ts_data, window=RUNNING_MEAN_WINDOW):
    """
    计算时间序列的滑动平均（中心对齐）
    ts_data: xarray.DataArray（1D/3D，支持多维度同时计算）
    return: 滑动平均后的时间序列
    """
    return ts_data.rolling(year=window, center=True, min_periods=3).mean()


# ---- helper functions ----
def es_magnus(Tc):
    # T in °C -> returns hPa
    return 0.611 * np.exp((17.5 * Tc) / (Tc + 240.978))
    # return 0.611 * np.exp((17.67 * Tc) / (Tc + 243.5))
    # return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))
#%%
# ---------------------- 数据处理函数（tas → vpd 替换）----------------------
def process_forcing_vpd(forcing_name: str, data_dir: str, m: list, models: list[str], mask: float) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    计算：多模式年平均比湿 + 多模式平均年比湿 + 多模式比湿1倍标准差
    返回：vpd_annual_stack (model, year, lat, lon) + vpd_annual_multi_mean (year, lat, lon) + vpd_annual_std (year, lat, lon)
    """
    data_dir = Path(data_dir)
    individual_vpd_annual = []

    # 关键替换：读取vpd比湿文件（而非tas温度文件）
    file_path = data_dir / f"scemip_single_{forcing_name}_vpd.nc"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing VPD file: {file_path}")

    print(f"Processing forcing experiment: {forcing_name} (VPD)")
    ds_vpd = xr.open_dataset(file_path)
    s_year, e_year = 2015, 2099
    yr = np.arange(s_year, e_year+1)

    for model in models:
        print(f"  Processing VPD for model: {model}")
        vpd = ds_vpd[model]  # 关键替换：tas → vpd
        time_dim = vpd.dims[0]
        
        # vpd_seasonal = vpd.sel(time=vpd.time.dt.month.isin(m))
        # vpd_seasonal = vpd_seasonal.sel(
        #     time=(vpd_seasonal.time.dt.year >= s_year) & 
        #          (vpd_seasonal.time.dt.year <= e_year)
        # )

        # 年平均 + 补全缺失年份（逻辑不变）
        # vpd_annual_clim = vpd.where(mask).groupby("time.year").mean()
        # vpd_annual_anom = vpd.where(mask).groupby("time.year") - vpd_annual_clim
        # vpd_annual_anom = vpd_annual_anom.sel(time=vpd.time.dt.month.isin(m))
        # individual_vpd_annual.append(vpd_annual_anom.groupby("time.year").mean().mean(dim=["lat", "lon"]))
        
        vpd_seasonal = vpd.sel(time=vpd.time.dt.month.isin(m))
        vpd_seasonal = vpd_seasonal.sel(
            time=(vpd_seasonal.time.dt.year >= s_year) & 
                 (vpd_seasonal.time.dt.year <= e_year)
        )

        # 年平均 + 补全缺失年份（逻辑不变）
        # vpd_annual = vpd_seasonal.where(mask).groupby("time.year").mean() 
        vpd_annual = vpd_seasonal.groupby("time.year").mean() 
        vpd_annual = vpd_annual.reindex(year=yr)
        individual_vpd_annual.append(vpd_annual)

    # 堆叠模式
    vpd_annual_stack = xr.concat(
        objs=individual_vpd_annual,
        dim=xr.DataArray(models, name="model", dims="model")
    )

    # 计算多模式平均和1倍标准差（沿model维度，逻辑不变）
    vpd_annual_multi_mean = vpd_annual_stack.mean(dim="model", skipna=True) 
    # vpd_annual_std = vpd_annual_stack.std(dim="model", skipna=True)  # 1倍标准差

    ds_vpd.close()
    return vpd_annual_stack, vpd_annual_multi_mean#, vpd_annual_std  # 返回比湿相关数据


#%%
# ---------------------- 主程序执行（tas → huss 替换）----------------------
if __name__ == "__main__":
    # ---------------------- 关键修改：加载ERA5比湿数据（替换温度t2m）----------------------
    # 注意：需确保ERA5数据文件中包含比湿变量（通常变量名为 vpd 或 q）
    # 若ERA5比湿变量名不是vpd，需修改下面的变量名（例如改为 q 或其他实际名称）
    path_vpd_glob = r"G:\VPD\data\ERA5_1960_2025\d2m_t2m.nc"
    path_sp_glob = r"G:\VPD\Precipitation\ERA5_sp_monthly.nc"
    path_mask_p125 = r"G:\VPD\results\mask\mask_indo_gangetic_p125.nc"
    # path_mask_p025 = r"G:\VPD\results\mask\mask_indo_gangetic.nc"                
    
    path_mask_p125 = r"G:\VPD\results\mask\mask_indo_gangetic_p125.nc"

    with xr.open_dataset(path_mask_p125) as f_mask:
        mask_p125 = f_mask["mask"]
        
    f_mask.close()

#%%
    # 配置参数（保持不变）
    DATA_DIR = "E:/LUMIP_DAMIP_CMIP/merged_data"

    # MODELS = [
    #         "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CESM2", "CNRM-CM6-1",
    #         "E3SM-2-0", "FGOALS-g3", "GISS-E2-1-G", "HadGEM3-GC31-LL",
    #         "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM"
    #         ]
    
    
    MODELS = [
            "ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR",
            "BCC-CSM2-MR", "CanESM5", "CAS-ESM2-0",
            "CMCC-ESM2", "EC-Earth3",
            "FGOALS-f3-L", "FGOALS-g3",
            "GFDL-ESM4", "INM-CM4-8", "INM-CM5-0",
            "IPSL-CM6A-LR", "KACE-1-0-G", "MIROC6",
            "MPI-ESM1-2-LR", "NorESM2-LM",
            "NorESM2-MM", "TaiESM1"
             ]
    
    # MODELS = [
    #         "BCC-CSM2-MR", 
    #         ]
    # 季节选择（保持不变）
    # MONTHS = [6, 7, 8, 9]  # 夏季：6-9月
    # MONTHS = [1, 2, 3, 4, 5, 10, 11, 12]  # 冬季（可选）
    
    SEASONS = {
            "Pre-monsoon": [3, 4, 5],
            "Monsoon": [6, 7, 8, 9],
            "Post-monsoon": [10, 11],
            "Winter": [12, 1, 2],
            # "All-season": np.arange(1, 13, 1)
            }
    
    
    FORCING_LIST = ["ssp126", "ssp245", "ssp370", "ssp585"] #"historical", "hist-aer", "hist-GHG", "hist-nat",  "historical-noLu"
    
    vpd_era_dict = {k: {} for k in SEASONS.keys()}
    # vpd_annual_dict = {}
    vpd_annual_multi_mean_dict = {k: {} for k in SEASONS.keys()}
    
    
    for forcing in FORCING_LIST:
        for sname, MONTHS in SEASONS.items():
            print(sname, forcing)

            _, vpd_annual_multi_mean = process_forcing_vpd(
                                                            forcing_name=forcing,
                                                            data_dir=DATA_DIR,
                                                            m=MONTHS,
                                                            models=MODELS,
                                                            mask=mask_p125
                                                            )
                # vpd_annual_dict[sname] = vpd_annual_stack
            vpd_annual_multi_mean_dict[sname][forcing] = vpd_annual_multi_mean
           
#%%        
    
    yr = np.arange(2015, 2099+1, 1)
    vpd_mean_trend_dict, vpd_mean_pval_dict = {k: {} for k in SEASONS.keys()}, {k: {} for k in SEASONS.keys()}
    for forcing in FORCING_LIST:
        for sname, MONTHS in SEASONS.items():
            
            vpd_mean_trend, vpd_mean_pval = xr.apply_ufunc(
                                                linear_trend,
                                                yr,
                                                vpd_annual_multi_mean_dict[sname][forcing]*10,  #.rolling(year=5, center=True, min_periods=3).mean()
                                                vectorize=True,
                                                input_core_dims=[['year'], ['year']],
                                                output_core_dims=[[] for _ in range(2)],
                                                dask="parallelized",
                                                output_dtypes=[np.float64, np.float64]
                                                ) 
        
            vpd_mean_trend_dict[sname][forcing], vpd_mean_pval_dict[sname][forcing] = vpd_mean_trend.data, vpd_mean_pval
     
            
            
#%%
from matplotlib.gridspec import GridSpec


from cartopy.io.shapereader import Reader as shpreader
import cartopy.feature as cfeat
import cartopy.crs as ccrs

# prj = ccrs.Mercator()
prj = ccrs.PlateCarree()
proj = prj

world_countries_path = r"G:\VPD\results\shape\world_countries\world_countries.shp"

# # draw the boundary of each country
world_countries = shpreader(world_countries_path).geometries()
world_countries = cfeat.ShapelyFeature(world_countries, prj, edgecolor='k', facecolor='none')

import colormaps as cmaps
import shapely.geometry as sgeom
from matplotlib.gridspec import GridSpec
# from matplotlib.ticker import MaxNLocator
# from matplotlib.patches import Wedge
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
# from skimage.segmentation import find_boundaries
import matplotlib.axes as maxes
import matplotlib.patches as patches

#%%
# Define region box
# -------------------------------------------------
extent = [60, 100, 5, 40]
xticks = [60, 70, 80, 90, 100]
yticks = [10, 20, 30, 40]

labels_lon = [rf'$\rm {tick}\degree E$' for tick in xticks]
labels_lat = [rf'$\rm {tick}\degree N$' for tick in yticks]

ilevs = np.arange(0, 0.15+0.01, 0.01)

seasons = [
    ("Pre-monsoon", "(a) Pre-monsoon"),
    ("Monsoon",     "(b) Monsoon"),
    ("Post-monsoon","(c) Post-monsoon"),
    ("Winter",      "(d) Winter")
]


n_forcings = len(FORCING_LIST)
n_seasons = len(seasons)

n_rows = 5  # Total rows (1 for ERA5 + 4 for SSP)
n_cols = 4  # Total columns (4 seasons)


ERA5_LABEL = "ERA5"  # First row data label

# meshgrid (use lon/lat from DataArray)
lon = vpd_mean_trend.lon
lat = vpd_mean_trend.lat
X, Y = np.meshgrid(lon, lat)

# -------------------------------------------------
# figure & layout
# -------------------------------------------------
fig = plt.figure(figsize=(18, 15), layout="constrained")
gs = GridSpec(4, 4, figure=fig)



for forcing_idx, forcing in enumerate(FORCING_LIST):
    # Forcing corresponds to row (0-3) in GridSpec(4,4)
    row = forcing_idx 
    
    for season_idx, (season, title) in enumerate(seasons):
        # Season corresponds to column (0-3) in GridSpec(4,4)
        col = season_idx
        
        # Step 1: Calculate global index (0-15) for current subplot
        global_idx = forcing_idx * n_seasons + season_idx  # Critical: unique index for 16 subplots
        
        # Step 2: Convert global index to letter (a-p)
        title_letter = chr(ord('a') + global_idx)  # ord('a')=97, chr(97)='a', ..., chr(112)='p'
        sequential_title = f"({title_letter})"  # Format: (a), (b), ..., (p)
        
        # Optional: Combined title (sequential letter + season/forcing for clarity)
        combined_title = f"({title_letter}) {season} ({forcing})"
        
        # Create subplot at gs[row, col] with cartopy projection
        ax = fig.add_subplot(gs[row, col], projection=proj)
        
        
    
        # ax = fig.add_subplot(gs[i // 2, i % 2], projection=proj)
        # ax = fig.add_subplot(gs[0, i], projection=proj)
        ax.set_extent(extent, crs=proj)
    
        ax.add_feature(world_countries, linewidth=0.75)
    
        ax.set_xticks(xticks, crs=proj)
        ax.set_yticks(yticks, crs=proj)
        ax.set_xticklabels(labels_lon, fontsize=14)
        ax.set_yticklabels(labels_lat, fontsize=14)
    
        cf = ax.contourf(
                        X,
                        Y,
                        vpd_mean_trend_dict[season][forcing],
                        levels=ilevs,
                        cmap=cmaps.WhiteBlueGreenYellowRed,
                        extend="max",
                        transform=ccrs.PlateCarree()
                        )
        
        # Downsample by factor N
        N = 1
        mask = vpd_mean_pval_dict[season][forcing] <= 0.05
        mask_ds = mask.isel(lat=slice(None, None, N),
                            lon=slice(None, None, N))
        
        lon2d, lat2d = xr.broadcast(mask_ds.lon, mask_ds.lat)

        # Scatter
        ax.scatter(lon2d.where(mask_ds), lat2d.where(mask_ds),
                   s=1, color='black', transform=proj)
        
        # edge = ax.contour(
        #                  igp_merra_lon, igp_merra_lat,  # Lon/lat of the mask grid
        #                  igp_mask_merra_numerical,
        #                  levels=[0.5],  # Critical: only extract the boundary between IGP and non-IGP
        #                  colors="red",
        #                  linewidths=1.2,
        #                  # alpha=boundary_alpha,
        #                  transform=proj,
        #                  alpha=0.5
        #                 )
    
        # Set subplot title (distinguish forcing and season)
        ax.set_title(combined_title, fontsize=16, loc="left")

        ax.tick_params(labelsize=14)
        
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes(
        #                          "right", size="4%", pad=0.3, axes_class=maxes.Axes
        #                          )
        # cax.remove()
        
        cf_last = cf   # store last handle
            
        
# -----------------------------
# shared colorbar
# -----------------------------
cbar = fig.colorbar(
                    cf_last,
                    ax=fig.axes,
                    orientation="horizontal",
                    # fraction=0.035,
                    aspect=35,
                    shrink=0.7,
                    pad=0.02
                    )

cbar.set_label(r"VPD trend [Kpa $\rm decade^{-1}$]", fontsize=18)
cbar.ax.tick_params(labelsize=18)
# cbar.set_ticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
cbar.set_ticks([0, 0.05, 0.1, 0.15])


fig.savefig("G:/VPD/results/img_ncc/VPD_trend_SSP_2D.jpg", format="jpg", dpi=600, bbox_inches="tight")
fig.savefig("G:/VPD/results/img_ncc/VPD_trend_SSP_2D.pdf", format="pdf", dpi=600, bbox_inches="tight")
fig.savefig("G:/VPD/results/img_ncc/VPD_trend_SSP_2D.tiff", format="tiff", dpi=600, bbox_inches="tight")
