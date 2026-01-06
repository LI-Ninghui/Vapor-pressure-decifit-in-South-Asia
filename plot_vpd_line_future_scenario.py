# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 19:29:25 2025

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
        vpd_annual = vpd_seasonal.where(mask).groupby("time.year").mean().mean(dim=["lat", "lon"]) 
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
        
            vpd_mean_trend_dict[sname][forcing], vpd_mean_pval_dict[sname][forcing] = vpd_mean_trend.data, vpd_mean_pval.data
     
            
            
#%%
from matplotlib.gridspec import GridSpec


# High distinguishability + scientific plotting aesthetics + no green confusion
ssp_colors = {
            "ssp126": "#2C3E50",    # Dark Slate Gray (low emission, stable, professional)
            "ssp245": "#3498DB",    # Bright Blue (medium-low emission, calm)
            "ssp370": "#E74C3C",    # Crimson Red (medium-high emission, warning)
            "ssp585": "#F39C12"     # Dark Orange (highest emission, extreme, high contrast)
            }

years = np.arange(2015, 2099+1, 1)
fig = plt.figure(figsize=(16, 10),  layout="tight")
# Create GridSpec for 3×2 layout
gs = GridSpec(3, 2, figure=fig)

# Reassign upper subplots with GridSpec (keep original positions)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])  # (e) VPD Trend Distribution (box plot)

# Merge last row's two columns for legend
# ax_leg = fig.add_subplot(gs[2, :])  # gs[2, :] = row 2, all columns (0 and 1)
# ax_leg.axis("off")

# ax1.axhline(0, color="gray", linewidth=1)
ax1.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["ssp126"], linestyle="--", color=ssp_colors["ssp126"], linewidth=0.5)
ax1.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["ssp126"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp126"], linewidth=2, label="ssp126")

ax1.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["ssp245"], linestyle="--", color=ssp_colors["ssp245"], linewidth=0.5)
ax1.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["ssp245"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp245"], linewidth=2, label="ssp245")

ax1.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["ssp370"], linestyle="--", color=ssp_colors["ssp370"], linewidth=0.5)
ax1.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["ssp370"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp370"], linewidth=2, label="ssp370")

ax1.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["ssp585"], linestyle="--", color=ssp_colors["ssp585"], linewidth=0.5)
ax1.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["ssp585"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp585"], linewidth=2, label="ssp585")

# ax1.set_ylim([-0.07, 0.07])
ax1.set_xlim([2015, 2099])
# Legend inside the upper part of the plot, one row
# ax1.legend(
#     loc="upper center",        # top-center of the plot
#     bbox_to_anchor=(0.6, 1.18), # move slightly down inside the panel
#     ncol=4,                   # one line with two entries
#     frameon=True,
#     fontsize=14,
#     # title_fontsize=14
#     )

ax1.set_xlabel("Year", fontsize=16)
ax1.set_ylabel(r"$\rm VPD$ [KPa]", fontsize=16)


ax1.set_title("(a) Pre-monsoon", fontsize=18, loc="left")
ax1.tick_params(labelsize=15)


# Second right axis for humidity (vapor pressure anomaly in hPa)
# c_e = 'orange'
# ax1t = ax1.twinx()
# # offset the third axis to the right
# # ax1t.spines["right"].set_position(("axes", 1.0))
# ax1t.spines['right'].set_color(c_e)
# # make patch invisible (so it doesn't cover the plot)
# ax1t.set_frame_on(True)
# # ax1t.patch.set_visible(False)
# ax1t.tick_params(axis='y', labelcolor=c_e, color=c_e)

# ax1t.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["hist-noLu"], 
#           color=c_e, 
#           lw=0.5, 
#           ls='--',
#           alpha=0.7)

# ax1t.plot(years, vpd_annual_multi_mean_dict["Pre-monsoon"]["hist-noLu"].rolling(year=5, center=True, min_periods=3).mean(), 
#           color=c_e, 
#           lw=1.5, 
#           ls='-',
#           alpha=0.7)


# -----------------------------------------------------------------------------
# ax2 = fig.add_subplot(3, 2, 2)

ax2.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["ssp126"], linestyle="--", color=ssp_colors["ssp126"], linewidth=0.5)
ax2.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["ssp126"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp126"], linewidth=2, label="ssp126")

ax2.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["ssp245"], linestyle="--", color=ssp_colors["ssp245"], linewidth=0.5)
ax2.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["ssp245"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp245"], linewidth=2, label="ssp245")

ax2.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["ssp370"], linestyle="--", color=ssp_colors["ssp370"], linewidth=0.5)
ax2.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["ssp370"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp370"], linewidth=2, label="ssp370")

ax2.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["ssp585"], linestyle="--", color=ssp_colors["ssp585"], linewidth=0.5)
ax2.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["ssp585"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp585"], linewidth=2, label="ssp585")

# ax2.set_ylim([-0.07, 0.07])
ax2.set_xlim([2015, 2099])
# Legend inside the upper part of the plot, one row
# ax2.legend(
#     loc="upper center",        # top-center of the plot
#     bbox_to_anchor=(0.6, 1.18), # move slightly down inside the panel
#     ncol=4,                   # one line with two entries
#     frameon=True,
#     fontsize=14,
#     title_fontsize=14
#     )

ax2.set_xlabel("Year", fontsize=16)
ax2.set_ylabel(r"$\rm VPD$ [KPa]", fontsize=16)

ax2.set_title("(b) Monsoon", fontsize=18, loc="left")

ax2.tick_params(labelsize=15)


# Second right axis for humidity (vapor pressure anomaly in hPa)
# c_e = 'orange'
# ax2t = ax2.twinx()
# # offset the third axis to the right
# # ax2t.spines["right"].set_position(("axes", 1.0))
# ax2t.spines['right'].set_color(c_e)
# # make patch invisible (so it doesn't cover the plot)
# ax2t.set_frame_on(True)
# # ax2t.patch.set_visible(False)
# ax2t.tick_params(axis='y', labelcolor=c_e, color=c_e)

# ax2t.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["hist-noLu"], 
#           color=c_e, 
#           lw=0.5, 
#           ls='--',
#           alpha=0.7)

# ax2t.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["hist-noLu"].rolling(year=5, center=True, min_periods=3).mean(), 
#           color=c_e, 
#           lw=1.5, 
#           ls='-',
#           alpha=0.7)


# -----------------------------------------------------------------------------
# ax3 = fig.add_subplot(3, 2, 3)

ax3.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["ssp126"], linestyle="--", color=ssp_colors["ssp126"], linewidth=0.5)
ax3.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["ssp126"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp126"], linewidth=2, label="ssp126")

ax3.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["ssp245"], linestyle="--", color=ssp_colors["ssp245"], linewidth=0.5)
ax3.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["ssp245"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp245"], linewidth=2, label="ssp245")

ax3.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["ssp370"], linestyle="--", color=ssp_colors["ssp370"], linewidth=0.5)
ax3.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["ssp370"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp370"], linewidth=2, label="ssp370")

ax3.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["ssp585"], linestyle="--", color=ssp_colors["ssp585"], linewidth=0.5)
ax3.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["ssp585"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp585"], linewidth=2, label="ssp585")

# ax3.set_ylim([-0.07, 0.07])
ax3.set_xlim([2015, 2099])
# Legend inside the upper part of the plot, one row
# ax3.legend(
#     loc="upper center",        # top-center of the plot
#     bbox_to_anchor=(0.6, 1.18), # move slightly down inside the panel
#     ncol=4,                   # one line with two entries
#     frameon=True,
#     fontsize=14,
#     title_fontsize=14
#     )

ax3.set_xlabel("Year", fontsize=16)
ax3.set_ylabel(r"$\rm VPD$ [KPa]", fontsize=16)

ax3.set_title("(c) Post-monsoon", fontsize=18, loc="left")

ax3.tick_params(labelsize=15)     



# Second right axis for humidity (vapor pressure anomaly in hPa)
# c_e = 'orange'
# ax3t = ax3.twinx()
# # offset the third axis to the right
# # ax3t.spines["right"].set_position(("axes", 1.0))
# ax3t.spines['right'].set_color(c_e)
# # make patch invisible (so it doesn't cover the plot)
# ax3t.set_frame_on(True)
# # ax3t.patch.set_visible(False)
# ax3t.tick_params(axis='y', labelcolor=c_e, color=c_e)

# ax3t.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["hist-noLu"], 
#           color=c_e, 
#           lw=0.5, 
#           ls='--',
#           alpha=0.7)

# ax3t.plot(years, vpd_annual_multi_mean_dict["Post-monsoon"]["hist-noLu"].rolling(year=5, center=True, min_periods=3).mean(), 
#           color=c_e, 
#           lw=1.5, 
#           ls='-',
#           alpha=0.7)


# -----------------------------------------------------------------------------
# ax4 = fig.add_subplot(3, 2, 4)

ax4.plot(years, vpd_annual_multi_mean_dict["Winter"]["ssp126"], linestyle="--", color=ssp_colors["ssp126"], linewidth=0.5)
ax4.plot(years, vpd_annual_multi_mean_dict["Winter"]["ssp126"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp126"], linewidth=2, label="ssp126")

ax4.plot(years, vpd_annual_multi_mean_dict["Winter"]["ssp245"], linestyle="--", color=ssp_colors["ssp245"], linewidth=0.5)
ax4.plot(years, vpd_annual_multi_mean_dict["Winter"]["ssp245"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp245"], linewidth=2, label="ssp245")

ax4.plot(years, vpd_annual_multi_mean_dict["Winter"]["ssp370"], linestyle="--", color=ssp_colors["ssp370"], linewidth=0.5)
ax4.plot(years, vpd_annual_multi_mean_dict["Winter"]["ssp370"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp370"], linewidth=2, label="ssp370")

ax4.plot(years, vpd_annual_multi_mean_dict["Winter"]["ssp585"], linestyle="--", color=ssp_colors["ssp585"], linewidth=0.5)
ax4.plot(years, vpd_annual_multi_mean_dict["Winter"]["ssp585"].rolling(year=5, center=True, min_periods=3).mean(), 
         linestyle="-", color=ssp_colors["ssp585"], linewidth=2, label="ssp585")

# ax4.set_ylim([-0.07, 0.07])
ax4.set_xlim([2015, 2099])

# Legend inside the upper part of the plot, one row
# ax4.legend(
#     loc="upper center",        # top-center of the plot
#     bbox_to_anchor=(0.57, 1.18), # move slightly down inside the panel
#     ncol=4,                   # one line with two entries
#     frameon=True,
#     fontsize=15,
#     title_fontsize=15
#     )

ax4.set_xlabel("Year", fontsize=16)
ax4.set_ylabel(r"$\rm VPD$ [KPa]", fontsize=16)

ax4.set_title("(d) Winter", fontsize=18, loc="left")

ax4.tick_params(labelsize=15)      


# Second right axis for humidity (vapor pressure anomaly in hPa)
# c_e = 'orange'
# ax4t = ax4.twinx()
# # offset the third axis to the right
# # ax4t.spines["right"].set_position(("axes", 1.0))
# ax4t.spines['right'].set_color(c_e)
# # make patch invisible (so it doesn't cover the plot)
# ax4t.set_frame_on(True)
# # ax4t.patch.set_visible(False)
# ax4t.tick_params(axis='y', labelcolor=c_e, color=c_e)

# ax4t.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["hist-noLu"], 
#           color=c_e, 
#           lw=0.5, 
#           ls='--',
#           alpha=0.7)

# ax4t.plot(years, vpd_annual_multi_mean_dict["Monsoon"]["hist-noLu"].rolling(year=5, center=True, min_periods=3).mean(), 
#           color=c_e, 
#           lw=1.5, 
#           ls='-',
#           alpha=0.7)


# Simulate grouped box plot data (4 SSP × 4 Seasons, 50 samples per combination)
# Trend level: SSP126 < SSP245 < SSP370 < SSP585; Pre > Post > Mon > Win
seasons = ["Pre-monsoon", "Monsoon", "Post-monsoon", "Winter"]
ssps = ["ssp126", "ssp245", "ssp370", "ssp585"]
grouped_box_data = {k: {} for k in SEASONS.keys()}

# Simulate grouped_box_data (EXPLICIT structure: {season: {ssp: data}})
# No hidden omissions, guaranteed every (season, ssp) pair has data
grouped_box_data = {}
for season in seasons:
    grouped_box_data[season] = {}
    for ssp in ssps:
        # Generate valid data for every combination (no empty values)
        ssp_base = float(ssp[3:]) / 1000
        data = np.random.normal(loc=ssp_base, scale=ssp_base * 0.2, size=50)
        grouped_box_data[season][ssp] = data

# ======================
# 2. Generate Data & Positions (SIMPLIFIED Logic + Guaranteed Length)
# ======================
box_width = 0.8
group_spacing = 1.5  # Increased for clarity, no impact on length
x_positions = []
all_box_data = []


n_ssps = len(ssps)
n_seasons = len(seasons)
expected_length = n_ssps * n_seasons  # Exact expected length: 16
# ----------------------
# Step 1: Simplified x_positions generation (guaranteed length = 16)
# Avoid complex floating-point calculations that cause length errors
# ----------------------
for i in range(expected_length):
    # Linear x positions with group spacing (SSP groups separated by group_spacing)
    group_idx = i // n_seasons  # Which SSP group (0-3)
    season_idx_in_group = i % n_seasons  # Which season in the group (0-3)
    x_pos = (group_idx * (n_seasons * box_width + group_spacing)) + (season_idx_in_group * box_width)
    x_positions.append(x_pos)

# ----------------------
# Step 2: Simplified all_box_data generation (guaranteed length = 16)
# Match the EXACT same order as x_positions (one-to-one correspondence)
# ----------------------
for ssp in ssps:  # Outer loop: SSP (group)
    for season in seasons:  # Inner loop: Season (within group)
        # Explicitly append data for every (ssp, season) pair
        data = grouped_box_data[season][ssp]
        all_box_data.append(data)

# ----------------------
# Step 3: FORCED Length Verification (Crash early with clear error if mismatch)
# ----------------------
if len(all_box_data) != len(x_positions):
    raise ValueError(
        f"Length Mismatch! all_box_data: {len(all_box_data)}, "
        f"x_positions: {len(x_positions)}. Expected both to be {expected_length}."
    )
print(f" Length Verified: Both lists have length {len(all_box_data)}")

# ======================
# 3. Draw Box Plot (No Hidden Mismatches)
# ======================

# Draw box plot (guaranteed no length mismatch)
box_plot = ax5.boxplot(
                    all_box_data,
                    positions=x_positions,
                    patch_artist=True,
                    widths=box_width,
                    medianprops={"color": "white", "linewidth": 1.2},
                    whiskerprops={"color": "black", "linewidth": 0.8},
                    capprops={"color": "black", "linewidth": 0.8},
                    flierprops={"marker": "o", "markerfacecolor": "gray", "markersize":4, "alpha": 0.4}
                    )

# ----------------------
# Step 4: Color Assignment (Guaranteed to match data/position order)
# ----------------------
color_idx = 0
for ssp in ssps:  # Outer loop: SSP (match data/position order)
    ssp_color = ssp_colors[ssp]
    for _ in range(n_seasons):  # Inner loop: Season (4 per SSP)
        if color_idx < len(box_plot["boxes"]):  # Safe guard against index overflow
            box_plot["boxes"][color_idx].set_facecolor(ssp_color)
            box_plot["boxes"][color_idx].set_alpha(0.7)
            color_idx += 1

# ======================
# 4. Axes Configuration (Clear Labels & Ticks)
# ======================
# ax5.set_xlabel("SSP Scenarios & Seasons", fontsize=14)
ax5.set_ylabel(r"[KPa $\rm decade^{-1}$]", fontsize=16)
ax5.set_title("(e) VPD trend (4 SSP × 4 Seasons)", fontsize=16, loc="left")
ax5.tick_params(labelsize=16)
ax5.grid(axis="y", alpha=0.3, linestyle="-")

# Set SSP group ticks (centered on each group)
ssp_tick_positions = []
for group_idx in range(n_ssps):
    group_center = (group_idx * (n_seasons * box_width + group_spacing)) + ((n_seasons - 1) * box_width / 2)
    ssp_tick_positions.append(group_center)
ax5.set_xticks(ssp_tick_positions)
ax5.set_xticklabels(ssps)




# ======================
# 5. Core: Refactor legend labels (add seasonal trend text/numbers, consistent SSP colors)
# ======================
# Get original SSP handles & labels (from ax4)

season_abbr = {"Pre-monsoon": "Pre-", "Monsoon": "Monsoon", "Post-monsoon": "Post-", "Winter": "Winter"} # Shorten season names for compactness

original_handles, original_labels = ax4.get_legend_handles_labels()
# Create compact legend axis (last row, span 2 columns)
ax_leg = fig.add_subplot(gs[2, 1])
ax_leg.axis("off")
# Refactor labels: integrate seasonal trend text/numbers (no extra lines, only text supplement)
refactored_labels = []
for ssp in original_labels:
    # Extract seasonal trend numbers (format to 1 decimal place for neatness)
    trend_text = f"({season_abbr['Pre-monsoon']}: {vpd_mean_trend_dict['Pre-monsoon'][ssp]:.2f}, " \
                 f"{season_abbr['Monsoon']}: {vpd_mean_trend_dict['Monsoon'][ssp]:.2f}, " \
                 f"{season_abbr['Post-monsoon']}: {vpd_mean_trend_dict['Post-monsoon'][ssp]:.2f}, " \
                 f"{season_abbr['Winter']}: {vpd_mean_trend_dict['Winter'][ssp]:.2f})"
    # New label: SSP name + seasonal trend text/numbers (color remains consistent with SSP)
    refactored_labels.append(f"{ssp} {trend_text}")


# ======================
# Compact legend (fit the short legend axis, no blank space)
# ======================
line_handles, line_labels = ax4.get_legend_handles_labels()
box_handles, box_labels   = ax5.get_legend_handles_labels()

leg1 = ax_leg.legend(
                    line_handles,
                    line_labels,
                    loc="upper center",        # Anchor at top center of the short axis
                    bbox_to_anchor=(0.5, 1.0), # Slight vertical adjustment (fit the short axis)
                    ncol=4,                    # 4 columns (horizontal compactness)
                    frameon=True,
                    fontsize=16,               # Compact font size (readable)
                    title="VPD Time Series (SSP Scenarios)",
                    title_fontsize=16,         # Compact title font size
                    # handlelength=1.0,          # Shorten legend line
                    handletextpad=0.5,         # Reduce line-text space
                    # columnspacing=1.5,         # Reduce column space
                    # borderpad=0.5              # Reduce inner margin
                    )

# IMPORTANT: keep the first legend
ax_leg.add_artist(leg1)


from matplotlib.patches import Patch
# leg2 = ax_leg.legend(
#                     box_handles,
#                     box_labels,
#                     loc="center",
#                     ncol=4,
#                     frameon=True,
#                     bbox_to_anchor=(0.5, 0.35), 
#                     fontsize=16,
#                     title="SSP scenarios",
#                     title_fontsize=16
#                     )

legend_elements = [
    Patch(facecolor=ssp_colors[ssp], alpha=0.7, label=ssp)
    for ssp in ssps  # 遍历所有SSP，与你的数据顺序一致
]

# 给ax5添加图例，可自定义位置、字体大小等样式
leg2 = ax_leg.legend(
                    handles=legend_elements,  # 传入构建好的图例条目
                    loc="center",
                    ncol=4,
                    frameon=True,
                    bbox_to_anchor=(0.5, 0.25), 
                    fontsize=16,
                    handlelength=1.7,
                    # title="SSP scenarios",
                    title_fontsize=16
                    )

# ======================
# 7. Optimize layout (no bottom blank space)
# ======================
# plt.subplots_adjust(
#     hspace=0.1,
#     wspace=0.2,
#     bottom=0.02,
#     top=0.98
# )

# ======================
# Adjust layout and save (optimized for cross-column legend)
# ======================
# plt.subplots_adjust(hspace=0.1, wspace=0.2)  # Fine-tune spacing between subplots
# fig.savefig("G:/VPD/results/img_ncc/VPD_ssp.jpg", format="jpg", dpi=1000, bbox_inches="tight")
# fig.savefig("G:/VPD/results/img_ncc/VPD_ssp.pdf", format="pdf", dpi=1000, bbox_inches="tight")
# fig.savefig("G:/VPD/results/img_ncc/VPD_ssp.tiff", format="tiff", dpi=1000, bbox_inches="tight")

plt.show()





