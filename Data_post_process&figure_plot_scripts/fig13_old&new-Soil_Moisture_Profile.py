import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import timedelta
from cftime import DatetimeNoLeap
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import string
import cmaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

# 设置全局字体为Helvetica
plt.rcParams['font.family'] = 'Helvetica'

def read_nc_moisture(nc_file, start_date_str, end_date_str):
    """读取并处理模拟的土壤湿度数据"""
    # 转换日期格式为DatetimeNoLeap对象
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str).timetuple()[:3])
    )
    
    # 读取数据并转换单位
    data = xr.open_dataset(nc_file)
    moisture = data.SOILLIQ.isel(lndgrid=0).rename({'levsoi': 'levgrnd'}).sel(time=time_slice)
    thick = data.DZSOI.isel(lndgrid=0).isel(levgrnd=slice(None, -5))
    moisture_v = moisture / thick / 1000.0  # 单位: m³/m³
    depth_levels = data.ZSOI.isel(lndgrid=0).isel(levgrnd=slice(None, -5))  # 获取深度信息
    
    return moisture_v, depth_levels

def read_observed_moisture(csv_file, start_date_str, end_date_str, depths):
    """读取并处理观测的土壤湿度数据"""
    # 读取数据并设置时间索引
    data = pd.read_csv(csv_file)
    data.index = pd.to_datetime(data.iloc[:, 0])
    
    # 筛选时间范围并处理异常值
    mask = (data.index >= pd.to_datetime(start_date_str)) & \
            (data.index <= pd.to_datetime(end_date_str))
    moistures = data[mask]
    
    # 确保湿度数据为数值类型
    numeric_columns = moistures.select_dtypes(include=[np.number]).columns
    moistures = moistures[numeric_columns]
    
    # 处理异常值
    moistures = moistures.apply(pd.to_numeric, errors='coerce')  # 将非数值转换为NaN
    moistures[moistures > 1] = np.NAN  # 土壤湿度不应超过1 m³/m³
    
    return moistures.T

def plot_moisture_profile(ax, time_grid, depth_grid, data, label, row_idx, levels, ylim, obs_data=None):
    """绘制单个土壤湿度剖面图"""
    # 绘制等值线填充图
    norm = mpl.colors.BoundaryNorm(levels, cmaps.MPL_YlGnBu.N, extend='both')
    c = ax.contourf(time_grid, depth_grid, data, 
                    cmap=cmaps.MPL_YlGnBu, norm=norm, levels=levels, extend='both')
    
    # 设置标题和轴
    ax.set_title(f'({label})', loc='left', fontsize=22, fontname='Helvetica')
    ax.set_ylim(0, ylim)
    ax.invert_yaxis()
    
    # 设置y轴刻度
    yticks = np.linspace(0, ylim, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontsize=18, fontname='Helvetica')
    
    ax.set_ylabel('Soil depth (m)', fontsize=20, fontname='Helvetica')
    
    # 加粗边框和主刻度线
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(axis='both', width=2.0, length=6.0)

    ax.tick_params(axis='both', labelsize=18)
    
    # 为每个图形添加colorbar
    cax = inset_axes(ax, width="5%", height="100%", loc='right',
                bbox_to_anchor=(0.1, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = plt.colorbar(c, cax=cax, orientation='vertical', ticks=levels[::2])
    cb.ax.tick_params(labelsize=18, direction='out', width=2.0, length=6.0)
    cb.outline.set_linewidth(1.5)
    cax.yaxis.set_ticks_position('right')
    cb.ax.text(1.0, 1.15, 'm³/m³', fontsize=18, ha='center', va='top', transform=cb.ax.transAxes, fontname='Helvetica')
    
    # 设置x轴时间格式
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))  # 设置在每月第1天
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # 使用%b显示月份缩写
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=18, fontname='Helvetica')  # 增大字体大小


if __name__ == "__main__":
    # 基础路径配置
    base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0'
    model_path = f'{base_path}/data/model_output/40SL_discret_scheme/MAQU'
    obs_path = f'{base_path}/data/Stations_soil_data/new_scheme/MAQU12SL'
    output_path = f'{base_path}/fig/202506'
    
    # MAQU站点配置
    depths = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160]
    dates = ('2022-10-01', '2023-05-01')
    ylim = 1.6

    # 创建图形 (2行4列)
    fig = plt.figure(figsize=(18, 10))
    gs = plt.GridSpec(2, 4, hspace=0.3, wspace=1.2)
    ax1 = fig.add_subplot(gs[0, 1:3])  # 观测数据放在第一行左边
    ax2 = fig.add_subplot(gs[1, 0:2])  # old模拟结果放在第一行右边
    ax3 = fig.add_subplot(gs[1, 2:])  # new模拟结果放在第二行中间

    # 读取模拟数据
    sim_data_old = read_nc_moisture(
        f'{model_path}/MAQU_model_output_old.nc',
        *dates
    )
    sim_data_new = read_nc_moisture(
        f'{model_path}/MAQU_model_output_new.nc',
        *dates
    )
    
    # 读取观测数据
    obs_data = read_observed_moisture(
        f'{obs_path}/MAQU_soil_moisture.csv',
        *dates, depths
    )

    # 处理时间范围
    time_obs = pd.date_range(start=dates[0], end=dates[1])
    time_sim = pd.date_range(start=dates[0], end=dates[1])
    
    # 创建网格
    obs_depths = np.array(depths) / 100
    time_grids = [
        np.meshgrid(time_obs, obs_depths)[0],
        np.meshgrid(time_sim, sim_data_old[1])[0],
        np.meshgrid(time_sim, sim_data_new[1])[0]
    ]
    depth_grids = [
        np.meshgrid(time_obs, obs_depths)[1],
        np.meshgrid(time_sim, sim_data_old[1])[1],
        np.meshgrid(time_sim, sim_data_new[1])[1]
    ]
    
    # 准备数据
    data_list = [obs_data, sim_data_old[0].values.T, sim_data_new[0].values.T]
    axes = [ax1, ax2, ax3]
    labels = ['a', 'b', 'c']
    
    # 设置统一的levels
    levels = np.arange(0, 0.51, 0.05)  # 土壤湿度范围0-0.5 m³/m³,间隔0.05
    
    # 绘制三个子图
    for i in range(3):
        row_idx = 0 if i < 1 else 1
        plot_moisture_profile(axes[i], time_grids[i], depth_grids[i],
            data_list[i], labels[i], row_idx, levels, ylim, obs_data if i > 0 else None)

    plt.savefig(f'{output_path}/fig13_old&new_scheme-Soil_Moisture_Profile.jpg',
                bbox_inches='tight', dpi=300)
    plt.show()
