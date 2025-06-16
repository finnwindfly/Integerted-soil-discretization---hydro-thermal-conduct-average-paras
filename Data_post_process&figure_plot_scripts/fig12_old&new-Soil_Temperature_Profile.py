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

def read_nc_temperature(nc_file, start_date_str, end_date_str):
    """读取并处理模拟的土壤温度数据"""
    # 转换日期格式为DatetimeNoLeap对象
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str).timetuple()[:3])
    )
    
    # 读取数据并转换温度单位
    data = xr.open_dataset(nc_file)
    temperature = data.TSOI.isel(lndgrid=0).sel(time=time_slice) - 273.15  # 转换为摄氏度
    depth_levels = data.ZSOI.isel(lndgrid=0)  # 获取深度信息
    
    return temperature, depth_levels

def read_observed_temperature(csv_file, start_date_str, end_date_str, depths):
    """读取并处理观测的土壤温度数据"""
    # 读取数据并设置时间索引
    data = pd.read_csv(csv_file)
    data.index = pd.to_datetime(data.iloc[:, 0])
    
    # 筛选时间范围并处理异常值
    mask = (data.index >= pd.to_datetime(start_date_str)) & \
            (data.index <= pd.to_datetime(end_date_str))
    temperatures = data[mask]
    
    # 确保温度数据为数值类型
    numeric_columns = temperatures.select_dtypes(include=[np.number]).columns
    temperatures = temperatures[numeric_columns]
    
    # 处理异常值
    temperatures = temperatures.apply(pd.to_numeric, errors='coerce')  # 将非数值转换为NaN
    temperatures[temperatures > 30] = np.NAN
    
    return temperatures.T

def plot_temperature_profile(ax, time_grid, depth_grid, data, label, row_idx, levels, ylim, obs_data=None):
    """绘制单个土壤温度剖面图"""
    # 绘制等值线填充图
    norm = mpl.colors.BoundaryNorm(levels, cmaps.BlueYellowRed.N, extend='both')
    c = ax.contourf(time_grid, depth_grid, data, 
                    cmap=cmaps.BlueYellowRed, norm=norm, levels=levels, extend='both')
    
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
    
    # 绘制0°C等温线
    cs = ax.contour(time_grid, depth_grid, data, levels=[0], colors='k', linewidths=1.0)
    plt.clabel(cs, inline=True, fontsize=10)  # 移除fontname参数
    
    # 如果提供了观测数据，绘制观测的0°C等温线
    if obs_data is not None:
        # 创建插值网格
        time_1d = time_grid[0]
        depth_1d = depth_grid[:, 0]
        time_mesh, depth_mesh = np.meshgrid(time_1d, depth_1d)
        
        # 准备插值的坐标点
        obs_time_1d = pd.date_range(start=time_1d[0], end=time_1d[-1], periods=obs_data.shape[1])
        obs_depth_1d = np.array([5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160]) / 100
        obs_time_mesh, obs_depth_mesh = np.meshgrid(obs_time_1d, obs_depth_1d)
        
        # 将时间转换为数值进行插值
        time_num = mdates.date2num(time_mesh)
        obs_time_num = mdates.date2num(obs_time_mesh)
        
        # 准备插值的点
        points = np.column_stack((obs_time_num.ravel(), obs_depth_mesh.ravel()))
        values = obs_data.to_numpy().ravel()  # 使用to_numpy()替代直接ravel()
        
        # 移除NaN值
        mask = ~np.isnan(values)
        points = points[mask]
        values = values[mask]
        
        # 进行插值
        grid_z = griddata(points, values, (time_num, depth_mesh), method='linear')
        
        # 绘制插值后的0°C等温线
        obs_cs = ax.contour(time_grid, depth_grid, grid_z, levels=[0], colors='gray', linestyles='dashed', linewidths=1.0)
        plt.clabel(obs_cs, inline=True, fontsize=10)  # 移除fontname参数

    # 为每个图形添加colorbar
    cax = inset_axes(ax, width="5%", height="100%", loc='right',
                bbox_to_anchor=(0.1, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = plt.colorbar(c, cax=cax, orientation='vertical', ticks=levels[::2])
    cb.ax.tick_params(labelsize=18, direction='out', width=2.0, length=6.0)
    cb.outline.set_linewidth(1.5)
    cax.yaxis.set_ticks_position('right')
    cb.ax.text(0.5, 1.15, '°C', fontsize=18, ha='center', va='top', transform=cb.ax.transAxes, fontname='Helvetica')
    
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
    sim_data_old = read_nc_temperature(
        f'{model_path}/MAQU_model_output_old.nc',
        *dates
    )
    sim_data_new = read_nc_temperature(
        f'{model_path}/MAQU_model_output_new.nc',
        *dates
    )
    
    # 读取观测数据
    obs_data = read_observed_temperature(
        f'{obs_path}/MAQU_soil_temperature.csv',
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
    levels = np.arange(-12, 12, 1)
    
    # 绘制三个子图
    for i in range(3):
        row_idx = 0 if i < 1 else 1
        plot_temperature_profile(axes[i], time_grids[i], depth_grids[i],
            data_list[i], labels[i], row_idx, levels, ylim, obs_data if i > 0 else None)

    # plt.tight_layout()
    plt.savefig(f'{output_path}/fig12_old&new_scheme-Soil_Temperature_Profile.jpg',
                bbox_inches='tight', dpi=300)
    plt.show()
