import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
from cftime import DatetimeNoLeap
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

def read_nc_moisture(nc_file, depths, start_date_str, end_date_str):  
    """读取并插值NC文件中的土壤湿度数据"""
    # 转换日期格式为DatetimeNoLeap对象
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str, format ='mixed', dayfirst=True).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str, format= 'mixed', dayfirst=True).timetuple()[:3])
    )
    
    # 读取数据并计算体积含水量
    data = xr.open_dataset(nc_file)
    moisture = data.SOILLIQ.isel(lndgrid=0).rename({'levsoi': 'levgrnd'})
    thick = data.DZSOI.isel(lndgrid=0).isel(levgrnd=slice(None, -5))
    moisture_v = moisture / thick / 1000.0
    
    # 对每个深度进行插值并返回列表
    return [moisture_v.sel(time=time_slice).isel(levgrnd=0) if depth == 0 
            else moisture_v.sel(time=time_slice).interp(levgrnd=depth/100.0) 
            for depth in depths]

def read_observed_moisture(csv_file, start_date_str, end_date_str):
    """读取观测数据CSV文件"""
    # 读取数据并设置时间索引
    data = pd.read_csv(csv_file)
    data.index = pd.to_datetime(data.iloc[:, 0])
    
    # 筛选时间范围并处理异常值(>1或<0)
    mask = (data.index >= pd.to_datetime(start_date_str)) & \
            (data.index <= pd.to_datetime(end_date_str))
    data = data[mask].apply(pd.to_numeric, errors='coerce')
    data[(data > 1) | (data < 0)] = pd.NA
    
    return data

def calculate_ts_score(sim_data, obs_data):
    """计算Taylor技巧评分(TS)"""
    if isinstance(sim_data, xr.DataArray):
        sim_data = sim_data.values
        
    # 去除NaN值
    mask = ~np.isnan(obs_data)
    sim = sim_data[mask]
    obs = obs_data[mask]
    
    # 计算TC (时间相关系数R)
    R = np.corrcoef(sim, obs)[0,1]
    
    # 计算标准差比值 σ
    sim_std = np.std(sim)
    obs_std = np.std(obs)
    sigma = sim_std / obs_std
    
    # 计算TS，R0设为1
    R0 = 1
    ts = (4 * (1 + R)) / ((sigma + 1/sigma)**2 * (1 + R0))
    
    return ts

def plot_moisture_comparison(ax, sim_data_old, sim_data_new, obs_data, depths, dates):
    """绘制土壤湿度对比图并计算TS评分"""
    start_date, end_date = dates
    time_range = pd.date_range(start_date, end_date)
    
    colors = ['#4682B4', '#CD5C5C', '#423c40']
    labels = ['Old', 'New', 'Obs']
    
    for i, depth in enumerate(depths):
        row = i // 4
        col = i % 4
        
        # 设置spine宽度
        for spine in ax[row, col].spines.values():
            spine.set_linewidth(2.0)
        
        # 绘制模拟数据和观测数据
        ax[row, col].plot(time_range, sim_data_old[i], color=colors[0], 
                        label=labels[0], linewidth=2.0)
        ax[row, col].plot(time_range, sim_data_new[i], color=colors[1], 
                        label=labels[1], linewidth=2.0)
        ax[row, col].plot(time_range, obs_data.iloc[:, i+1], color=colors[2], 
                        label=labels[2], linewidth=2.0)
        
        ax[row, col].set_ylim(0, 0.5)
        
        # 计算TS评分
        ts_old = calculate_ts_score(sim_data_old[i], obs_data.iloc[:, i+1].values)
        ts_new = calculate_ts_score(sim_data_new[i], obs_data.iloc[:, i+1].values)
        
        # 添加TS评分文本
        if row == 0:
            ax[row, col].text(0.56, 0.25, f'TS = {ts_old:.3f}', transform=ax[row, col].transAxes,
                            color=colors[0], fontsize=14)
            ax[row, col].text(0.56, 0.15, f'TS = {ts_new:.3f}', transform=ax[row, col].transAxes,
                            color=colors[1], fontsize=14)
        else:
            ax[row, col].text(0.56, 0.9, f'TS = {ts_old:.3f}', transform=ax[row, col].transAxes,
                            color=colors[0], fontsize=14)
            ax[row, col].text(0.56, 0.8, f'TS = {ts_new:.3f}', transform=ax[row, col].transAxes,
                            color=colors[1], fontsize=14)
        
        # 设置标题(带字母标记)
        subplot_label = chr(97 + i)  # 97是字母'a'的ASCII码
        ax[row, col].set_title(f'({subplot_label}) {depth} cm', loc='left', fontsize=14)
        
        # 设置y轴标签和刻度
        if col == 0:
            ax[row, col].set_ylabel('Soil Moisture (m³/m³)', fontsize=14)
        ax[row, col].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        # 设置x轴刻度格式和数量
        months = mdates.MonthLocator(interval=2)  # 每2个月显示一个刻度
        ax[row, col].xaxis.set_major_locator(months)
        ax[row, col].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        ax[row, col].tick_params(axis='both', labelsize=14, width=2.0)
        ax[row, col].set_xlim(start_date, end_date)
        
        if i == 8:
            ax[row, col].legend(frameon=False, loc='upper left', fontsize=14)
                

if __name__ == "__main__":
    # 基础路径配置
    base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/data/model_output/40SL_discret_scheme/MAQU'
    obs_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/data/Stations_soil_data/new_scheme/MAQU12SL'
    output_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/fig/202506'
    
    # 配置参数
    depths = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160]  # 根据MAQU站观测数据处理.py中的深度配置
    dates = ('2022-10-01', '2023-09-01')
    
    # 读取模拟数据
    sim_data_old = read_nc_moisture(
        f'{base_path}/MAQU_model_output_old.nc',
        depths, *dates
    )
    sim_data_new = read_nc_moisture(
        f'{base_path}/MAQU_model_output_new.nc',
        depths, *dates
    )
    
    # 读取观测数据
    obs_data = read_observed_moisture(
        f'{obs_path}/MAQU_soil_moisture.csv',
        *dates
    )
    
    # 创建3x4的图形
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(14, 9))
    
    # 绘图
    plot_moisture_comparison(ax, sim_data_old, sim_data_new, obs_data, depths, 
                            [pd.to_datetime(d) for d in dates])
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig11_old&new_scheme-Soil_Moisture-restart.jpg', bbox_inches='tight', dpi=300)
    plt.show()
