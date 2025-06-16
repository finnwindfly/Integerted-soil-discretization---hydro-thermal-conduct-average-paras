import os  
import numpy as np  
import pandas as pd  
import xarray as xr  
import matplotlib.pyplot as plt  
from datetime import timedelta  
from cftime import DatetimeNoLeap  
import matplotlib.colors as mcolors   
import matplotlib.ticker as ticker  
import matplotlib.dates as mdates  
import string
import cmaps

def read_nc_moisture(nc_file, start_date_str, end_date_str):
    """读取并处理模拟的土壤湿度数据"""
    # 转换日期格式为DatetimeNoLeap对象
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str, format='mixed', dayfirst=True).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str, format='mixed', dayfirst=True).timetuple()[:3])
    )
    
    # 读取数据并转换湿度单位
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
    data['Timestamp'] = pd.to_datetime(data.iloc[:, 0], format='mixed', dayfirst=True)
    data.set_index('Timestamp', inplace=True)
    
    # 转换日期字符串为datetime对象
    start_dt = pd.to_datetime(start_date_str, format='mixed', dayfirst=True)
    end_dt = pd.to_datetime(end_date_str, format='mixed', dayfirst=True)
    
    # 筛选时间范围并处理异常值
    mask = (data.index >= start_dt) & (data.index <= end_dt)
    moistures = data[mask]
    moistures[moistures > 30] = np.NAN
    
    # 删除2月29日的数据
    leap_day_mask = ~((moistures.index.month == 2) & (moistures.index.day == 29))
    moistures = moistures[leap_day_mask]
    
    print(f"观测湿度数据时间范围: {moistures.index.min()} 到 {moistures.index.max()}")
    print(f"观测湿度数据长度: {len(moistures)}")
    
    return moistures.T

def plot_moisture_profile(ax, time_grid, depth_grid, data, label, site_index, col_index, color_norm, ylim, ctl_time):
    """绘制单个土壤湿度剖面图"""
    # 绘制等值线填充图
    c = ax.contourf(time_grid, depth_grid, data, levels=20, cmap=cmaps.MPL_YlGnBu, norm=color_norm, extend='both')
    
    # 设置标题和轴
    ax.set_title(f'({label})', loc='left', fontsize=20)
    ax.set_ylim(0, ylim)
    ax.invert_yaxis()

    # 设置y轴刻度
    yticks = np.linspace(0, ylim, 5)  # 生成5个均匀分布的刻度
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontsize=18)
    
    # 为第一列添加y轴标签
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Soil depth (m)', fontsize=20)
        
    ax.tick_params(axis='both', labelsize=18)
    
    # 只在每行的最后一列（第4列）添加colorbar
    if col_index == 3:  # 第4列（索引为3）
        cbar = plt.colorbar(c, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.set_title('m³/m³', fontsize=18, pad=10)
        cbar.ax.title.set_position((0.85, 1.0))
        cbar.formatter = ticker.FormatStrFormatter('%.2f')
        cbar.update_ticks()
    
    # 从CTL时间轴生成每两个月的刻度
    # 转换CTL时间为pandas datetime
    ctl_times_pd = pd.to_datetime([str(t) for t in ctl_time.values])
    
    # 找到9月1日的起始点
    start_time = ctl_times_pd[0]
    
    # 生成从9月开始每两个月的时间点
    tick_times = []
    
    # 手动生成每两个月的时间点：9月、11月、1月、3月、5月、7月、9月
    months_to_show = [9, 11, 1, 3, 5, 7, 9]  # 从9月开始的月份序列
    
    for i, target_month in enumerate(months_to_show):
        if i < 2:  # 9月和11月是第一年
            target_year = start_time.year
        else:  # 其他月份是第二年
            target_year = start_time.year + 1
            
        # 找到最接近目标月份第一天的时间点
        target_date = pd.Timestamp(year=target_year, month=target_month, day=1)
        
        # 在CTL时间序列中找到最接近的时间点
        closest_idx = np.argmin(np.abs(ctl_times_pd - target_date))
        closest_time = ctl_times_pd[closest_idx]
        
        # 确保这个时间点在合理范围内（不超过15天的差异）
        if abs((closest_time - target_date).days) <= 15:
            tick_times.append(closest_time)
    
    # 设置x轴刻度
    if tick_times:
        ax.set_xticks(tick_times)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # 只有最后一行显示x轴标签，其他行只显示刻度
    if site_index != 6:
        ax.set_xticklabels([])

    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    ax.tick_params(axis='x', labelsize=18, which='major')

def plot_all_profiles(obs_data, ctl_data, sep1_data, sep2_data,
                        obs_depths, ctl_depths, sep1_depths, sep2_depths,
                        site_index, ax, start_date, end_date, labels, ylim):
    """绘制所有土壤湿度剖面图"""
    
    # 打印调试信息
    print(f"\n站点 {site_index + 1} 湿度时间信息:")
    print(f"起始日期: {start_date}")
    print(f"结束日期: {end_date}")
    print(f"CTL湿度数据时间范围: {ctl_data.time.values[0]} 到 {ctl_data.time.values[-1]}")
    print(f"CTL湿度数据长度: {len(ctl_data.time)}")
    
    # 处理观测时间范围 - 从start_date到end_date，删除闰日
    time_obs = pd.date_range(start=start_date, end=end_date, freq='D')
    leap_day_mask_obs = ~((time_obs.month == 2) & (time_obs.day == 29))
    time_obs = time_obs[leap_day_mask_obs]
    
    # 处理模拟时间范围 - 基于CTL数据的实际时间
    ctl_times_pd = pd.to_datetime([str(t) for t in ctl_data.time.values])
    time_sim = ctl_times_pd
    
    print(f"处理后观测湿度时间长度: {len(time_obs)}")
    print(f"处理后模拟湿度时间长度: {len(time_sim)}")
    
    # 确保数据长度匹配
    min_obs_len = min(len(time_obs), obs_data.shape[1])
    min_sim_len = min(len(time_sim), len(ctl_data.time))
    
    time_obs = time_obs[:min_obs_len]
    time_sim = time_sim[:min_sim_len]
    obs_data_trimmed = obs_data.iloc[:, :min_obs_len]
    
    # 创建网格
    time_grids = [
        np.meshgrid(time_obs, obs_depths)[0],
        np.meshgrid(time_sim, ctl_depths)[0],
        np.meshgrid(time_sim, sep1_depths)[0],
        np.meshgrid(time_sim, sep2_depths)[0]
    ]
    depth_grids = [
        np.meshgrid(time_obs, obs_depths)[1],
        np.meshgrid(time_sim, ctl_depths)[1],
        np.meshgrid(time_sim, sep1_depths)[1],
        np.meshgrid(time_sim, sep2_depths)[1]
    ]
    data_list = [obs_data_trimmed.values, ctl_data.values.T, sep1_data.values.T, sep2_data.values.T]
    
    # 统一色彩规范 - 确保每个站点所有列都使用相同的norm
    norm = mcolors.Normalize(vmin=0, vmax=0.4)
    
    # 绘制四个子图
    for i in range(4):
        label = labels[site_index * 4 + i]
        plot_moisture_profile(ax[site_index, i], time_grids[i], depth_grids[i], 
                                data_list[i], label, site_index, i, norm, ylim, ctl_data.time)

if __name__ == "__main__":
    # 基础路径配置
    base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0'
    
    # 站点配置 - 确保所有日期都是9月1日到次年9月1日
    sites = {
        'NAMORS': {'depths': [10, 20, 40, 80], 'dates': ('2015-09-01', '2016-09-01'), 'ylim': 3.2},
        'MADUO': {'depths': [10, 40, 80, 160], 'dates': ('2017-09-01', '2018-09-01'), 'ylim': 4.0},
        'MAQU': {'depths': [10, 40, 80, 160], 'dates': ('2022-09-01', '2023-09-01'), 'ylim': 2.0},
        'Ngari': {'depths': [0, 20, 50, 100], 'dates': ('2012-09-01', '2013-09-01'), 'ylim': 3.0},
        'Qoms': {'depths': [10, 20, 40, 80], 'dates': ('2015-09-01', '2016-09-01'), 'ylim': 1.6},
        'SETORS': {'depths': [10, 20, 60, 100], 'dates': ('2009-09-01', '2010-09-01'), 'ylim': 1.5},
        'YAKOU': {'depths': [10, 20, 40, 80], 'dates': ('2018-09-01', '2019-09-01'), 'ylim': 3.0}
    }

    # 创建图形
    fig, ax = plt.subplots(nrows=len(sites), ncols=4, figsize=(22, 3.3 * len(sites)))
    
    # 生成标签
    labels = [f"{row}{col}" for row in string.ascii_lowercase[:len(sites)] 
            for col in ['1', '2', '3', '4']]  # 修改标签序号

    # 处理每个站点数据并绘图
    for i, (site, info) in enumerate(sites.items()):
        print(f"\n\n=== 处理湿度站点: {site} ===")
        
        # 读取模拟数据
        sim_data = [read_nc_moisture(
            f'{base_path}/data/model_output/{site}/final_use/run/{site}_model_output_{suffix}.nc',
            *info['dates']
        ) for suffix in ['CTL', 'SEP1', 'SEP2']]
        
        # 读取观测数据
        obs_data = read_observed_moisture(
            f'{base_path}/data/Stations_soil_data/{site}/{site}_soil_moisture.csv',
            *info['dates'], info['depths']
        )
        
        # 绘图
        plot_all_profiles(obs_data, *[data[0] for data in sim_data],
                         np.array(info['depths']) / 100, *[data[1] for data in sim_data],
                         i, ax, *[pd.to_datetime(d, format='%Y-%m-%d') for d in info['dates']],
                        labels, info['ylim'])

    plt.tight_layout()
    plt.savefig(f'{base_path}/fig/202506/fig8_Soil_Moisture_Profile_CTL&SEP1&SEP2_nobedrock.jpg',
                bbox_inches='tight', dpi=300)
    plt.show()
