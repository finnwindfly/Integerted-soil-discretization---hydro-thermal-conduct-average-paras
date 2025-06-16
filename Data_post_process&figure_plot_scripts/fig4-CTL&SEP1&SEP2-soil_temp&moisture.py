import os  
import numpy as np  
import pandas as pd  
import xarray as xr  
import matplotlib.pyplot as plt  
from datetime import timedelta  
from cftime import DatetimeNoLeap  
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

def read_nc_temperature(nc_file, depths, start_date_str, end_date_str):  
    """读取并插值NC文件中的土壤温度数据"""
    # 转换日期格式为DatetimeNoLeap对象
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str, format='mixed', dayfirst=True).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str, format='mixed', dayfirst=True).timetuple()[:3])
    )
    
    # 读取数据并转换为摄氏度
    temp = xr.open_dataset(nc_file).TSOI.isel(lndgrid=0) - 273.15
    
    # 对每个深度进行插值并返回列表
    return [temp.sel(time=time_slice).isel(levgrnd=0) if depth == 0 
            else temp.sel(time=time_slice).interp(levgrnd=depth/100.0) 
            for depth in depths]

def read_nc_moisture(nc_file, depths, start_date_str, end_date_str):  
    """读取并插值NC文件中的土壤湿度数据"""
    # 转换日期格式为DatetimeNoLeap对象
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str, format='mixed', dayfirst=True).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str, format='mixed', dayfirst=True).timetuple()[:3])
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

def read_observed_temperature(csv_file, start_date_str, end_date_str):  
    """读取观测数据CSV文件"""
    # 读取数据并设置时间索引
    data = pd.read_csv(csv_file)
    data.index = pd.to_datetime(data.iloc[:, 0], format='mixed', dayfirst=True)
    
    # 筛选时间范围并处理异常值(>30℃)
    mask = (data.index >= pd.to_datetime(start_date_str, format='mixed', dayfirst=True)) & \
        (data.index <= pd.to_datetime(end_date_str, format='mixed', dayfirst=True))
    data = data[mask].apply(pd.to_numeric, errors='coerce')
    data[data > 30] = pd.NA
    
    # 删除2月29日的数据
    leap_day_mask = ~((data.index.month == 2) & (data.index.day == 29))
    data = data[leap_day_mask]
    
    return data

def read_observed_moisture(csv_file, start_date_str, end_date_str):  
    """读取观测数据CSV文件"""
    # 读取数据并设置时间索引
    data = pd.read_csv(csv_file)
    data.index = pd.to_datetime(data.iloc[:, 0], format='mixed', dayfirst=True)
    
    # 筛选时间范围并处理异常值(>30)
    mask = (data.index >= pd.to_datetime(start_date_str, format='mixed', dayfirst=True)) & \
        (data.index <= pd.to_datetime(end_date_str, format='mixed', dayfirst=True))
    data = data[mask].apply(pd.to_numeric, errors='coerce')
    data[data > 30] = pd.NA
    
    # 删除2月29日的数据
    leap_day_mask = ~((data.index.month == 2) & (data.index.day == 29))
    data = data[leap_day_mask]
    
    return data

def get_frozen_periods(temp_data, time_index):
    """计算冻结期"""
    frozen_mask = temp_data < 0
    frozen_periods = []
    start_idx = None
    
    for i in range(len(frozen_mask)):
        if frozen_mask[i] and start_idx is None:
            start_idx = i
        elif not frozen_mask[i] and start_idx is not None:
            frozen_periods.append((time_index[start_idx], time_index[i-1]))
            start_idx = None
            
    # 处理最后一段冻结期
    if start_idx is not None:
        frozen_periods.append((time_index[start_idx], time_index[-1]))
        
    return frozen_periods

def plot_temp_moisture_comparison(ax, site_data, site_index, depth_labels, total_sites, site_ylim):
    """绘制温度湿度对比图并分析冻结期"""
    # 解包数据
    sim_temp, sim_moist, obs_temp, obs_moist, dates, depths = site_data
    start_date, end_date = dates
    
    # 生成时间序列并删除2月29日
    time_obs = pd.date_range(start_date, end_date)
    leap_day_mask = ~((time_obs.month == 2) & (time_obs.day == 29))
    time_obs = time_obs[leap_day_mask]
    
    # 模拟数据时间序列
    time_sim_raw = pd.date_range(start_date, end_date - timedelta(days=1)) if site_index in [0, 4] else pd.date_range(start_date, end_date)
    leap_day_mask_sim = ~((time_sim_raw.month == 2) & (time_sim_raw.day == 29))
    time_sim = time_sim_raw[leap_day_mask_sim]
    
    # 颜色和标签配置
    temp_colors = ['#4682B4', '#CD5C5C', '#F4A460']
    moist_colors = ['#4682B4', '#CD5C5C', '#F4A460'] 
    obs_color = 'black'  # 改为黑色
    labels = ['CTL', 'SEP1', 'SEP2']
    
    # 计算月份起始点
    month_starts = pd.date_range(start=time_obs[0].replace(day=1), end=time_obs[-1], freq='MS')
    
    # 分析并打印冻结期
    print(f"\n站点 {site_index + 1} 的冻结期分析:")
    print(f"模拟数据时间长度: {len(time_sim)}")
    print(f"观测数据时间长度: {len(obs_temp)}")
    print(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    
    # 绘制每个深度的子图
    for i, depth in enumerate(depths):
        print(f"\n深度 {depth}cm:")
        
        # 创建双y轴
        ax_t = ax[site_index, i]  # 温度轴
        ax_m = ax_t.twinx()       # 湿度轴
        
        # 检查数据长度并进行对齐
        min_len = min(len(time_sim), len(obs_temp), len(obs_moist))
        if len(time_sim) > min_len:
            time_sim_plot = time_sim[:min_len]
        else:
            time_sim_plot = time_sim
            
        # 检查模拟数据长度
        for j in range(3):
            sim_temp_len = len(sim_temp[j][i])
            sim_moist_len = len(sim_moist[j][i])
            print(f"{labels[j]} 温度数据长度: {sim_temp_len}, 湿度数据长度: {sim_moist_len}")
        
        # 分析每个模拟方案的冻结期
        for j, scheme in enumerate(['CTL', 'SEP1', 'SEP2']):
            # 确保数据长度一致
            temp_data = sim_temp[j][i][:min_len] if len(sim_temp[j][i]) > min_len else sim_temp[j][i]
            frozen_periods = get_frozen_periods(temp_data, time_sim_plot[:len(temp_data)])
            if frozen_periods:
                print(f"\n{scheme}方案冻结期:")
                for start, end in frozen_periods:
                    print(f"从 {start.strftime('%Y-%m-%d')} 到 {end.strftime('%Y-%m-%d')}")
        
        # 分析观测数据的冻结期
        obs_temp_values = obs_temp.iloc[:min_len, i+1] if len(obs_temp) > min_len else obs_temp.iloc[:, i+1]
        obs_frozen_periods = get_frozen_periods(obs_temp_values.values, obs_temp.index[:len(obs_temp_values)])
        if obs_frozen_periods:
            print("\n观测数据冻结期:")
            for start, end in obs_frozen_periods:
                print(f"从 {start.strftime('%Y-%m-%d')} 到 {end.strftime('%Y-%m-%d')}")
        
        # 绘制温度数据（实线）
        for j in range(3):
            temp_data = sim_temp[j][i][:min_len] if len(sim_temp[j][i]) > min_len else sim_temp[j][i]
            time_plot = time_sim_plot[:len(temp_data)]
            ax_t.plot(time_plot, temp_data, color=temp_colors[j], linestyle='-',
                     label=labels[j] if site_index == 0 and i == 0 else "", 
                     linewidth=3.0, zorder=2)
        
        obs_temp_plot = obs_temp.iloc[:min_len, i+1] if len(obs_temp) > min_len else obs_temp.iloc[:, i+1]
        ax_t.plot(obs_temp.index[:len(obs_temp_plot)], obs_temp_plot, color=obs_color, linestyle='-',
                 label='obs' if site_index == 0 and i == 0 else "",
                 linewidth=3.0, zorder=3)
        
        # 绘制湿度数据（虚线）
        for j in range(3):
            moist_data = sim_moist[j][i][:min_len] if len(sim_moist[j][i]) > min_len else sim_moist[j][i]
            time_plot = time_sim_plot[:len(moist_data)]
            ax_m.plot(time_plot, moist_data, color=moist_colors[j], linestyle='--',
                     label=labels[j] if site_index == 1 and i == 0 else "", 
                     linewidth=3.0, zorder=1)
        
        obs_moist_plot = obs_moist.iloc[:min_len, i+1] if len(obs_moist) > min_len else obs_moist.iloc[:, i+1]
        ax_m.plot(obs_moist.index[:len(obs_moist_plot)], obs_moist_plot, color=obs_color, linestyle='--',
                 label='obs' if site_index == 1 and i == 0 else "",
                 linewidth=3.0, zorder=1)
        
        # 设置温度轴样式 - 使用统一的ylim
        ax_t.axhline(y=0, color='gray', linestyle='-.')
        temp_ylim = site_ylim['temp']
        ax_t.set_ylim(temp_ylim)
        yticks = np.linspace(temp_ylim[0], temp_ylim[1], 5)
        ax_t.set_yticks(yticks)
        ax_t.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
        # 设置湿度轴样式 - 使用统一的ylim
        moist_ylim = site_ylim['moist']
        ax_m.set_ylim(moist_ylim)
        moist_yticks = np.linspace(moist_ylim[0], moist_ylim[1], 5)
        ax_m.set_yticks(moist_yticks)
        ax_m.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        
        # 设置标题 - 增大字体
        ax_t.set_title(f'({depth_labels[site_index]}{i+1}) {depth} cm', loc='left', fontsize=18)
        
        # 设置y轴标签 - 增大字体
        if i == 0:
            ax_t.set_ylabel('Temperature (°C)', fontsize=18)
            ax_m.set_ylabel('')
        elif i == len(depths) - 1:
            ax_t.set_ylabel('')
            ax_m.set_ylabel(r'Moisture ($mm^3/mm^3$)', fontsize=18)
        else:
            ax_t.set_ylabel('')
            ax_m.set_ylabel('')
        
        # 设置y轴刻度标签显示 - 增大字体
        if i != 0:
            ax_t.set_yticklabels([])
        if i != len(depths) - 1:
            ax_m.set_yticklabels([])
            
        ax_t.tick_params(axis='y', labelsize=16, width=2, length=6)
        ax_m.tick_params(axis='y', labelsize=16, width=2, length=6)
        
        # 设置x轴
        ax_t.set_xlim(start_date, end_date)
        ax_t.set_xticks(month_starts)
        ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_t.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_t.xaxis.set_minor_locator(ticker.NullLocator())
        
        # X轴标签显示逻辑 - 增大字体
        if site_index != (total_sites - 1):
            ax_t.set_xticklabels([])
        ax_t.tick_params(axis='x', labelsize=16, width=2, length=6, which='major', 
                        labelbottom=(site_index==(total_sites-1)))
        
        # 设置图例 - 增大字体
        if site_index == 0 and i == 0:
            temp_handles, temp_labels = ax_t.get_legend_handles_labels()
            ax_t.legend(temp_handles, temp_labels, loc='upper center', fontsize=14, frameon=False)
        if site_index == 1 and i == 0:
            moist_handles, moist_labels = ax_m.get_legend_handles_labels()
            ax_m.legend(moist_handles, moist_labels, loc='upper center', fontsize=14, frameon=False)
        
        # 边框加粗
        for spine in ax_t.spines.values():
            spine.set_linewidth(2)
        for spine in ax_m.spines.values():
            spine.set_linewidth(2)

if __name__ == "__main__":
    # 基础路径配置
    base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0'
    
    # 站点配置
    sites = {
        'NAMORS': {'depths': [10, 20, 40, 80], 'dates': ('2015-10-01', '2016-06-01')},
        'MADUO': {'depths': [10, 40, 80, 160], 'dates': ('1/10/2017', '1/6/2018')},
        'MAQU': {'depths': [10, 40, 80, 160], 'dates': ('1/10/2022', '1/6/2023')},
        'Ngari': {'depths': [0, 20, 50, 100], 'dates': ('1/10/2012', '1/6/2013')},
        'Qoms': {'depths': [10, 20, 40, 80], 'dates': ('1/10/2015', '1/6/2016')},
        'SETORS': {'depths': [10, 20, 60, 100], 'dates': ('2009-10-01', '2010-06-01')},
        'YAKOU': {'depths': [10, 20, 40, 80], 'dates': ('2018-10-01', '2019-06-01')}
    }

    # 每个站点的统一ylim配置
    site_ylims = {
        'NAMORS': {'temp': (-20, 15), 'moist': (0.0, 0.5)},
        'MADUO': {'temp': (-20, 15), 'moist': (0.0, 0.4)},
        'MAQU': {'temp': (-25, 20), 'moist': (0.0, 0.7)},
        'Ngari': {'temp': (-25, 25), 'moist': (0.0, 0.3)},
        'Qoms': {'temp': (-20, 15), 'moist': (0.0, 0.4)},
        'SETORS': {'temp': (-20, 15), 'moist': (0.05, 0.7)},
        'YAKOU': {'temp': (-25, 15), 'moist': (0.05, 0.6)}
    }

    # 深度标签
    depth_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    
    # 创建图形
    fig, ax = plt.subplots(nrows=len(sites), ncols=4, figsize=(20, 3 * len(sites)))
    fig.subplots_adjust(wspace=0.14)

    # 处理每个站点数据并绘图
    for i, (site, info) in enumerate(sites.items()):
        print(f"\n\n分析站点: {site}")
        
        # 读取模拟温度数据
        sim_temp = [read_nc_temperature(
            f'{base_path}/data/model_output/{site}/final_use/run/{site}_model_output_{suffix}.nc',
            info['depths'], *info['dates']
        ) for suffix in ['CTL', 'SEP1', 'SEP2']]
        
        # 读取模拟湿度数据
        sim_moist = [read_nc_moisture(
            f'{base_path}/data/model_output/{site}/final_use/run/{site}_model_output_{suffix}.nc',
            info['depths'], *info['dates']
        ) for suffix in ['CTL', 'SEP1', 'SEP2']]
        
        # 读取观测温度数据
        obs_temp = read_observed_temperature(
            f'{base_path}/data/Stations_soil_data/{site}/{site}_soil_temperature.csv',
            *info['dates']
        )
        
        # 读取观测湿度数据
        obs_moist = read_observed_moisture(
            f'{base_path}/data/Stations_soil_data/{site}/{site}_soil_moisture.csv',
            *info['dates']
        )
        
        # 绘图
        site_data = (sim_temp, sim_moist, obs_temp, obs_moist,
                    [pd.to_datetime(d, format='mixed', dayfirst=True) for d in info['dates']], 
                    info['depths'])
        plot_temp_moisture_comparison(ax, site_data, i, depth_labels, len(sites), site_ylims[site])

    plt.tight_layout()
    plt.savefig(f'{base_path}/fig/202506/fig4_combined_Temperature_Moisture_CTL&SEP1&SEP2_nobedrock.jpg', 
                bbox_inches='tight', dpi=300)
    plt.show()
