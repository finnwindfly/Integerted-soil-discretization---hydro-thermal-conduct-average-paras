import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
from cftime import DatetimeNoLeap

def read_nc_temperature(nc_file, depths, start_date_str, end_date_str):
    """读取并插值NC文件中的土壤温度数据"""
    # 转换日期格式为DatetimeNoLeap对象
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str, dayfirst=True).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str, dayfirst=True).timetuple()[:3])
    )
    
    # 读取数据并转换为摄氏度
    temp = xr.open_dataset(nc_file).TSOI.isel(lndgrid=0) - 273.15
    
    # 对每个深度进行插值并返回列表
    return [temp.sel(time=time_slice).isel(levgrnd=0) if depth == 0 
            else temp.sel(time=time_slice).interp(levgrnd=depth/100.0) 
            for depth in depths]

def read_observed_temperature(csv_file, start_date_str, end_date_str):
    """读取观测数据CSV文件"""
    # 读取数据并设置时间索引
    data = pd.read_csv(csv_file)
    data.index = pd.to_datetime(data.iloc[:, 0], dayfirst=True)
    
    # 筛选时间范围并处理异常值(>30℃)
    mask = (data.index >= pd.to_datetime(start_date_str, dayfirst=True)) & \
        (data.index <= pd.to_datetime(end_date_str, dayfirst=True))
    data = data[mask].apply(pd.to_numeric, errors='coerce')
    data[data > 30] = pd.NA
    
    # 剔除2月29日的数据
    data = data[~((data.index.month == 2) & (data.index.day == 29))]
    
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

def plot_bar_charts(site_data, site_name, depths, ax, subplot_label=''):
    """绘制柱状图"""
    sim_data, obs_data = site_data[0], site_data[1]
    
    # 计算TS
    stats = []
    for depth_idx, depth in enumerate(depths):
        depth_stats = []
        for scheme_idx in range(3):
            ts = calculate_ts_score(
                sim_data[scheme_idx][depth_idx],
                obs_data.iloc[:, depth_idx+1]
            )
            depth_stats.append(ts)
        stats.append(depth_stats)
    
    # 设置柱状图参数
    x = np.arange(len(depths))
    width = 0.25
    colors = ['#4682B4', '#CD5C5C', '#F4A460']
    labels = ['CTL', 'SEP1', 'SEP2']
    
    # 绘制柱状图
    bars = []
    for i in range(3):
        bar = ax.bar(x + i*width - width, [s[i] for s in stats], width, 
                color=colors[i], alpha=0.8)
        bars.append(bar)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)
    
    # 设置图形属性
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}cm' for d in depths], fontsize=12)
    
    # 修改站点名称显示
    display_name = site_name
    if site_name == 'Ngari':
        display_name = 'NADORS'
    elif site_name == 'Qoms':
        display_name = 'QOMS'
    ax.set_title(f'({subplot_label}) {display_name}', loc='left', fontsize=14, pad=10)
    
    ax.set_ylabel('Taylor Skill Score', fontsize=12)
    ax.set_ylim(0, 1)  # TS的范围是0-1
    
    # 设置y轴刻度字体和格式
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=12, width=1.5, length=4)
    
    # 加粗轴线
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    return bars, labels

def plot_relative_change(site_data, site_name, depths, ax, subplot_label=''):
    """绘制相对变化柱状图"""
    sim_data, obs_data = site_data[0], site_data[1]
    
    # 计算TS和相对变化
    changes_sep1 = []
    changes_sep2 = []
    for depth_idx, depth in enumerate(depths):
        # 计算CTL的TS
        ctl_ts = calculate_ts_score(
            sim_data[0][depth_idx],
            obs_data.iloc[:, depth_idx+1]
        )
        
        # 计算SEP1的TS和相对变化
        sep1_ts = calculate_ts_score(
            sim_data[1][depth_idx],
            obs_data.iloc[:, depth_idx+1]
        )
        changes_sep1.append(((sep1_ts - ctl_ts) / ctl_ts) * 100)
        
        # 计算SEP2的TS和相对变化
        sep2_ts = calculate_ts_score(
            sim_data[2][depth_idx],
            obs_data.iloc[:, depth_idx+1]
        )
        changes_sep2.append(((sep2_ts - ctl_ts) / ctl_ts) * 100)
    
    # 绘制柱状图
    x = np.arange(len(depths))
    width = 0.25  # 与plot_bar_charts保持一致
    
    # 绘制SEP1和SEP2的bar
    bars = []
    bars.append(ax.bar(x - width/2, changes_sep1, width, color='#CD5C5C', alpha=0.8))
    bars.append(ax.bar(x + width/2, changes_sep2, width, color='#F4A460', alpha=0.8))
    
    # 添加水平线表示0%
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.15, linewidth=1.5)
    
    # 设置图形属性
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}cm' for d in depths], fontsize=12)
    
    # 修改站点名称显示
    display_name = site_name
    if site_name == 'Ngari':
        display_name = 'NADORS'
    elif site_name == 'Qoms':
        display_name = 'QOMS'
    ax.set_title(f'({subplot_label}) {display_name}', loc='left', fontsize=14, pad=10)
    
    ax.set_ylabel('Relative Change (%)', fontsize=12)
    
    # 设置y轴刻度字体和格式
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='both', labelsize=12, width=1.5, length=4)
    
    # 加粗轴线
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    return bars, ['SEP1-CTL', 'SEP2-CTL']

if __name__ == "__main__":
    # 基础路径配置
    base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0'
    
    # 站点配置
    sites = {
        'NAMORS': {'depths': [10, 20, 40, 80], 'dates': ('2015-10-01', '2016-05-01')},
        'MADUO': {'depths': [10, 40, 80, 160], 'dates': ('1/10/2017', '1/5/2018')},
        'MAQU': {'depths': [10, 40, 80, 160], 'dates': ('1/10/2022', '1/5/2023')},
        'Ngari': {'depths': [0, 20, 50, 100], 'dates': ('1/10/2012', '1/5/2013')},
        'Qoms': {'depths': [10, 20, 40, 80], 'dates': ('1/10/2015', '1/5/2016')},
        'SETORS': {'depths': [10, 20, 60, 100], 'dates': ('1/10/2009', '1/5/2010')},
        'YAKOU': {'depths': [10, 20, 40, 80], 'dates': ('1/10/2018', '1/5/2019')}
    }

    # 创建两个图形
    figs = []
    for fig_type in ['original', 'relative_change']:
        fig = plt.figure(figsize=(9, 9))
        figs.append(fig)
    
    # 处理每个站点数据并绘制图形
    for i, (site, info) in enumerate(sites.items()):
        if i < 7:  # 只处理前7个站点
            print(f"\n\n分析站点: {site}")
            
            # 读取模拟数据
            sim_data = [read_nc_temperature(
                f'{base_path}/data/model_output/{site}/final_use/run/{site}_model_output_{suffix}.nc',
                info['depths'], *info['dates']
            ) for suffix in ['CTL', 'SEP1', 'SEP2']]
            
            # 读取观测数据
            obs_data = read_observed_temperature(
                f'{base_path}/data/Stations_soil_data/{site}/{site}_soil_temperature.csv',
                *info['dates']
            )
            
            site_data = (sim_data, obs_data)
            
            # 在两个图中分别绘制
            for fig_idx, fig in enumerate(figs):
                plt.figure(fig.number)
                ax = plt.subplot(3, 3, i+1)
                
                if fig_idx == 0:  # 原始TS图
                    subplot_label = chr(97 + i)
                    bars, labels = plot_bar_charts(site_data, site, info['depths'], ax, subplot_label)
                else:  # 相对变化图
                    subplot_label = chr(97 + i) 
                    bars, labels = plot_relative_change(site_data, site, info['depths'], ax, subplot_label)
                
                # 在第8个子图位置添加图例
                if i == 6:  # 第7个站点绘制完成后
                    legend_ax = plt.subplot(3, 3, 8)
                    legend_ax.axis('off')
                    legend_ax.legend(bars, labels, fontsize=12, frameon=False, 
                                loc='center', ncol=len(labels))
            
            # 输出TS统计量
            print(f"\n{site}站点Taylor技巧评分:")
            print("深度\t方案\tTS得分")
            print("-" * 30)
            for depth_idx, depth in enumerate(info['depths']):
                for scheme_idx, scheme in enumerate(['CTL', 'SEP1', 'SEP2']):
                    ts = calculate_ts_score(
                        sim_data[scheme_idx][depth_idx],
                        obs_data.iloc[:, depth_idx+1]
                    )
                    print(f"{depth}cm\t{scheme}\t{ts:.4f}")

    # 保存两个图形
    for fig_idx, fig in enumerate(figs):
        plt.figure(fig.number)
        plt.tight_layout()
        suffix = ['original', 'relative_change'][fig_idx]
        fig.savefig(f'{base_path}/fig/after250104/fig6_Taylor_Skill_Score_All_Sites_Temperature_{suffix}.png',
                    bbox_inches='tight', dpi=600)
    
    plt.show()
