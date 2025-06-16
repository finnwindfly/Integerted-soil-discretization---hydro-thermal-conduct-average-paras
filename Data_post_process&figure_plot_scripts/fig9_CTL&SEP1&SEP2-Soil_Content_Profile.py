import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from cftime import DatetimeNoLeap
import matplotlib.colors as mcolors

def read_nc_data(nc_file, start_date_str, end_date_str):
    """读取NC文件数据"""
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str, dayfirst=True).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str, dayfirst=True).timetuple()[:3])
    )
    
    data = xr.open_dataset(nc_file)
    return (
        data.ORGANIC_COL.isel(lndgrid=0).rename({'levsoi': 'levgrnd'}).sel(time=time_slice).mean(dim='time'),
        data.SAND_COL.isel(lndgrid=0).rename({'levsoi': 'levgrnd'}).sel(time=time_slice).mean(dim='time'),
        data.WATSAT.isel(lndgrid=0).isel(levgrnd=slice(None, -5)),
        data.BD_COL.isel(lndgrid=0).sel(time=time_slice).isel(levgrnd=slice(None, -5)).mean(dim='time'),
        data.ZSOI.isel(lndgrid=0).isel(levgrnd=slice(None, -5))
    )

def plot_profile(ax, data, depth, style, ylim, col_index, site_index):
    """绘制单个剖面图"""
    ax.plot(data, depth, **style)
    ax.set_ylim(0, ylim)
    ax.invert_yaxis()
    
    # 设置坐标轴
    x_min, x_max = np.min(data), np.max(data)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # 样式设置
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.tick_params(axis='both', labelsize=18, width=2)
    ax.xaxis.tick_top()
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['top'].set_linewidth(2.5)
    
    # 设置标签
    if col_index == 0:
        ax.set_ylabel('Soil Depth (m)', fontsize=20)
    if site_index == 6:
        ax.set_xlabel(['Organic (%)', 'Sand (%)', 'Porosity (m³/m³)', 'Bulk Density (kg/m³)'][col_index], fontsize=20, labelpad=20)
    
    # 只在第一行添加图例，位置在右下角
    if site_index == 0:
        ax.legend(loc='lower center', fontsize=16, frameon=False)
    
    # 修改子图标签格式
    row_letter = chr(ord('a') + site_index)
    label = f'({row_letter}{col_index+1})'
    ax.text(0.15, 0.1, label, transform=ax.transAxes, fontsize=16)

# 基本配置
model_output_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/data/model_output'
base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0'

sites = {
    'NAMORS': (3.2, 2015), 'MADUO': (4.0, 2017), 'MAQU': (2.0, 2022),
    'Ngari': (3.0, 2012), 'Qoms': (1.6, 2015), 'SETORS': (1.5, 2009),
    'YAKOU': (3.0, 2018)
}

colors = {'organic': '#4682B4', 'sand': '#CD5C5C', 'porosity': '#87CEFA', 'bulk': '#F4A460'}
markers = ['o', 's', '^']
alphas = [1.0, 0.8, 0.6]

# 创建图形
fig, ax = plt.subplots(len(sites), 4, figsize=(24, 4 * len(sites)))

# 绘图循环
for i, (site, (ylim, year)) in enumerate(sites.items()):  # 遍历每个站点
    data_lists = [[], [], [], [], []]  # 存储每个站点的5种数据类型
    
    # 读取CTL, SEP1, SEP2三种方案的数据
    for suffix in ['CTL', 'SEP1', 'SEP2']:
        nc_file = f"{model_output_path}/{site}/final_use/run/{site}_model_output_{suffix}.nc"
        data = read_nc_data(nc_file, f'{year}-09-01', f'{year+1}-09-01')
        for j in range(5):
            data_lists[j].append(data[j])
    
    # 绘制四种土壤特性的剖面图
    for j, (data_type, base_color) in enumerate(colors.items()):
        # 为每种方案(CTL/SEP1/SEP2)绘制曲线
        for k in range(3):
            style = {
                'marker': markers[k],
                'color': tuple(min(1.0, c * (1.3 if k == 1 else 0.7 if k == 2 else 1.0)) for c in mcolors.to_rgb(base_color)),
                'label': f"{data_type}_{['CTL', 'SEP1', 'SEP2'][k]}",
                'linewidth': 3.5,
                'markersize': 8,
                'alpha': alphas[k]
            }
            plot_profile(ax[i, j], data_lists[j][k], data_lists[4][k], style, ylim, j, i)

plt.tight_layout(h_pad=0.2, w_pad=0.1)  # 进一步减小行距和列距
plt.savefig(f'{base_path}/fig/202506/fig9_CTL&SEP1&SEP2-Soil_Content_Profile.jpg',
            bbox_inches='tight', dpi=600)
plt.show()
