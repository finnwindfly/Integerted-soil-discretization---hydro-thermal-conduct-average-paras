import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from cftime import DatetimeNoLeap
import matplotlib.colors as mcolors
from matplotlib import font_manager

# 设置全局字体为 Helvetica
plt.rcParams['font.family'] = 'Helvetica'

def read_nc_data(nc_file, start_date_str, end_date_str):
    """读取NC文件数据"""
    time_slice = slice(
        DatetimeNoLeap(*pd.to_datetime(start_date_str, dayfirst=True).timetuple()[:3]),
        DatetimeNoLeap(*pd.to_datetime(end_date_str, dayfirst=True).timetuple()[:3])
    )
    
    data = xr.open_dataset(nc_file)
    return (
        # 第一行数据
        data.ORGANIC_COL.isel(lndgrid=0).rename({'levsoi': 'levgrnd'}).sel(time=time_slice).mean(dim='time'),
        data.SAND_COL.isel(lndgrid=0).rename({'levsoi': 'levgrnd'}).sel(time=time_slice).mean(dim='time'),
        data.WATSAT.isel(lndgrid=0).isel(levgrnd=slice(None, -5)),
        data.BD_COL.isel(lndgrid=0).sel(time=time_slice).isel(levgrnd=slice(None, -5)).mean(dim='time'),
        # 第二行数据
        data.HKSAT_COL.isel(lndgrid=0).isel(levgrnd=slice(None, -5)).sel(time=time_slice).mean(dim='time'),
        data.TKSAT_COL.isel(lndgrid=0).isel(levgrnd=slice(None, -5)).sel(time=time_slice).mean(dim='time'),
        data.CSOL_COL.isel(lndgrid=0).isel(levgrnd=slice(None, -5)).sel(time=time_slice).mean(dim='time'),
        data.SUCSAT_COL.isel(lndgrid=0).isel(levgrnd=slice(None, -5)).sel(time=time_slice).mean(dim='time'),
        data.ZSOI.isel(lndgrid=0).isel(levgrnd=slice(None, -5))
    )

def plot_profile(ax, data, depth, style, ylim, col_index, row_index):
    """绘制单个剖面图"""
    ax.plot(data, depth, **style)
    ax.set_ylim(0, ylim)
    ax.invert_yaxis()
    
    # 设置坐标轴
    x_min, x_max = np.min(data), np.max(data)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    
    # 根据列索引设置不同的格式化方式
    if (row_index == 1 and col_index >= 2) or (row_index == 0 and col_index == 3):  # SUCSAT和BD_COL使用科学计数法
        def sci_format(x, p):
            coef = x/10**np.floor(np.log10(abs(x)))
            exp = int(np.floor(np.log10(abs(x))))
            return f'{coef:.2f}'
        ax.xaxis.set_major_formatter(plt.FuncFormatter(sci_format))
        exp = int(np.floor(np.log10(abs(x_max))))
        ax.text(1.02, 1., f'×10$^{exp}$', transform=ax.transAxes, fontsize=16)
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # 样式设置
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.tick_params(axis='both', labelsize=18)
    ax.xaxis.tick_top()
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['left'].set_linewidth(2.5)  # 加粗边框
    ax.spines['top'].set_linewidth(2.5)   # 加粗边框
    
    # 设置标签
    if col_index == 0:
        ax.set_ylabel('Soil Depth (m)', fontsize=20)
    
    labels_row1 = ['Organic (%)', 'Sand (%)', 'Porosity (m³/m³)', 'Bulk Density (kg/m³)']
    labels_row2 = ['TKSAT (W/m/K)', 'HKSAT (mm/s)', 'SUCSAT (mm)', 'CSOL (J/m³/K)']
    labels = labels_row1 if row_index == 0 else labels_row2
    ax.set_xlabel(labels[col_index], fontsize=20, labelpad=20)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=14, frameon=False)  # 加大图例字体
    
    # 修改子图标签格式
    label = f'({chr(ord("a") + col_index + row_index*4)})'
    ax.text(0.15, 0.1, label, transform=ax.transAxes, fontsize=16)

# 基本配置
model_output_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/data/model_output/40SL_discret_scheme/MAQU'
base_path =  '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/fig/202506'

site = 'MAQU'
ylim = 2.0
year = 2022

colors_row1 = {'organic': '#3b658c', 'porosity': '#c82423', 'sand': '#28a92b', 'bulk': '#ff7f0e'}
colors_row2 = {'tksat': '#3b658c', 'hksat': '#c82423', 'sucsat': '#28a92b', 'csol': '#ff7f0e'}
markers = ['o', 's']
alphas = [1.0, 0.8]

# 创建图形
fig, ax = plt.subplots(2, 4, figsize=(16, 12))

# 存储数据
data_lists = [[] for _ in range(9)]  # 存储9种数据类型

# 读取old和new两种方案的数据
for suffix in ['old', 'new']:
    nc_file = f"{model_output_path}/MAQU_model_output_{suffix}.nc"
    data = read_nc_data(nc_file, f'{year}-10-01', f'{year+1}-05-01')
    for j in range(9):
        data_lists[j].append(data[j])

# 绘制第一行四种土壤特性的剖面图
for j, (data_type, base_color) in enumerate(colors_row1.items()):
    for k in range(2):
        style = {
            'marker': markers[k],
            'color': tuple(min(1.0, c * (1.3 if k == 1 else 1.0)) for c in mcolors.to_rgb(base_color)),
            'label': f"{data_type}_{['old', 'new'][k]}",
            'linewidth': 3.0,  # 加粗线条
            'markersize': 8,
            'alpha': alphas[k]
        }
        plot_profile(ax[0,j], data_lists[j][k], data_lists[8][k], style, ylim, j, 0)

# 绘制第二行四种土壤特性的剖面图
for j, (data_type, base_color) in enumerate(colors_row2.items()):
    for k in range(2):
        style = {
            'marker': markers[k],
            'color': tuple(min(1.0, c * (1.3 if k == 1 else 1.0)) for c in mcolors.to_rgb(base_color)),
            'label': f"{data_type}_{['old', 'new'][k]}",
            'linewidth': 3.0,  # 加粗线条
            'markersize': 8,
            'alpha': alphas[k]
        }
        plot_profile(ax[1,j], data_lists[j+4][k], data_lists[8][k], style, ylim, j, 1)

plt.tight_layout(h_pad=2.0)
plt.savefig(f'{base_path}/fig15_old&new-Soil_Content_Profile.jpg',
            bbox_inches='tight', dpi=300)
plt.show()
