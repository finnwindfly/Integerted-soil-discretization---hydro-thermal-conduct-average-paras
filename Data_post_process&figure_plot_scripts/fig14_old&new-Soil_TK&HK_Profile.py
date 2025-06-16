import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import timedelta
from cftime import DatetimeNoLeap
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import string
import cmaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['font.family'] = 'Helvetica'

def read_nc_data(nc_file, start_date_str, end_date_str, var_name):
    """读取并处理模拟数据"""
    start_date = pd.to_datetime(start_date_str,format='mixed', dayfirst=True)
    end_date = pd.to_datetime(end_date_str,format='mixed', dayfirst=True)
    
    start_date_noleap = DatetimeNoLeap(start_date.year, start_date.month, start_date.day)
    end_date_noleap = DatetimeNoLeap(end_date.year, end_date.month, end_date.day)
    
    data = xr.open_dataset(nc_file)
    var_data = data[var_name].isel(lndgrid=0)
    if var_name == 'HK_COL':
        var_data = np.log(var_data)
    depth_levels = data.ZSOI.isel(lndgrid=0)
    
    return var_data.sel(time=slice(start_date_noleap, end_date_noleap)), depth_levels

def plot_profile(ax, time_grid, depth_grid, data, label, ylim, row_idx, is_tk=True):
    """绘制剖面图"""
    if is_tk:
        levels = np.arange(0, 2.0, 0.1)
        cmap = cmaps.cmocean_matter[:190]
        units = 'W/m·K'
    else:
        levels = np.arange(-30, -8, 1)
        cmap = cmaps.MPL_PuBu[20:100][::-1]
        units = 'ln(m/s)'
        
    norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both')
    c = ax.contourf(time_grid, depth_grid, data, cmap=cmap, norm=norm, levels=levels, extend='both')
    
    ax.set_title(f'({label})', loc='left', fontsize=22)
    ax.set_ylim(0, ylim)
    ax.invert_yaxis()
    
    yticks = np.linspace(0, ylim, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontsize=18)
    ax.set_ylabel('Soil depth (m)', fontsize=20)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(axis='both', width=2.0, length=6.0, labelsize=18)

    cax = inset_axes(ax, width="5%", height="100%", loc='right',
                bbox_to_anchor=(0.1, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = plt.colorbar(c, cax=cax, orientation='vertical', ticks=levels[::2])
    cb.ax.tick_params(labelsize=18, direction='out', width=2.0, length=6.0)
    cb.outline.set_linewidth(1.5)
    cb.ax.text(1.5, 1.15, units, fontsize=18, ha='center', va='top', transform=cb.ax.transAxes)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=18)

def main():
    base_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0'
    model_path = f'{base_path}/data/model_output/40SL_discret_scheme/MAQU'
    output_path = f'{base_path}/fig/202506'
    
    dates = ('2022-10-01', '2023-05-01')
    ylim = 1.6

    fig = plt.figure(figsize=(16, 8))
    gs = plt.GridSpec(2, 2, hspace=0.4, wspace=0.4)
    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]

    time_sim = pd.date_range(start=dates[0], end=dates[1])
    
    for scheme in ['old', 'new']:
        data_tk, depths_tk = read_nc_data(f'{model_path}/MAQU_model_output_{scheme}.nc', *dates, 'TKSOIL_COL')
        data_hk, depths_hk = read_nc_data(f'{model_path}/MAQU_model_output_{scheme}.nc', *dates, 'HK_COL')
        
        idx = 0 if scheme == 'old' else 1
        for i, (data, depths, is_tk) in enumerate([(data_tk, depths_tk, True), (data_hk, depths_hk, False)]):
            time_grid, depth_grid = np.meshgrid(time_sim, depths)
            plot_profile(axes[i*2+idx], time_grid, depth_grid, data.values.T, 
                        string.ascii_lowercase[i*2+idx], ylim, i, is_tk)

    plt.savefig(f'{output_path}/fig14_old&new_scheme-Soil_TK&HK_Profile.jpg',
                bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
