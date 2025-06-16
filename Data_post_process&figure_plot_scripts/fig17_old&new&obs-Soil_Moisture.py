import xarray as xr
import numpy as np
import pandas as pd
import shapefile
import cmaps

import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import t

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cftime import DatetimeNoLeap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mpl_colors
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

# 设置全局字体为Helvetica
plt.rcParams['font.family'] = 'Helvetica'

def load_and_process_humidity_data(nc_path, time_slice, layer, is_modified=False):
        # 打开NetCDF文件
        nc_data = xr.open_dataset(nc_path)
        
        # 根据是否为修改后的方案选择不同的层数
        if is_modified:
                humid = nc_data['SOILLIQ'].sel(time=time_slice).isel(lev=layer)
                dzsoi = nc_data['ORIDZ'].isel(time=0).isel(lev=layer).drop('time')
        else:
                humid = nc_data['SOILLIQ'].sel(time=time_slice).isel(lev=layer)
                dzsoi = nc_data['ORIDZ'].isel(time=0).isel(lev=layer).drop('time')
        
        humid = humid / dzsoi / 1000.0
        humid_mean = humid.mean(dim='time')
        
        return humid_mean

def shp2clip(originfig, ax, shpfile, fieldVals):
        sf = shapefile.Reader(shpfile)
        vertices = []
        codes = []
        pts = sf.shapes()[0].points
        prt = list(sf.shapes()[0].parts) + [len(pts)]
        for i in range(len(prt) - 1):
                for j in range(prt[i], prt[i+1]):
                        vertices.append((pts[j][0], pts[j][1]))
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
                codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)

        for contour in originfig.collections:
                contour.set_clip_path(clip)
        for line in ax.lines:
                line.set_clip_path(clip)
        return clip

def plot_map_set(ax, lon, lat, data, levels, title, shapefile_path, cb_ticks, cmap, clip_shape=False, clip_args=None):
        ax.set_extent([75.2, 105.5, 25.0, 42.3], crs=ccrs.PlateCarree())
        ax.set_xticks(list(range(74, 107, 8)), crs=ccrs.PlateCarree())
        ax.set_yticks(list(range(24, 41, 4)), crs=ccrs.PlateCarree())
        ax.tick_params(labelcolor='k', length=5, width=3, labelsize=18)

        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.spines['geo'].set_linewidth(3)

        reader = Reader(shapefile_path)
        geoms = reader.geometries()
        ax.add_geometries(geoms, ccrs.PlateCarree(), lw=2, fc='none')

        norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend='both')
        cn = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, levels=levels, extend='both')

        ax.set_title(title, fontsize=20, loc='left', fontname='Helvetica')

        if clip_shape and clip_args is not None:
                clip = shp2clip(cn, ax, shapefile_path, clip_args)

        cax = inset_axes(ax, width="5%", height="100%", loc='right',
                        bbox_to_anchor=(0.1, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(cn, cax=cax, orientation='vertical', ticks=cb_ticks)
        cb.ax.tick_params(labelsize=18, direction='out', width=2.0, length=6.0)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontname='Helvetica')
        cb.outline.set_linewidth(1.5)
        cax.yaxis.set_ticks_position('right')
        cb.ax.text(1.0, 1.15, 'm³/m³', fontsize=18, ha='center', va='top', transform=cb.ax.transAxes, fontname='Helvetica')

ft_slice_time = slice("2017-10-01", "2018-05-01")

# 定义新旧方案的层数
layers = {
        '10cm': {'old': 2, 'new': 4},
        '40cm': {'old': 5, 'new': 10},
        '100cm': {'old': 8, 'new': 16}
}

original_path = '/Volumes/Finn‘sT7/Data/博士资料/Evaluation of the Impact of Soil Discretization Schemes on Soil Moisture and Heat Transport on the Tibetan Plateau Using CLM5.0/model_output/150x300_1206_TP_CTL.clm2_SOILLIQ_ORIDZ.h0.2017-09-01-00000.nc'
modified_path = '/Volumes/Finn‘sT7/Data/博士资料/Evaluation of the Impact of Soil Discretization Schemes on Soil Moisture and Heat Transport on the Tibetan Plateau Using CLM5.0/model_output/150x300_TP_SVD_scheme.clm2_SOILLIQ_ORIDZ.h0.2017-09-01-00000.nc'

# 处理所有深度的数据
results = {}
for depth in ['10cm', '40cm', '100cm']:
        # 获取原始方案数据
        humid_original = load_and_process_humidity_data(
                original_path, ft_slice_time, layers[depth]['old'], is_modified=False)
        
        # 获取修改方案数据
        humid_modified = load_and_process_humidity_data(
                modified_path, ft_slice_time, layers[depth]['new'], is_modified=True)
        
        # 读取对应深度的观测数据
        obs_path = f'/Volumes/Finn‘sT7/Data/博士资料/Evaluation of the Impact of Soil Discretization Schemes on Soil Moisture and Heat Transport on the Tibetan Plateau Using CLM5.0/OBS_regional_data/SMCI_9km/SMCI_9km_2017_2018_{depth}_interp.nc'
        obs_data = xr.open_dataset(obs_path)
        obs_smci = obs_data['SMCI'].sel(time=ft_slice_time)
        obs_mean = obs_smci.mean(dim='time')
        obs_mean = obs_mean.fillna(0) * 0.001
        
        # 确保观测数据与模型数据具有相同的坐标
        obs_interp = obs_mean.interp(lat=humid_original.lat, lon=humid_original.lon, method='linear')
        
        # 计算差值
        results[depth] = {
                'original_diff': humid_original - obs_interp,
                'modified_diff': humid_modified - obs_interp
        }

lon = humid_original.lon.values
lat = humid_original.lat.values

shapefile_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/老师学生资料相关/盛丹睿/气候区/tibet_shp/qingz(bnd)(geo).shp'

fig = plt.figure(figsize=(16, 12))
# 调整子图之间的间距
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1],
                hspace=0.3,  # 减小行间距
                wspace=0.2)  # 增加列间距
crs = ccrs.PlateCarree()

depths = ['10cm', '40cm', '100cm']
titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
plot_args = []

for i, depth in enumerate(depths):
        # 原始方案差值
        plot_args.append((gs[i, 0], results[depth]['original_diff'], 
                        np.arange(-0.2, 0.2, 0.02), titles[i*2], 
                        np.arange(-0.2, 0.2, 0.08)))
        # 修改方案差值
        plot_args.append((gs[i, 1], results[depth]['modified_diff'], 
                        np.arange(-0.2, 0.2, 0.02), titles[i*2+1], 
                        np.arange(-0.2, 0.2, 0.08)))

for arg in plot_args:
        ax = fig.add_subplot(arg[0], projection=crs)
        plot_map_set(ax, lon, lat, arg[1], arg[2], arg[3], shapefile_path, 
                        arg[4], cmap=cmaps.cmocean_balance, clip_shape=True, clip_args=[0])

plt.savefig(
        fname='/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/fig/202506/fig17-new-old%new&obs_TP_Moisture.jpg',
        bbox_inches='tight', dpi=300
)
plt.show()
