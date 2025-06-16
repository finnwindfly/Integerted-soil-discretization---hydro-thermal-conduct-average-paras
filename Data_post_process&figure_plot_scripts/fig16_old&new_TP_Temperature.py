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

def load_and_process_temp_data(nc_path, time_slice, is_modified=False):
        nc_data = xr.open_dataset(nc_path)
        temp_shallow = nc_data['TSOI'].sel(time=time_slice).sel(levgrnd=slice(None, 0.4)) - 273.15
        temp_shallow_mean = temp_shallow.mean(dim='time').mean(dim='levgrnd')

        # 处理深层湿度数据，根据是否为修改后的方案使用不同的deep_slice
        # deep_slice_actual = slice(18, 37) if is_modified else deep_slice
        temp_deep = nc_data['TSOI'].sel(time=time_slice).sel(levgrnd=slice(1.0, 2.0)) - 273.15
        temp_deep_mean = temp_deep.mean(dim='time').mean(dim='levgrnd')

        # 处理最小湿度数据，根据是否为修改后的方案使用不同的slice范围
        # min_slice = slice(0, 37) if is_modified else slice(0, 12)
        temp_min = nc_data['TSOI'].sel(time=time_slice).sel(levgrnd=slice(None, 2.0)) - 273.15
        temp_min_val = temp_min.min(dim='time').mean(dim='levgrnd')
        return temp_shallow_mean, temp_deep_mean, temp_min_val

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
        cb.ax.text(1.0, 1.15, '°C', fontsize=18, ha='center', va='top', transform=cb.ax.transAxes, fontname='Helvetica')

ft_slice_time = slice("2017-10-01", "2018-05-01")
# shallow_slice = 5
# deep_slice = 8

original_path = '/Volumes/Finn‘sT7/Data/博士资料/Evaluation of the Impact of Soil Discretization Schemes on Soil Moisture and Heat Transport on the Tibetan Plateau Using CLM5.0/model_output/150x300_1206_TP_CTL.clm2_TSOI.h0.2017-09-01-00000.nc'
modified_path = '/Volumes/Finn‘sT7/Data/博士资料/Evaluation of the Impact of Soil Discretization Schemes on Soil Moisture and Heat Transport on the Tibetan Plateau Using CLM5.0/model_output/150x300_TP_SVD_scheme.clm2_TSOI.h0.2017-09-01-00000.nc'

temp_original_shallow, temp_original_deep, temp_original_min = load_and_process_temp_data(original_path, ft_slice_time, is_modified=False)
temp_modified_shallow, temp_modified_deep, temp_modified_min = load_and_process_temp_data(modified_path, ft_slice_time, is_modified=True)

temp_df_shallow = temp_modified_shallow - temp_original_shallow
temp_df_deep = temp_modified_deep - temp_original_deep
temp_df_min = temp_modified_min - temp_original_min

lon = temp_original_shallow.lon.values
lat = temp_original_shallow.lat.values

shapefile_path = '/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/老师学生资料相关/盛丹睿/气候区/tibet_shp/qingz(bnd)(geo).shp'

fig = plt.figure(figsize=(21, 12))
# 修改GridSpec参数来减小子图间距
gs = GridSpec(3, 3, figure=fig, wspace=0.45, hspace=0.1)  # 减小hspace值来减小行间距
crs = ccrs.PlateCarree()

plot_args = [
        (gs[0, 0], temp_original_shallow, np.arange(-10, 11, 1), "(a)", np.arange(-10, 11, 4)),
        (gs[0, 1], temp_modified_shallow, np.arange(-10, 11, 1), "(b)", np.arange(-10, 11, 4)),
        (gs[0, 2], temp_df_shallow, np.arange(-4, 2.1, 0.2), "(c)", np.arange(-4, 2.1, 0.8)),
        (gs[1, 0], temp_original_deep, np.arange(-10, 11, 1), "(d)", np.arange(-10, 11, 4)),
        (gs[1, 1], temp_modified_deep, np.arange(-10, 11, 1), "(e)", np.arange(-10, 11, 4)),
        (gs[1, 2], temp_df_deep, np.arange(-4, 2.1, 0.2), "(f)", np.arange(-4, 2.1, 0.8)),
        (gs[2, 0], temp_original_min, np.arange(-15, 11, 1), "(g)", np.arange(-15, 11, 3)),
        (gs[2, 1], temp_modified_min, np.arange(-15, 11, 1), "(h)", np.arange(-15, 11, 3)),
        (gs[2, 2], temp_df_min, np.arange(-2, 7, 0.2), "(i)", np.arange(-2, 7, 1.2)),
]

for arg in plot_args:
        ax = fig.add_subplot(arg[0], projection=crs)
        plot_map_set(ax, lon, lat, arg[1], arg[2], arg[3], shapefile_path, arg[4], cmap=cmaps.cmocean_balance, clip_shape=True, clip_args=[0])

plt.savefig(
        fname='/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/fig/202506/fig16-old%new_TP_Temperature.jpg',
        bbox_inches='tight', dpi=300
        )
plt.show()

