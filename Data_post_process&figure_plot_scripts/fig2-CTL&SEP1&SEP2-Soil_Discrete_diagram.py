import numpy as np
import matplotlib.pyplot as plt

# 设置土壤层数
nlevsoi = 20
nlevsoi_10sl = 10
nlevsoi_40sl = 40

# 初始化数组
dzsoi = np.zeros(nlevsoi)
zisoi = np.zeros(nlevsoi + 1)
zsoi = np.zeros(nlevsoi)
dzsoi_10sl = np.zeros(nlevsoi_10sl)
dzsoi_40sl = np.zeros(nlevsoi_40sl)
zisoi_10sl = np.zeros(nlevsoi_10sl + 1)
zsoi_10sl = np.zeros(nlevsoi_10sl)
zisoi_40sl = np.zeros(nlevsoi_40sl + 1)
zsoi_40sl = np.zeros(nlevsoi_40sl)

# 计算20层方案的深度
for j in range(1, 5):
    dzsoi[j - 1] = j * 0.02
for j in range(5, 14):
    dzsoi[j - 1] = dzsoi[3] + (j - 4) * 0.04
for j in range(14, nlevsoi + 1):
    dzsoi[j - 1] = dzsoi[12] + (j - 13) * 0.10

zisoi[0] = 0.0
for j in range(1, nlevsoi + 1):
    zisoi[j] = np.sum(dzsoi[:j])
for j in range(nlevsoi):
    zsoi[j] = (zisoi[j] + zisoi[j+1]) * 0.5

# 计算10层方案的深度
for j in range(1, nlevsoi_10sl + 1):
    dzsoi_10sl[j - 1] = dzsoi[2*j - 2] + dzsoi[2*j - 1]

zisoi_10sl[0] = 0.0
for j in range(1, nlevsoi_10sl + 1):
    zisoi_10sl[j] = np.sum(dzsoi_10sl[:j])
for j in range(nlevsoi_10sl):
    zsoi_10sl[j] = (zisoi_10sl[j] + zisoi_10sl[j+1]) * 0.5

# 计算40层方案的深度
for j in range(1, nlevsoi_40sl + 1):
    dzsoi_40sl[j - 1] = dzsoi[(j + 1) // 2 - 1] / 2

zisoi_40sl[0] = 0.0
for j in range(1, nlevsoi_40sl + 1):
    zisoi_40sl[j] = np.sum(dzsoi_40sl[:j])
for j in range(nlevsoi_40sl):
    zsoi_40sl[j] = (zisoi_40sl[j] + zisoi_40sl[j+1]) * 0.5

# 绘图设置
fig, ax = plt.subplots(figsize=(6, 10))
bar_width = 0.1
indices = np.array([0, bar_width, 2*bar_width])

# 绘制主图
schemes = [(zisoi_10sl, '#CC79A8', 'SEP1-10 soil layers'),
            (zisoi, '#009F73', 'CTL-20 soil layers'),
            (zisoi_40sl, '#9F4923', 'SEP2-40 soil layers')]

for i, (data, color, label) in enumerate(schemes):
    ax.bar(indices[i], max(data), bar_width, label=label, color=color, alpha=0.7)
    for val in data:
        ax.hlines(val, indices[i]-bar_width/2, indices[i]+bar_width/2, colors='k', linestyles='--')

# 主图样式设置
ax.set_xlim(-0.05, indices[-1] + 0.05)
ax.set_ylim(0, 8.6)
ax.invert_yaxis()
ax.set_xticks(indices)
ax.set_xticklabels([s[2] for s in schemes], rotation=45, ha="left", rotation_mode="anchor", fontweight='bold')
ax.xaxis.tick_top()
ax.set_ylabel('Depth (m)', fontweight='bold')
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.tick_params(axis='y', labelsize=12)  # 设置y轴tickslabel字体大小
ax.spines[:].set_linewidth(1.5)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# 添加放大图
inset_ax = fig.add_axes([1.04, 0.2, 0.2, 0.68])
filter_limit = 0.92

for i, (data, color, _) in enumerate(schemes):
    filtered_data = data[data <= filter_limit]
    inset_ax.bar(indices[i], max(filtered_data), bar_width, color=color, alpha=0.7)
    for val in filtered_data:
        inset_ax.hlines(val, indices[i]-bar_width/2, indices[i]+bar_width/2, colors='k', linestyles='--')

# 放大图样式设置
inset_ax.set_xlim(-0.05, indices[-1] + 0.05)
inset_ax.set_ylim(0, 0.92)
inset_ax.invert_yaxis()
inset_ax.set_xticklabels([])
inset_ax.xaxis.set_ticks_position('none')
inset_ax.set_ylabel('Depth (m)', fontweight='bold')
inset_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
inset_ax.tick_params(axis='y', labelsize=12)  # 设置y轴tickslabel字体大小
inset_ax.spines[:].set_linewidth(1.5)
inset_ax.spines['right'].set_visible(False)
inset_ax.spines['bottom'].set_visible(False)

# 添加注释
inset_ax.annotate('', xy=(-0.75, 0.9), xytext=(0., 0.9), xycoords='axes fraction',
                arrowprops=dict(facecolor='black', arrowstyle="->", lw=2))
inset_ax.annotate('figure at depth less than 1m', xy=(1, 1), xytext=(5, -5),
                xycoords='axes fraction', textcoords='offset points',
                ha='left', va='top', fontsize=10, fontweight='bold', rotation=270)

plt.savefig('/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/fig/202506/fig2_CTL&SEP1&SEP2-Soil_Discrete_diagram.png'
            ,bbox_inches='tight', dpi=1200)
plt.show()
