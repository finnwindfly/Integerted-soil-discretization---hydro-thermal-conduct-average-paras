import matplotlib.pyplot as plt
import datetime
import pandas as pd

# 完整的数据字典
data_dict = {
    'NAMORS': {
        'OBS': {
            'First': ('10.04', '04.04'),
            'Second': ('11.12', '03.17'),
            'Third': ('12.12', '03.14'),
            'Fourth': ('01.02', '04.04')
        },
        'CTL': {
            'First': ('10.27', '04.02'),
            'Second': ('11.01', '04.06'),
            'Third': ('11.09', '04.22'),
            'Fourth': ('11.28', '04.30')
        },
        'SEP1': {
            'First': ('10.14', '04.05'),
            'Second': ('10.28', '04.06'),
            'Third': ('11.09', '04.30'),
            'Fourth': ('11.24', '04.30')
        },
        'SEP2': {
            'First': ('10.26', '04.02'),
            'Second': ('11.07', '04.06'),
            'Third': ('11.15', '04.22'),
            'Fourth': ('12.03', '04.30')
        }
    },
    'MADUO': {
        'OBS': {
            'First': ('10.26', '04.18'),
            'Second': ('11.05', '04.22'),
            'Third': ('11.20', '05.01'),
            'Fourth': ('12.20', '05.15')
        },
        'CTL': {
            'First': ('10.27', '04.16'),
            'Second': ('11.04', '05.01'),
            'Third': ('11.16', '05.01'),
            'Fourth': ('12.16', '05.03')
        },
        'SEP1': {
            'First': ('10.26', '04.16'),
            'Second': ('11.04', '05.01'),
            'Third': ('11.14', '05.01'),
            'Fourth': ('12.11', '05.05')
        },
        'SEP2': {
            'First': ('10.27', '04.16'),
            'Second': ('11.05', '04.24'),
            'Third': ('11.19', '05.01'),
            'Fourth': ('12.17', '05.01')
        }
    },
    'MAQU': {
        'OBS': {
            'First': ('11.30', '03.17'),
            'Second': ('12.23', '04.04'),
            'Third': ('01.22', '03.18'),
            'Fourth': ('-', '-')
        },
        'CTL': {
            'First': ('12.03', '04.07'),
            'Second': ('12.23', '04.18'),
            'Third': ('01.26', '04.02'),
            'Fourth': ('-', '-')
        },
        'SEP1': {
            'First': ('11.22', '04.06'),
            'Second': ('12.20', '04.10'),
            'Third': ('01.04', '04.20'),
            'Fourth': ('-', '-')
        },
        'SEP2': {
            'First': ('12.01', '04.04'),
            'Second': ('12.30', '04.13'),
            'Third': ('03.06', '03.18'),
            'Fourth': ('-', '-')
        }
    },
    'NADORS': {
        'OBS': {
            'First': ('10.24', '03.16'),
            'Second': ('11.14', '04.13'),
            'Third': ('12.16', '04.30'),
            'Fourth': ('02.28', '05.01')
        },
        'CTL': {
            'First': ('10.25', '03.18'),
            'Second': ('11.10', '03.13'),
            'Third': ('12.17', '03.26'),
            'Fourth': ('12.22', '04.03')
        },
        'SEP1': {
            'First': ('10.23', '04.04'),
            'Second': ('10.26', '03.22'),
            'Third': ('11.15', '04.08'),
            'Fourth': ('12.05', '04.22')
        },
        'SEP2': {
            'First': ('10.25', '03.08'),
            'Second': ('11.12', '03.13'),
            'Third': ('11.28', '03.26'),
            'Fourth': ('12.27', '04.03')
        }
    },
    'QOMS': {
        'OBS': {
            'First': ('12.16', '02.10'),
            'Second': ('12.18', '02.09'),
            'Third': ('12.21', '02.02'),
            'Fourth': ('01.21', '02.18')
        },
        'CTL': {
            'First': ('11.19', '02.16'),
            'Second': ('12.16', '02.19'),
            'Third': ('12.22', '03.14'),
            'Fourth': ('01.15', '03.11')
        },
        'SEP1': {
            'First': ('11.09', '03.06'),
            'Second': ('11.20', '03.08'),
            'Third': ('12.21', '03.26'),
            'Fourth': ('01.15', '02.25')
        },
        'SEP2': {
            'First': ('11.20', '02.16'),
            'Second': ('12.17', '02.19'),
            'Third': ('12.24', '03.08'),
            'Fourth': ('01.21', '02.18')
        }
    },
    'SETORS': {
        'OBS': {
            'First': ('-', '-'),
            'Second': ('-', '-'),
            'Third': ('-', '-'),
            'Fourth': ('-', '-')
        },
        'CTL': {
            'First': ('12.29', '01.24'),
            'Second': ('-', '-'),
            'Third': ('-', '-'),
            'Fourth': ('-', '-')
        },
        'SEP1': {
            'First': ('12.28', '01.22'),
            'Second': ('01.05', '01.10'),
            'Third': ('-', '-'),
            'Fourth': ('-', '-')
        },
        'SEP2': {
            'First': ('12.29', '01.13'),
            'Second': ('-', '-'),
            'Third': ('-', '-'),
            'Fourth': ('-', '-')
        }
    },
    'YAKOU': {
        'OBS': {
            'First': ('10.07', '05.01'),
            'Second': ('10.09', '05.01'),
            'Third': ('10.20', '04.30'),
            'Fourth': ('10.21', '05.01')
        },
        'CTL': {
            'First': ('10.27', '04.08'),
            'Second': ('10.31', '04.15'),
            'Third': ('11.08', '04.22'),
            'Fourth': ('11.24', '04.29')
        },
        'SEP1': {
            'First': ('10.09', '04.06'),
            'Second': ('10.30', '04.12'),
            'Third': ('11.11', '04.25'),
            'Fourth': ('11.25', '05.01')
        },
        'SEP2': {
            'First': ('10.12', '04.06'),
            'Second': ('10.30', '04.16'),
            'Third': ('11.10', '04.22'),
            'Fourth': ('11.25', '05.01')
        }
    }
}

def convert_to_datetime(date_str):
    if date_str == '-':
        return None
    month, day = map(int, date_str.split('.'))
    year = 2000 if month >= 6 else 2001
    return datetime.datetime(year, month, day)

def plot_freezing_periods(ax, station_data, station, subplot_label, col):
    colors = ['#423c40', '#CD5C5C', '#87CEFA', '#F4A460']  # 降低饱和度的颜色
    labels = ['OBS', 'CTL', 'SEP1', 'SEP2']
    experiments = ['OBS', 'CTL', 'SEP1', 'SEP2']
    layers = ['First', 'Second', 'Third', 'Fourth']
    
    # 设置y轴刻度和标签
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    
    for layer_idx, layer in enumerate(layers):
        for exp_idx, exp in enumerate(experiments):
            y = layer_idx + exp_idx * 0.15  # 减小间距
            
            start_str, end_str = station_data[exp][layer]
            if start_str != '-' and end_str != '-':
                start = convert_to_datetime(start_str)
                end = convert_to_datetime(end_str)
                if start and end:
                    ax.hlines(y, start, end, colors[exp_idx], linewidth=4.0,
                            label=labels[exp_idx] if layer_idx == 0 and station == 'MAQU' else "")

    ax.set_ylim(-0.2, len(layers)-0.2)
    ax.set_title(f'({subplot_label}) {station}', loc='left', fontsize=12)
    
    # 只在第一列显示ylabel
    if col == 0:
        ax.set_ylabel('Soil Layers', fontsize=15)
    
    # 只在YAKOU, QOMS和SETORS站点显示xlabel
    if station in ['YAKOU', 'QOMS', 'SETORS']:
        ax.set_xlabel('Month', fontsize=15)
    else:
        ax.set_xlabel('')
    
    # 修改x轴刻度，每个子图显示6个月份
    months = [datetime.datetime(2000, 10, 1),  # Oct
             datetime.datetime(2000, 11, 15),  # Nov
             datetime.datetime(2001, 1, 1),   # Jan
             datetime.datetime(2001, 2, 15),  # Feb
             datetime.datetime(2001, 4, 1),   # Apr
             datetime.datetime(2001, 5, 15)]  # May
    ax.set_xticks(months)
    ax.set_xticklabels(['Oct', 'Nov', 'Jan', 'Feb', 'Apr', 'May'])
    
    # 加粗轴线
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 在MAQU站点显示legend
    if station == 'MAQU':
        ax.legend(frameon=False, loc='upper left')

# 创建图形
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
stations = ['NAMORS', 'MADUO', 'MAQU', 'NADORS', 'QOMS', 'SETORS', 'YAKOU']

for i, station in enumerate(stations):
    row = i // 3
    col = i % 3
    subplot_label = chr(97 + i)  # 生成子图标签 a, b, c...
    plot_freezing_periods(axes[row, col], data_dict[station], station, subplot_label, col)

# 删除多余的子图
axes[2, 1].remove()
axes[2, 2].remove()

plt.tight_layout()
plt.savefig('/Users/finn/Library/CloudStorage/OneDrive-Personal/学习工作相关/博士生工作/博士工作相关/科研工作/文章工作/Sensitivity analysis and parameterization scheme optimization for soil vertical heterogeneity on the Tibetan Plateau based on the Community Land Model 5.0/fig/202506/fig5_Soil_Frozen_Period_CTL&SEP1&SEP2.jpg', 
            bbox_inches='tight', dpi=300)
plt.show()
