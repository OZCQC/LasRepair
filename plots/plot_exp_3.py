#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制exp_3.txt数据的折线图
每个实体（Beers, Hospital, Flight, Walmart）在不同迭代次数下的表现
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']  # 使用DejaVu Sans作为默认字体，支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def read_data(file_path):
    """
    读取数据文件并解析
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        iterations: 迭代次数列表（x轴数据）
        entities_data: 字典，键为实体名称，值为对应的数值列表
    """
    entities_data = {}
    iterations = []
    
    # 尝试不同的编码方式
    encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    lines = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            if lines:
                break
        except Exception as e:
            print(f"尝试编码 {encoding} 失败: {e}")
            continue
    
    if not lines:
        raise ValueError(f"无法读取文件: {file_path}")
    
    # 解析第一行，获取迭代次数
    first_line = lines[0].strip().split('\t')
    # 第一列是标题"EXP3: different iteration"，从第二列开始是迭代次数
    iterations = [int(x) for x in first_line[1:] if x.strip()]
    
    # 解析后续每一行，每一行代表一个实体
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) < 2:  # 跳过空行
            continue
        
        entity_name = parts[0]  # 第一列是实体名称
        # 从第二列开始是数值，过滤掉空值
        values = []
        for val in parts[1:]:
            val = val.strip()
            if val:  # 如果值不为空
                try:
                    values.append(float(val))
                except ValueError:
                    # 如果无法转换为浮点数，跳过
                    pass
        
        if values:  # 只有当有有效值时才添加
            entities_data[entity_name] = values
    
    return iterations, entities_data

def plot_line_chart(iterations, entities_data, output_path):
    """
    绘制折线图并保存为PDF
    
    Args:
        iterations: 迭代次数列表（x轴数据）
        entities_data: 字典，键为实体名称，值为对应的数值列表
        output_path: 输出PDF文件路径
    """
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 定义颜色和标记样式，使不同实体易于区分
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 为每个实体绘制折线
    for idx, (entity_name, values) in enumerate(entities_data.items()):
        # 获取该实体对应的迭代次数（可能少于总迭代次数）
        entity_iterations = iterations[:len(values)]
        
        # 绘制折线图
        ax.plot(entity_iterations, values, 
                marker=markers[idx % len(markers)],  # 循环使用标记样式
                color=colors[idx % len(colors)],     # 循环使用颜色
                linewidth=2,                          # 线宽
                markersize=8,                         # 标记大小
                label=entity_name,                    # 图例标签
                alpha=0.8)                            # 透明度
    
    # 设置坐标轴标签
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    
    # 设置标题
    ax.set_title('EXP3: Performance Across Different Iterations', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 添加网格线，使图表更易读
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 设置x轴刻度为整数
    ax.set_xticks(iterations)
    
    # 调整布局，确保所有元素都能显示
    plt.tight_layout()
    
    # 保存为PDF格式
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    
    # 显示图表（可选，如果在交互式环境中）
    # plt.show()

if __name__ == '__main__':
    # 获取脚本所在目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据文件路径（使用绝对路径）
    data_file = os.path.join(script_dir, 'exp_3.txt')
    
    # 输出PDF文件路径（使用绝对路径）
    output_pdf = os.path.join(script_dir, 'exp_3_plot.pdf')
    
    # 读取数据
    print("正在读取数据...")
    print(f"数据文件路径: {data_file}")
    iterations, entities_data = read_data(data_file)
    
    print(f"迭代次数: {iterations}")
    print(f"实体数量: {len(entities_data)}")
    for entity, values in entities_data.items():
        print(f"  {entity}: {len(values)} 个数据点")
    
    # 绘制折线图
    print("\n正在绘制折线图...")
    plot_line_chart(iterations, entities_data, output_pdf)
    
    print("\n完成！")

