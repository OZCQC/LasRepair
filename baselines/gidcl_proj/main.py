"""
GIDCL主程序 - 完整可运行版本

使用说明:
1. 确保安装所有依赖: pip install -r requirements.txt
2. 在.env文件中设置OPENAI_API_KEY
3. 运行: python main.py
"""

import pandas as pd
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入GIDCL
from gidcl import GIDCL


def create_example_data():
    """
    创建示例数据（基于论文Table 1）
    """
    print("创建示例数据集...")
    
    # 脏数据
    dirty_data = {
        'TupleID': ['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
        'ProviderID': [111303, 111303, '1x1303', 10001, 10001, 10001, 10001],
        'City': ['Monticello', 'Monticello', 'Monticello', 'Monticello', 'Dothan', 'Dothan', 'Dothan'],
        'State': ['VA,Jasper', 'VAA', 'AR', 'AL', 'AL,Houston', 'AR', 'AL'],
        'Zip': [31064, 31064, 71655, 36301, 36301, None, 36301],
        'County': ['Jasper', 'Jasper', 'Drew', 'Houston', None, 'Houston', 'Houst']
    }
    
    # 真实数据（Ground Truth）
    ground_truth_data = {
        'TupleID': ['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
        'ProviderID': [111303, 111303, 111303, 10001, 10001, 10001, 10001],
        'City': ['Monticello', 'Monticello', 'Monticello', 'Dothan', 'Dothan', 'Dothan', 'Dothan'],
        'State': ['VA', 'VA', 'AR', 'AL', 'AL', 'AL', 'AL'],
        'Zip': [31064, 31064, 71655, 36301, 36301, 36301, 36301],
        'County': ['Jasper', 'Jasper', 'Drew', 'Houston', 'Houston', 'Houston', 'Houston']
    }
    
    return pd.DataFrame(dirty_data), pd.DataFrame(ground_truth_data)


def main():
    """
    主函数
    """
    print("=" * 80)
    print("GIDCL 演示程序")
    print("=" * 80)
    
    # 检查API密钥
    api_key = "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"
    if not api_key:
        print("\n警告: 未找到OPENAI_API_KEY环境变量")
        print("请在.env文件中设置: OPENAI_API_KEY=your_api_key_here")
        print("\n或者在代码中直接提供API密钥:")
        print("gidcl = GIDCL(llm_api_key='your_api_key_here')")
        return
    
    print(f"\n✓ 找到API密钥: {api_key[:10]}...")
    
    # 创建示例数据
    dirty_table, ground_truth = create_example_data()
    
    print("\n脏数据表:")
    print(dirty_table)
    print(f"\n数据规模: {len(dirty_table)} 行 x {len(dirty_table.columns)} 列")
    
    # 初始化GIDCL
    print("\n初始化GIDCL...")
    gidcl = GIDCL(
        num_clusters=5,          # 聚类数量（小数据集用5）
        labeling_budget=3,       # 标注预算（小数据集用3）
        tau=0.85,                # 函数接受阈值
        lambda_param=4.0,        # 图学习权重
        llm_api_key=api_key,     # API密钥
        llm_model="gpt-4o-mini"  # 使用的模型
    )
    
    try:
        # 执行数据清洗
        print("\n开始数据清洗...")
        cleaned_table = gidcl.fit(dirty_table, ground_truth)
        
        # 输出结果
        print("\n" + "=" * 80)
        print("清洗结果")
        print("=" * 80)
        
        print("\n原始脏数据:")
        print(dirty_table)
        
        print("\n清洗后数据:")
        print(cleaned_table)
        
        print("\n真实数据 (Ground Truth):")
        print(ground_truth)
        
        # 获取可解释模式
        patterns = gidcl.get_interpretable_patterns()
        
        print("\n" + "=" * 80)
        print("可解释的数据清洗模式")
        print("=" * 80)
        
        print("\n检测函数 (Detection Functions):")
        for col, func in patterns['detection_functions'].items():
            print(f"  {col}: {func}")
        
        print("\n修正函数 (Correction Functions):")
        for col, func in patterns['correction_functions'].items():
            print(f"  {col}: {func}")
        
        print("\n函数依赖 (Functional Dependencies):")
        for source, target in patterns['functional_dependencies']:
            print(f"  {source} -> {target}")
        
        # 保存结果
        print("\n保存结果...")
        os.makedirs('results', exist_ok=True)
        
        cleaned_table.to_csv('results/cleaned_data.csv', index=False)
        print("  ✓ 清洗后数据已保存到: results/cleaned_data.csv")
        
        # 保存报告
        with open('results/cleaning_report.txt', 'w', encoding='utf-8') as f:
            f.write("GIDCL 数据清洗报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("数据规模:\n")
            f.write(f"  行数: {len(dirty_table)}\n")
            f.write(f"  列数: {len(dirty_table.columns)}\n\n")
            
            f.write("检测函数:\n")
            for col, func in patterns['detection_functions'].items():
                f.write(f"  {col}: {func}\n")
            
            f.write("\n修正函数:\n")
            for col, func in patterns['correction_functions'].items():
                f.write(f"  {col}: {func}\n")
            
            f.write("\n函数依赖:\n")
            for source, target in patterns['functional_dependencies']:
                f.write(f"  {source} -> {target}\n")
        
        print("  ✓ 清洗报告已保存到: results/cleaning_report.txt")
        
        print("\n" + "=" * 80)
        print("数据清洗完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请检查:")
        print("1. API密钥是否正确")
        print("2. 是否有足够的API配额")
        print("3. 网络连接是否正常")
        raise


if __name__ == "__main__":
    main()