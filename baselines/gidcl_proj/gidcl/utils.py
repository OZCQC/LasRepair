"""
工具函数模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Any
import re
from collections import Counter


def evaluate_metrics(true_set: Set, pred_set: Set) -> Dict[str, float]:
    """
    计算精确率、召回率和F1分数
    
    Args:
        true_set: 真实的错误集合
        pred_set: 预测的错误集合
    
    Returns:
        包含precision, recall, f1的字典
    """
    if len(pred_set) == 0:
        precision = 0.0
    else:
        tp = len(true_set & pred_set)
        precision = tp / len(pred_set)
    
    if len(true_set) == 0:
        recall = 0.0
    else:
        tp = len(true_set & pred_set)
        recall = tp / len(true_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def serialize_tuple(row: pd.Series, sep: str = ' [SEP] ') -> str:
    """
    序列化一行数据为字符串
    
    Args:
        row: DataFrame的一行
        sep: 分隔符
    
    Returns:
        序列化后的字符串
    """
    parts = []
    for col, val in row.items():
        parts.append(f"<COL>{col}<VAL>{val}")
    return sep.join(parts)


def apply_pattern_function(pattern: str, value: Any) -> bool:
    """
    应用检测模式
    
    Args:
        pattern: 正则表达式模式
        value: 待检测的值
    
    Returns:
        True表示错误，False表示干净
    """
    if pd.isna(value):
        return True  # 缺失值视为错误
    
    value_str = str(value)
    
    try:
        match = re.match(pattern, value_str)
        # 如果不匹配模式，说明有错误
        return match is None
    except re.error:
        # 如果模式无效，返回False（假设干净）
        return False


def apply_corruption_function(func_description: str, value: Any) -> Any:
    """
    应用污染函数
    
    Args:
        func_description: 函数描述
        value: 干净值
    
    Returns:
        污染后的值
    """
    if pd.isna(value):
        return value
    
    if np.random.rand() > 0.7:  # 只有30%概率污染
        return value
    
    value_str = str(value)
    
    # 解析函数描述并应用
    desc_lower = func_description.lower()
    
    if 'replace' in desc_lower and 'digit' in desc_lower:
        # 替换数字
        if len(value_str) > 0 and any(c.isdigit() for c in value_str):
            pos = np.random.choice([i for i, c in enumerate(value_str) if c.isdigit()])
            return value_str[:pos] + 'x' + value_str[pos+1:]
    
    elif 'add' in desc_lower or 'insert' in desc_lower:
        # 添加字符
        if len(value_str) > 0:
            pos = np.random.randint(0, len(value_str) + 1)
            char = np.random.choice([',', '.', '-', 'x'])
            return value_str[:pos] + char + value_str[pos:]
    
    elif 'remove' in desc_lower or 'delete' in desc_lower:
        # 删除字符
        if len(value_str) > 1:
            pos = np.random.randint(0, len(value_str))
            return value_str[:pos] + value_str[pos+1:]
    
    elif 'missing' in desc_lower or 'null' in desc_lower:
        # 变为None
        if np.random.rand() > 0.5:
            return None
    
    # 默认：随机添加字符
    if len(value_str) > 0:
        pos = np.random.randint(0, len(value_str) + 1)
        return value_str[:pos] + 'x' + value_str[pos:]
    
    return value


def apply_correction_function(func_description: str, value: Any, examples: List[tuple] = None) -> Any:
    """
    应用修正函数
    
    Args:
        func_description: 函数描述
        value: 脏值
        examples: (脏值, 干净值)示例对
    
    Returns:
        修正后的值
    """
    if pd.isna(value):
        # 缺失值填充：使用示例中的众数
        if examples:
            clean_values = [c for d, c in examples if pd.notna(c)]
            if clean_values:
                return Counter(clean_values).most_common(1)[0][0]
        return value
    
    value_str = str(value)
    desc_lower = func_description.lower()
    
    # 先检查是否有直接匹配的示例
    if examples:
        for dirty, clean in examples:
            if str(dirty) == value_str:
                return clean
    
    # 根据描述应用修正
    if 'remove' in desc_lower and 'comma' in desc_lower:
        # 移除逗号及其后的内容
        return value_str.split(',')[0].strip()
    
    elif 'remove' in desc_lower and ('after' in desc_lower or 'extra' in desc_lower):
        # 移除额外字符
        return re.sub(r'[^A-Za-z0-9\s]', '', value_str)
    
    elif 'replace' in desc_lower and 'x' in desc_lower:
        # 替换x为数字
        if 'x' in value_str.lower():
            # 尝试从示例中推断正确的数字
            if examples:
                for dirty, clean in examples:
                    if 'x' in str(dirty).lower() and str(dirty).replace('x', '').replace('X', '') == value_str.replace('x', '').replace('X', ''):
                        # 找到相似的模式
                        return clean
            # 默认替换为1
            return re.sub(r'[xX]', '1', value_str)
    
    elif 'uppercase' in desc_lower or 'upper' in desc_lower:
        return value_str.upper()
    
    elif 'lowercase' in desc_lower or 'lower' in desc_lower:
        return value_str.lower()
    
    elif 'trim' in desc_lower or 'strip' in desc_lower:
        return value_str.strip()
    
    elif 'capitalize' in desc_lower:
        return value_str.capitalize()
    
    # 如果没有匹配的规则，返回原值
    return value


def extract_functional_dependencies(
    df: pd.DataFrame,
    confidence_threshold: float = 0.9
) -> List[Tuple[str, str]]:
    """
    提取函数依赖关系
    
    Args:
        df: 数据表
        confidence_threshold: 置信度阈值
    
    Returns:
        函数依赖列表 [(源列, 目标列), ...]
    """
    fds = []
    columns = df.columns.tolist()
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            # 检查 col1 -> col2
            grouped = df.groupby(col1)[col2].nunique()
            if len(grouped) > 0 and (grouped == 1).mean() >= confidence_threshold:
                fds.append((col1, col2))
            
            # 检查 col2 -> col1
            grouped = df.groupby(col2)[col1].nunique()
            if len(grouped) > 0 and (grouped == 1).mean() >= confidence_threshold:
                fds.append((col2, col1))
    
    return fds


def apply_functional_dependency(
    df: pd.DataFrame,
    fd: Tuple[str, str],
    reference_df: pd.DataFrame
) -> pd.DataFrame:
    """
    应用函数依赖进行修正
    
    Args:
        df: 待修正的数据表
        fd: 函数依赖 (源列, 目标列)
        reference_df: 参考数据表（高质量数据）
    
    Returns:
        修正后的数据表
    """
    source_col, target_col = fd
    df_copy = df.copy()
    
    # 从参考数据中构建映射
    mapping = {}
    for idx in reference_df.index:
        source_val = reference_df.loc[idx, source_col]
        target_val = reference_df.loc[idx, target_col]
        
        if pd.notna(source_val) and pd.notna(target_val):
            if source_val not in mapping:
                mapping[source_val] = []
            mapping[source_val].append(target_val)
    
    # 使用众数
    for key in mapping:
        most_common = Counter(mapping[key]).most_common(1)
        if most_common:
            mapping[key] = most_common[0][0]
    
    # 应用映射
    for idx in df_copy.index:
        source_val = df_copy.loc[idx, source_col]
        if source_val in mapping:
            df_copy.loc[idx, target_col] = mapping[source_val]
    
    return df_copy


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)