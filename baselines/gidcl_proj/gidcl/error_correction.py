"""
GIDCL错误修正模块 - LLM集成版
实现基于LLM的隐式和显式错误修正，以及基于图的依赖修正
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Any
import warnings
warnings.filterwarnings('ignore')

from .llm_interface import LLMInterface
from .utils import (
    apply_correction_function,
    extract_functional_dependencies,
    apply_functional_dependency
)


class ErrorCorrector:
    """
    错误修正器 - LLM集成版
    包含隐式修正（LLM生成）、显式修正（规则）和依赖修正（图）
    """
    
    def __init__(
        self,
        tau: float = 0.85,
        lambda_param: float = 4.0,
        llm_api_key: str = None,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        初始化错误修正器
        
        Args:
            tau: 函数接受阈值
            lambda_param: 图学习权重参数
            llm_api_key: LLM API密钥
            llm_model: LLM模型名称
        """
        self.tau = tau
        self.lambda_param = lambda_param
        
        # 初始化LLM接口
        self.llm = LLMInterface(api_key=llm_api_key, model=llm_model)
        
        self.correction_functions = {}  # F_corr: 显式修正函数
        self.functional_dependencies = []  # 发现的函数依赖
        
    def correct(
        self,
        df: pd.DataFrame,
        T_err: Set[Tuple[int, str]],
        T_label: Dict,
        T_pseudo: Dict,
        T_coreset: Set,
        graph,
        clusters: Dict[int, List[int]],
        error_detector
    ) -> pd.DataFrame:
        """
        执行错误修正
        
        Args:
            df: 原始数据表
            T_err: 检测到的错误
            T_label: 标注数据
            T_pseudo: 伪标签数据
            T_coreset: 高置信度干净数据
            graph: 图结构
            clusters: 聚类结果
            error_detector: 错误检测器
        
        Returns:
            修正后的数据表
        """
        df_clean = df.copy()
        
        print(f"  -> Correcting {len(T_err)} detected errors...")
        
        # Step 1: 使用LLM生成修正函数
        print("    - Using LLM to generate correction functions...")
        self._generate_correction_functions_with_llm(df, T_label)
        
        # Step 2: 隐式修正（基于LLM和上下文）
        print("    - Applying implicit corrections with LLM...")
        df_clean = self._implicit_correction_with_llm(
            df_clean, T_err, T_label, T_pseudo, T_coreset, clusters
        )
        
        # Step 3: 显式修正（基于规则）
        print("    - Applying explicit corrections...")
        df_clean = self._explicit_correction(
            df_clean, T_err, error_detector, clusters
        )
        
        # Step 4: 发现并应用函数依赖
        print("    - Discovering and applying functional dependencies...")
        self._discover_functional_dependencies(df_clean, T_label, T_coreset)
        df_clean = self._apply_functional_dependencies(
            df_clean, T_label, T_coreset
        )
        
        return df_clean
    
    def _generate_correction_functions_with_llm(
        self,
        df: pd.DataFrame,
        T_label: Dict
    ):
        """
        使用LLM生成修正函数
        """
        # 按列组织标注数据
        col_pairs = {}
        for (idx, col), label_info in T_label.items():
            if label_info['is_error']:
                if col not in col_pairs:
                    col_pairs[col] = []
                col_pairs[col].append((
                    label_info['dirty'],
                    label_info['clean']
                ))
        
        # 为每列生成修正函数
        for col, pairs in col_pairs.items():
            if not pairs:
                continue
            
            print(f"      Generating correction function for: {col}")
            
            try:
                # 使用LLM生成修正函数描述
                detection_pattern = ""  # 可以从检测器获取
                func_desc = self.llm.generate_correction_function(
                    column_name=col,
                    dirty_clean_pairs=pairs,
                    detection_pattern=detection_pattern
                )
                
                # 评估函数
                correct = 0
                for dirty, clean in pairs:
                    corrected = apply_correction_function(func_desc, dirty, pairs)
                    if str(corrected) == str(clean):
                        correct += 1
                
                accuracy = correct / len(pairs) if pairs else 0
                print(f"        Function: {func_desc[:60]}... (accuracy: {accuracy:.2f})")
                
                if accuracy >= self.tau:
                    self.correction_functions[col] = func_desc
                    
            except Exception as e:
                print(f"        Error generating function: {e}")
                continue
    
    def _implicit_correction_with_llm(
        self,
        df: pd.DataFrame,
        T_err: Set[Tuple[int, str]],
        T_label: Dict,
        T_pseudo: Dict,
        T_coreset: Set,
        clusters: Dict[int, List[int]]
    ) -> pd.DataFrame:
        """
        隐式修正：使用LLM基于上下文修正
        """
        df_corrected = df.copy()
        
        # 构建修正知识库（示例）
        examples_by_col = {}
        for (idx, col), label_info in {**T_label, **T_pseudo}.items():
            if label_info['is_error']:
                if col not in examples_by_col:
                    examples_by_col[col] = []
                examples_by_col[col].append((
                    label_info['dirty'],
                    label_info['clean']
                ))
        
        # 按列修正错误
        corrected_count = 0
        max_llm_calls = 20  # 限制LLM调用次数
        
        for idx, col in T_err:
            if corrected_count >= max_llm_calls:
                # 超过限制，使用简单方法
                self._simple_correction(df_corrected, idx, col, examples_by_col.get(col, []))
                continue
            
            value = df.iloc[idx][col]
            
            # 获取上下文值
            context_values = df[col].dropna().unique().tolist()[:10]
            examples = examples_by_col.get(col, [])[:5]
            
            if examples:
                try:
                    # 使用LLM修正
                    corrected_value = self.llm.correct_value(
                        dirty_value=value,
                        column_name=col,
                        examples=examples,
                        context_values=context_values
                    )
                    
                    df_corrected.iloc[idx, df_corrected.columns.get_loc(col)] = corrected_value
                    corrected_count += 1
                    
                except Exception as e:
                    # LLM失败，使用简单方法
                    self._simple_correction(df_corrected, idx, col, examples)
            else:
                # 没有示例，使用简单方法
                self._simple_correction(df_corrected, idx, col, examples)
        
        print(f"      Corrected {corrected_count} errors using LLM")
        return df_corrected
    
    def _simple_correction(
        self,
        df: pd.DataFrame,
        idx: int,
        col: str,
        examples: List[Tuple]
    ):
        """
        简单的修正方法（不使用LLM）
        """
        # 使用众数或示例
        if examples:
            from collections import Counter
            clean_values = [c for d, c in examples]
            most_common = Counter(clean_values).most_common(1)
            if most_common:
                df.iloc[idx, df.columns.get_loc(col)] = most_common[0][0]
        else:
            # 使用列的众数
            col_values = df[col].dropna().tolist()
            if col_values:
                from collections import Counter
                most_common = Counter(col_values).most_common(1)
                if most_common:
                    df.iloc[idx, df.columns.get_loc(col)] = most_common[0][0]
    
    def _explicit_correction(
        self,
        df: pd.DataFrame,
        T_err: Set[Tuple[int, str]],
        error_detector,
        clusters: Dict[int, List[int]]
    ) -> pd.DataFrame:
        """
        显式修正：应用修正函数
        """
        df_corrected = df.copy()
        
        for idx, col in T_err:
            if col in self.correction_functions:
                original_value = df.iloc[idx][col]
                func_desc = self.correction_functions[col]
                
                # 获取示例
                examples = []
                corrected_value = apply_correction_function(func_desc, original_value, examples)
                
                # 简单验证：检查修正后是否更好
                if pd.notna(corrected_value) and str(corrected_value) != str(original_value):
                    df_corrected.iloc[idx, df_corrected.columns.get_loc(col)] = corrected_value
        
        return df_corrected
    
    def _discover_functional_dependencies(
        self,
        df: pd.DataFrame,
        T_label: Dict,
        T_coreset: Set
    ):
        """
        发现函数依赖
        """
        # 构建高质量数据子集
        high_quality_indices = set()
        for (idx, col) in T_coreset:
            high_quality_indices.add(idx)
        for (idx, col) in T_label.keys():
            high_quality_indices.add(idx)
        
        if not high_quality_indices:
            return
        
        df_hq = df.iloc[list(high_quality_indices)]
        
        # 发现函数依赖
        self.functional_dependencies = extract_functional_dependencies(df_hq, confidence_threshold=0.9)
        
        print(f"      Found {len(self.functional_dependencies)} functional dependencies")
    
    def _apply_functional_dependencies(
        self,
        df: pd.DataFrame,
        T_label: Dict,
        T_coreset: Set
    ) -> pd.DataFrame:
        """
        应用函数依赖进行修正
        """
        if not self.functional_dependencies:
            return df
        
        df_corrected = df.copy()
        
        # 构建参考数据
        reference_indices = set()
        for (idx, col) in T_coreset:
            reference_indices.add(idx)
        for (idx, col) in T_label.keys():
            reference_indices.add(idx)
        
        if not reference_indices:
            return df_corrected
        
        df_ref = df.iloc[list(reference_indices)]
        
        # 应用每个函数依赖
        for fd in self.functional_dependencies:
            try:
                df_corrected = apply_functional_dependency(df_corrected, fd, df_ref)
            except Exception as e:
                print(f"      Warning: Failed to apply FD {fd}: {e}")
                continue
        
        return df_corrected
    
    def get_correction_functions(self) -> Dict:
        """获取修正函数"""
        return self.correction_functions
    
    def get_discovered_fds(self) -> List[Tuple[str, str]]:
        """获取发现的函数依赖"""
        return self.functional_dependencies