"""
GIDCL核心类 - 完整可运行版本
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

from .graph_structure_learning import GraphStructureLearning
from .error_detection import ErrorDetector
from .error_correction import ErrorCorrector
from .utils import evaluate_metrics


class GIDCL:
    """
    GIDCL主类：集成图结构学习、错误检测和错误修正
    """
    
    def __init__(
        self,
        num_clusters: int = 20,
        labeling_budget: int = 20,
        tau: float = 0.85,
        lambda_param: float = 4.0,
        llm_api_key: str = None,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        初始化GIDCL系统
        
        Args:
            num_clusters: 聚类数量
            labeling_budget: 标注预算（最多标注的元组数）
            tau: 函数接受阈值
            lambda_param: 图学习中的权重参数
            llm_api_key: LLM API密钥
            llm_model: LLM模型名称
        """
        self.num_clusters = num_clusters
        self.labeling_budget = labeling_budget
        self.tau = tau
        self.lambda_param = lambda_param
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        
        # 初始化各个模块
        self.gsl = GraphStructureLearning(num_clusters=num_clusters)
        self.error_detector = ErrorDetector(
            tau=tau,
            llm_api_key=llm_api_key,
            llm_model=llm_model
        )
        self.error_corrector = ErrorCorrector(
            tau=tau,
            lambda_param=lambda_param,
            llm_api_key=llm_api_key,
            llm_model=llm_model
        )
        
        self.T_label = None  # 标注数据
        self.T_pseudo = None  # 伪标注数据
        self.T_coreset = None  # 高置信度的干净数据
        self.T_err = None  # 检测到的错误
        
    def fit(self, dirty_table: pd.DataFrame, ground_truth: pd.DataFrame = None):
        """
        训练GIDCL模型
        
        Args:
            dirty_table: 脏数据表
            ground_truth: 真实数据（用于自动标注，如果为None则需要人工标注）
        
        Returns:
            cleaned_table: 清洗后的数据表
        """
        print("=" * 80)
        print("GIDCL: Graph-Enhanced Interpretable Data Cleaning with LLMs")
        print("=" * 80)
        
        # Step 1: 图结构学习和样本选择
        print("\n[Step 1/4] Graph Structure Learning and Sample Selection...")
        representative_indices = self.gsl.fit_transform(
            dirty_table, 
            labeling_budget=self.labeling_budget
        )
        
        # Step 2: 获取标注数据
        print(f"\n[Step 2/4] Collecting Labels for {len(representative_indices)} representative tuples...")
        self.T_label = self._collect_labels(
            dirty_table, 
            representative_indices, 
            ground_truth
        )
        
        if len(self.T_label) == 0:
            raise ValueError("No labeled data available. Please provide ground truth.")
        
        print(f"  -> Collected {len(self.T_label)} labeled cells")
        
        # Step 3: 错误检测（Creator-Critic工作流）
        print("\n[Step 3/4] Error Detection (Creator-Critic Workflow)...")
        detection_results = self.error_detector.fit(
            dirty_table,
            self.T_label,
            self.gsl.get_clusters()
        )
        
        self.T_err = detection_results['T_err']
        self.T_pseudo = detection_results['T_pseudo']
        self.T_coreset = detection_results['T_coreset']
        
        print(f"  -> Detected {len(self.T_err)} errors")
        print(f"  -> Generated {len(self.T_pseudo)} pseudo-labeled samples")
        print(f"  -> Identified {len(self.T_coreset)} high-confidence clean cells")
        
        # Step 4: 错误修正
        print("\n[Step 4/4] Error Correction...")
        cleaned_table = self.error_corrector.correct(
            dirty_table,
            self.T_err,
            self.T_label,
            self.T_pseudo,
            self.T_coreset,
            self.gsl.get_graph(),
            self.gsl.get_clusters(),
            self.error_detector
        )
        
        print("\n" + "=" * 80)
        print("Data Cleaning Completed!")
        print("=" * 80)
        
        # 如果有ground truth，评估结果
        if ground_truth is not None:
            self._evaluate_results(dirty_table, cleaned_table, ground_truth)
        
        return cleaned_table
    
    def _collect_labels(
        self, 
        dirty_table: pd.DataFrame, 
        indices: List[int], 
        ground_truth: pd.DataFrame = None
    ) -> Dict:
        """
        收集标注数据
        
        Args:
            dirty_table: 脏数据表
            indices: 需要标注的行索引
            ground_truth: 真实数据（如果有）
        
        Returns:
            标注数据字典
        """
        T_label = {}
        
        if ground_truth is not None:
            # 自动标注
            for idx in indices:
                if idx >= len(dirty_table) or idx >= len(ground_truth):
                    continue
                    
                dirty_tuple = dirty_table.iloc[idx]
                clean_tuple = ground_truth.iloc[idx]
                
                # 记录每个单元格的标注
                for col in dirty_table.columns:
                    cell_key = (idx, col)
                    dirty_val = dirty_tuple[col]
                    clean_val = clean_tuple[col]
                    
                    # 处理NaN比较
                    is_error = False
                    if pd.isna(dirty_val) and pd.isna(clean_val):
                        is_error = False
                    elif pd.isna(dirty_val) or pd.isna(clean_val):
                        is_error = True
                    else:
                        is_error = str(dirty_val) != str(clean_val)
                    
                    T_label[cell_key] = {
                        'dirty': dirty_val,
                        'clean': clean_val,
                        'is_error': is_error
                    }
        else:
            raise ValueError("Ground truth is required for automatic labeling.")
        
        return T_label
    
    def _evaluate_results(
        self, 
        dirty_table: pd.DataFrame, 
        cleaned_table: pd.DataFrame, 
        ground_truth: pd.DataFrame
    ):
        """
        评估清洗结果
        """
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        
        # 计算错误检测指标
        if self.T_err is not None:
            detection_metrics = self._evaluate_detection(dirty_table, ground_truth)
            print("\nError Detection Performance:")
            print(f"  Precision: {detection_metrics['precision']:.4f}")
            print(f"  Recall: {detection_metrics['recall']:.4f}")
            print(f"  F1-Score: {detection_metrics['f1']:.4f}")
        
        # 计算错误修正指标
        correction_metrics = self._evaluate_correction(
            dirty_table, cleaned_table, ground_truth
        )
        print("\nError Correction Performance:")
        print(f"  Precision: {correction_metrics['precision']:.4f}")
        print(f"  Recall: {correction_metrics['recall']:.4f}")
        print(f"  F1-Score: {correction_metrics['f1']:.4f}")
        
        # 计算整体准确率
        total_cells = len(dirty_table) * len(dirty_table.columns)
        correct_cells = 0
        
        for i in range(len(cleaned_table)):
            for col in cleaned_table.columns:
                cleaned_val = cleaned_table.iloc[i][col]
                truth_val = ground_truth.iloc[i][col]
                
                if pd.isna(cleaned_val) and pd.isna(truth_val):
                    correct_cells += 1
                elif not pd.isna(cleaned_val) and not pd.isna(truth_val):
                    if str(cleaned_val) == str(truth_val):
                        correct_cells += 1
        
        accuracy = correct_cells / total_cells if total_cells > 0 else 0
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print("=" * 80)
    
    def _evaluate_detection(
        self, 
        dirty_table: pd.DataFrame, 
        ground_truth: pd.DataFrame
    ) -> Dict:
        """评估错误检测性能"""
        true_errors = set()
        detected_errors = set(self.T_err)
        
        # 找出所有真实错误
        for i in range(len(dirty_table)):
            for col in dirty_table.columns:
                dirty_val = dirty_table.iloc[i][col]
                truth_val = ground_truth.iloc[i][col]
                
                is_error = False
                if pd.isna(dirty_val) and pd.isna(truth_val):
                    is_error = False
                elif pd.isna(dirty_val) or pd.isna(truth_val):
                    is_error = True
                else:
                    is_error = str(dirty_val) != str(truth_val)
                
                if is_error:
                    true_errors.add((i, col))
        
        return evaluate_metrics(true_errors, detected_errors)
    
    def _evaluate_correction(
        self, 
        dirty_table: pd.DataFrame,
        cleaned_table: pd.DataFrame, 
        ground_truth: pd.DataFrame
    ) -> Dict:
        """评估错误修正性能"""
        # 真实错误集合
        true_errors = set()
        for i in range(len(dirty_table)):
            for col in dirty_table.columns:
                dirty_val = dirty_table.iloc[i][col]
                truth_val = ground_truth.iloc[i][col]
                
                is_error = False
                if pd.isna(dirty_val) and pd.isna(truth_val):
                    is_error = False
                elif pd.isna(dirty_val) or pd.isna(truth_val):
                    is_error = True
                else:
                    is_error = str(dirty_val) != str(truth_val)
                
                if is_error:
                    true_errors.add((i, col))
        
        # 修正的错误集合（原来错误，现在正确）
        corrected_errors = set()
        for i in range(len(cleaned_table)):
            for col in cleaned_table.columns:
                cleaned_val = cleaned_table.iloc[i][col]
                truth_val = ground_truth.iloc[i][col]
                dirty_val = dirty_table.iloc[i][col]
                
                # 检查是否原来错误
                was_error = False
                if pd.isna(dirty_val) and pd.isna(truth_val):
                    was_error = False
                elif pd.isna(dirty_val) or pd.isna(truth_val):
                    was_error = True
                else:
                    was_error = str(dirty_val) != str(truth_val)
                
                # 检查是否现在正确
                is_correct = False
                if pd.isna(cleaned_val) and pd.isna(truth_val):
                    is_correct = True
                elif not pd.isna(cleaned_val) and not pd.isna(truth_val):
                    is_correct = str(cleaned_val) == str(truth_val)
                
                if was_error and is_correct:
                    corrected_errors.add((i, col))
        
        return evaluate_metrics(true_errors, corrected_errors)
    
    def get_interpretable_patterns(self) -> Dict:
        """
        获取可解释的数据清洗模式
        
        Returns:
            包含检测函数、修正函数和发现的依赖关系的字典
        """
        return {
            'detection_functions': self.error_detector.get_detection_functions(),
            'correction_functions': self.error_corrector.get_correction_functions(),
            'functional_dependencies': self.error_corrector.get_discovered_fds()
        }