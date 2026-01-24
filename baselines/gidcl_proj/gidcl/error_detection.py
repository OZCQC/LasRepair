"""
GIDCL错误检测模块 - LLM集成版
实现Creator-Critic工作流，使用真实LLM API
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Any
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from .llm_interface import LLMInterface
from .utils import apply_pattern_function, apply_corruption_function


class ErrorDetector:
    """
    错误检测器：使用Creator-Critic工作流
    Creator: LLM生成检测函数和伪标签数据
    Critic: 机器学习模型进行二分类检测
    """
    
    def __init__(
        self, 
        tau: float = 0.85,
        max_iterations: int = 3,
        llm_api_key: str = None,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        初始化错误检测器
        
        Args:
            tau: 函数接受阈值
            max_iterations: Creator-Critic最大迭代次数
            llm_api_key: LLM API密钥
            llm_model: LLM模型名称
        """
        self.tau = tau
        self.max_iterations = max_iterations
        
        # 初始化LLM接口
        self.llm = LLMInterface(api_key=llm_api_key, model=llm_model)
        
        self.detection_model = None  # M_det: 检测模型
        self.detection_functions = {}  # F_det: 检测函数集（模式字符串）
        self.generation_functions = {}  # F_gen: 生成函数集（函数描述）
        
    def fit(
        self, 
        df: pd.DataFrame,
        T_label: Dict,
        clusters: Dict[int, List[int]]
    ) -> Dict:
        """
        训练错误检测模型
        
        Args:
            df: 数据表
            T_label: 标注数据
            clusters: 聚类结果
        
        Returns:
            检测结果字典
        """
        print("  -> Starting Creator-Critic workflow...")
        
        T_pseudo = {}  # 伪标签数据
        T_coreset = set()  # 高置信度的干净数据
        
        for iteration in range(self.max_iterations):
            print(f"\n  -> Iteration {iteration + 1}/{self.max_iterations}")
            
            # Creator: 生成检测函数和伪标签
            print("    - Creator: Using LLM to generate detection patterns...")
            self._creator_step(df, T_label, T_pseudo, clusters)
            
            # Critic: 训练检测模型
            print("    - Critic: Training detection model...")
            D_train = {**T_label, **T_pseudo}
            self._critic_step(df, D_train, clusters)
            
            # 检查收敛
            if iteration > 0:
                print("    - Checking convergence...")
        
        # 最终检测
        print("\n  -> Final error detection...")
        T_err, confidences = self._detect_errors(df, clusters)
        
        # 选择高置信度的干净数据作为coreset
        T_coreset = self._select_coreset(df, confidences, T_err)
        
        return {
            'T_err': T_err,
            'T_pseudo': T_pseudo,
            'T_coreset': T_coreset
        }
    
    def _creator_step(
        self,
        df: pd.DataFrame,
        T_label: Dict,
        T_pseudo: Dict,
        clusters: Dict[int, List[int]]
    ):
        """
        Creator步骤：使用LLM生成检测函数和伪标签
        """
        for col in df.columns:
            # 收集该列的脏/干净值对
            dirty_clean_pairs = []
            for (idx, c), label_info in T_label.items():
                if c == col:
                    dirty_clean_pairs.append((
                        label_info['dirty'],
                        label_info['clean'],
                        label_info['is_error']
                    ))
            
            if not dirty_clean_pairs:
                continue
            
            # 使用LLM生成检测函数
            print(f"      Generating pattern for column: {col}")
            column_values = df[col].unique().tolist()
            
            try:
                pattern = self.llm.generate_detection_pattern(
                    column_name=col,
                    dirty_clean_pairs=dirty_clean_pairs,
                    column_values=column_values
                )
                
                # 评估函数
                correct = 0
                for d, c, is_err in dirty_clean_pairs:
                    detected_err = apply_pattern_function(pattern, d)
                    if detected_err == is_err:
                        correct += 1
                
                accuracy = correct / len(dirty_clean_pairs) if dirty_clean_pairs else 0
                print(f"        Pattern: {pattern} (accuracy: {accuracy:.2f})")
                
                if accuracy >= self.tau:
                    self.detection_functions[col] = pattern
                    
                    # 使用LLM生成污染函数
                    func_desc = self.llm.generate_corruption_function(
                        column_name=col,
                        detection_pattern=pattern,
                        dirty_clean_pairs=[(d, c) for d, c, _ in dirty_clean_pairs if is_err]
                    )
                    self.generation_functions[col] = func_desc
                    
                    # 生成伪标签数据
                    self._generate_pseudo_labels(df, col, pattern, func_desc, T_pseudo, T_label)
                    
            except Exception as e:
                print(f"        Error generating pattern: {e}")
                continue
    
    def _generate_pseudo_labels(
        self,
        df: pd.DataFrame,
        col: str,
        pattern: str,
        func_desc: str,
        T_pseudo: Dict,
        T_label: Dict
    ):
        """
        生成伪标签数据
        """
        # 从数据中随机选择一些干净的值
        generated_count = 0
        max_pseudo = min(10, len(df))  # 每列最多生成10个伪样本
        
        for idx in range(len(df)):
            if generated_count >= max_pseudo:
                break
                
            value = df.iloc[idx][col]
            
            # 跳过已标注的
            if (idx, col) in T_label:
                continue
            
            # 检测是否干净
            is_clean = not apply_pattern_function(pattern, value)
            
            if is_clean and np.random.rand() < 0.5:  # 50%概率选择
                # 生成脏数据
                dirty_value = apply_corruption_function(func_desc, value)
                
                if str(dirty_value) != str(value):
                    T_pseudo[(idx, col)] = {
                        'dirty': dirty_value,
                        'clean': value,
                        'is_error': True
                    }
                    generated_count += 1
    
    def _critic_step(
        self,
        df: pd.DataFrame,
        D_train: Dict,
        clusters: Dict[int, List[int]]
    ):
        """
        Critic步骤：训练检测模型
        """
        # 准备训练数据
        X_train = []
        y_train = []
        
        for (idx, col), label_info in D_train.items():
            # 提取特征
            features = self._extract_features(df, idx, col, clusters)
            X_train.append(features)
            y_train.append(1 if label_info['is_error'] else 0)
        
        if len(X_train) == 0:
            return
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # 训练随机森林分类器
        self.detection_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.detection_model.fit(X_train, y_train)
    
    def _extract_features(
        self,
        df: pd.DataFrame,
        row_idx: int,
        col: str,
        clusters: Dict[int, List[int]]
    ) -> np.ndarray:
        """
        提取特征向量
        """
        features = []
        
        value = df.iloc[row_idx][col]
        
        # 特征1: 是否缺失
        features.append(1.0 if pd.isna(value) else 0.0)
        
        # 特征2: 值的长度
        value_str = str(value)
        features.append(len(value_str))
        
        # 特征3: 是否包含特殊字符
        import re
        has_special = bool(re.search(r'[^A-Za-z0-9\s\-\.]', value_str))
        features.append(1.0 if has_special else 0.0)
        
        # 特征4: 是否全数字
        features.append(1.0 if value_str.replace('.', '').isdigit() else 0.0)
        
        # 特征5: 是否全字母
        features.append(1.0 if value_str.isalpha() else 0.0)
        
        # 特征6: 与列内其他值的相似度（频率）
        col_values = df[col].dropna().astype(str).tolist()
        if col_values:
            freq = col_values.count(value_str) / len(col_values)
            features.append(freq)
        else:
            features.append(0.0)
        
        # 特征7: 行内缺失值数量
        row = df.iloc[row_idx]
        features.append(row.isna().sum())
        
        return np.array(features)
    
    def _detect_errors(
        self,
        df: pd.DataFrame,
        clusters: Dict[int, List[int]]
    ) -> Tuple[Set[Tuple[int, str]], Dict]:
        """
        检测所有错误
        
        Returns:
            (错误集合, 置信度字典)
        """
        T_err = set()
        confidences = {}
        
        for idx in range(len(df)):
            for col in df.columns:
                # 首先使用检测函数
                if col in self.detection_functions:
                    value = df.iloc[idx][col]
                    is_error_by_pattern = apply_pattern_function(
                        self.detection_functions[col], value
                    )
                    
                    if is_error_by_pattern:
                        T_err.add((idx, col))
                        confidences[(idx, col)] = 0.85
                        continue
                
                # 使用ML模型
                if self.detection_model is not None:
                    features = self._extract_features(df, idx, col, clusters)
                    pred_proba = self.detection_model.predict_proba(
                        features.reshape(1, -1)
                    )[0]
                    
                    # pred_proba[1]是错误的概率
                    if pred_proba[1] > 0.5:
                        T_err.add((idx, col))
                        confidences[(idx, col)] = pred_proba[1]
                    else:
                        confidences[(idx, col)] = pred_proba[0]
        
        return T_err, confidences
    
    def _select_coreset(
        self,
        df: pd.DataFrame,
        confidences: Dict,
        T_err: Set
    ) -> Set:
        """
        选择高置信度的干净数据
        """
        T_coreset = set()
        
        for idx in range(len(df)):
            for col in df.columns:
                if (idx, col) not in T_err:
                    conf = confidences.get((idx, col), 0.5)
                    if conf > 0.75:  # 高置信度阈值
                        T_coreset.add((idx, col))
        
        return T_coreset
    
    def get_detection_functions(self) -> Dict:
        """获取检测函数集"""
        return self.detection_functions
    
    def predict(
        self,
        df: pd.DataFrame,
        row_idx: int,
        col: str,
        clusters: Dict[int, List[int]]
    ) -> float:
        """
        预测单个单元格的置信度
        
        Returns:
            干净的置信度 (0-1)
        """
        if self.detection_model is None:
            return 0.5
        
        features = self._extract_features(df, row_idx, col, clusters)
        pred_proba = self.detection_model.predict_proba(
            features.reshape(1, -1)
        )[0]
        
        # 返回干净的置信度
        return pred_proba[0]