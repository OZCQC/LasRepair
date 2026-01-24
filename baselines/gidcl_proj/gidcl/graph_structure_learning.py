"""
GIDCL图结构学习模块
实现将关系表转换为图，学习图嵌入，并进行聚类和样本选择
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx


class GraphStructureLearning:
    """
    图结构学习类
    负责图构建、嵌入学习、聚类和代表性样本选择
    """
    
    def __init__(self, num_clusters: int = 20, embedding_dim: int = 128):
        """
        初始化图结构学习模块
        
        Args:
            num_clusters: 聚类数量
            embedding_dim: 嵌入维度
        """
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        
        self.graph = None  # 构建的图
        self.embeddings = None  # 节点嵌入
        self.clusters = None  # 聚类结果
        self.cluster_labels = None  # 每个元组的聚类标签
        
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        labeling_budget: int = 20
    ) -> List[int]:
        """
        执行图结构学习并选择代表性样本
        
        Args:
            df: 输入数据表
            labeling_budget: 标注预算
        
        Returns:
            需要标注的行索引列表
        """
        print("  -> Building graph from table...")
        self.graph = self._build_graph(df)
        
        print("  -> Learning graph embeddings...")
        self.embeddings = self._learn_embeddings(df)
        
        print("  -> Clustering tuples...")
        self.clusters, self.cluster_labels = self._cluster_tuples()
        
        print("  -> Selecting representative samples...")
        representative_indices = self._select_representative_samples(
            df, 
            labeling_budget
        )
        
        return representative_indices
    
    def _build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        将关系表转换为有向属性图
        
        图结构: (tuple_node, attribute_edge, value_node)
        
        Args:
            df: 输入数据表
        
        Returns:
            有向图
        """
        G = nx.DiGraph()
        
        # 为每一行创建中心节点
        for idx in range(len(df)):
            tuple_node = f"tuple_{idx}"
            G.add_node(tuple_node, type='tuple', index=idx)
            
            # 为每个属性值创建边和值节点
            for col in df.columns:
                value = df.iloc[idx][col]
                value_node = f"value_{col}_{value}"
                
                # 添加值节点
                if not G.has_node(value_node):
                    G.add_node(value_node, type='value', attribute=col, value=value)
                
                # 添加从元组到值的边（带属性标签）
                G.add_edge(tuple_node, value_node, attribute=col)
        
        return G
    
    def _learn_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        学习图节点嵌入
        
        使用简化的方法：基于特征向量的嵌入
        
        Args:
            df: 输入数据表
        
        Returns:
            元组的嵌入矩阵 (n_tuples, embedding_dim)
        """
        # 简化方法：使用one-hot编码 + 降维
        embeddings = []
        
        for idx in range(len(df)):
            # 为每个元组创建特征向量
            features = []
            
            for col in df.columns:
                value = df.iloc[idx][col]
                
                # 数值特征
                if pd.api.types.is_numeric_dtype(df[col]):
                    if pd.notna(value):
                        features.append(float(value))
                    else:
                        features.append(0.0)
                # 类别特征（使用哈希）
                else:
                    value_str = str(value)
                    # 简单哈希到[0,1]范围
                    hash_val = hash(value_str) % 10000 / 10000.0
                    features.append(hash_val)
            
            embeddings.append(features)
        
        embeddings = np.array(embeddings)
        
        # 标准化
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        
        # 如果特征维度不够，进行填充；如果太多，进行降维
        if embeddings.shape[1] < self.embedding_dim:
            # 填充零
            padding = np.zeros((embeddings.shape[0], 
                               self.embedding_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        elif embeddings.shape[1] > self.embedding_dim:
            # PCA降维（简化版：只取前N个特征）
            embeddings = embeddings[:, :self.embedding_dim]
        
        return embeddings
    
    def _cluster_tuples(self) -> Tuple[Dict[int, List[int]], np.ndarray]:
        """
        对元组进行聚类
        
        Returns:
            (聚类字典, 聚类标签数组)
        """
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # 构建聚类字典
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        return clusters, cluster_labels
    
    def _select_representative_samples(
        self, 
        df: pd.DataFrame, 
        budget: int
    ) -> List[int]:
        """
        选择代表性样本进行标注
        
        策略：
        1. 计算每个聚类的内部平均相似度
        2. 选择相似度最低的聚类（更有可能包含错误）
        3. 在选中的聚类中选择最异常的样本
        
        Args:
            df: 数据表
            budget: 标注预算
        
        Returns:
            选中的行索引列表
        """
        # 计算每个聚类的平均内部距离
        cluster_scores = {}
        
        for cluster_id, indices in self.clusters.items():
            if len(indices) <= 1:
                cluster_scores[cluster_id] = 0.0
                continue
            
            # 计算聚类内所有点对的平均距离
            cluster_embeddings = self.embeddings[indices]
            distances = []
            for i in range(len(cluster_embeddings)):
                for j in range(i+1, len(cluster_embeddings)):
                    dist = np.linalg.norm(
                        cluster_embeddings[i] - cluster_embeddings[j]
                    )
                    distances.append(dist)
            
            avg_distance = np.mean(distances) if distances else 0.0
            cluster_scores[cluster_id] = avg_distance
        
        # 选择平均距离最大的聚类（内部最不一致的聚类）
        sorted_clusters = sorted(
            cluster_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 从选中的聚类中选择样本
        selected_indices = []
        
        for cluster_id, _ in sorted_clusters:
            if len(selected_indices) >= budget:
                break
            
            indices = self.clusters[cluster_id]
            
            if len(indices) == 0:
                continue
            
            # 在聚类中选择最异常的点（距离聚类中心最远的点）
            cluster_embeddings = self.embeddings[indices]
            center = cluster_embeddings.mean(axis=0)
            
            distances = [
                np.linalg.norm(cluster_embeddings[i] - center) 
                for i in range(len(cluster_embeddings))
            ]
            
            # 选择距离最大的点
            outlier_idx = np.argmax(distances)
            selected_indices.append(indices[outlier_idx])
        
        # 如果还不够，随机选择
        if len(selected_indices) < budget:
            remaining = budget - len(selected_indices)
            all_indices = set(range(len(df)))
            available = list(all_indices - set(selected_indices))
            if available:
                additional = np.random.choice(
                    available, 
                    size=min(remaining, len(available)), 
                    replace=False
                )
                selected_indices.extend(additional.tolist())
        
        return selected_indices[:budget]
    
    def get_graph(self) -> nx.DiGraph:
        """获取构建的图"""
        return self.graph
    
    def get_clusters(self) -> Dict[int, List[int]]:
        """获取聚类结果"""
        return self.clusters
    
    def get_cluster_label(self, row_idx: int) -> int:
        """
        获取指定行的聚类标签
        
        Args:
            row_idx: 行索引
        
        Returns:
            聚类标签
        """
        if self.cluster_labels is not None and row_idx < len(self.cluster_labels):
            return self.cluster_labels[row_idx]
        return 0
    
    def update_graph(self, df: pd.DataFrame):
        """
        更新图结构（用于错误修正后的重新学习）
        
        Args:
            df: 更新后的数据表
        """
        print("  -> Updating graph structure...")
        self.graph = self._build_graph(df)
        self.embeddings = self._learn_embeddings(df)