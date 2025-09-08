import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import pickle


class GraphRefiner:
    def __init__(self, device: str):
        self.device = device
        self.gnn_model = None
        self.dependency_detector = DependencyDetector()
        self.graph_builder = GraphBuilder()
        self.trained = False
        
    def refine_table(self, 
                    table: pd.DataFrame, 
                    correction_results: List) -> pd.DataFrame:
        
        if not self.trained:
            self._train_on_table(table)
        
        graph_data = self.graph_builder.build_graph(table)
        
        dependencies = self.dependency_detector.discover_dependencies(table, graph_data)
        
        refined_table = self._apply_dependency_constraints(table, dependencies, correction_results)
        
        return refined_table
    
    def _train_on_table(self, table: pd.DataFrame):
        graph_data = self.graph_builder.build_graph(table)
        
        self.gnn_model = GNNModel(
            input_dim=graph_data.x.shape[1],
            hidden_dim=64,
            output_dim=32
        ).to(self.device)
        
        self._train_gnn([graph_data])
        self.trained = True
    
    def _train_gnn(self, graph_data_list: List[Data], epochs: int = 100):
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for data in graph_data_list:
                data = data.to(self.device)
                optimizer.zero_grad()
                
                out = self.gnn_model(data.x, data.edge_index)
                
                loss = self._compute_reconstruction_loss(out, data.x)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(graph_data_list):.4f}")
    
    def _compute_reconstruction_loss(self, embeddings: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        reconstructed = torch.mm(embeddings, embeddings.t())
        target = torch.mm(original_features, original_features.t())
        return F.mse_loss(reconstructed, target)
    
    def _apply_dependency_constraints(self, 
                                    table: pd.DataFrame, 
                                    dependencies: List[Dict], 
                                    correction_results: List) -> pd.DataFrame:
        
        refined_table = table.copy()
        
        for dep in dependencies:
            if dep['type'] == 'functional_dependency':
                refined_table = self._enforce_functional_dependency(refined_table, dep)
            elif dep['type'] == 'inclusion_dependency':
                refined_table = self._enforce_inclusion_dependency(refined_table, dep)
        
        return refined_table
    
    def _enforce_functional_dependency(self, table: pd.DataFrame, dependency: Dict) -> pd.DataFrame:
        determinant_cols = dependency['determinant']
        dependent_col = dependency['dependent']
        
        if not all(col in table.columns for col in determinant_cols + [dependent_col]):
            return table
        
        refined_table = table.copy()
        
        grouped = table.groupby(determinant_cols)[dependent_col].agg(['first', 'nunique'])
        
        for group_key, group_data in grouped.iterrows():
            if group_data['nunique'] > 1:
                most_common_value = group_data['first']
                
                if isinstance(group_key, tuple):
                    mask = (table[determinant_cols] == pd.Series(group_key, index=determinant_cols)).all(axis=1)
                else:
                    mask = table[determinant_cols[0]] == group_key
                
                refined_table.loc[mask, dependent_col] = most_common_value
        
        return refined_table
    
    def _enforce_inclusion_dependency(self, table: pd.DataFrame, dependency: Dict) -> pd.DataFrame:
        source_col = dependency['source']
        target_col = dependency['target']
        
        if source_col not in table.columns or target_col not in table.columns:
            return table
        
        refined_table = table.copy()
        target_values = set(table[target_col].dropna().unique())
        
        for idx, value in enumerate(table[source_col]):
            if pd.notna(value) and value not in target_values:
                closest_match = self._find_closest_match(value, target_values)
                if closest_match:
                    refined_table.iloc[idx, table.columns.get_loc(source_col)] = closest_match
        
        return refined_table
    
    def _find_closest_match(self, value: Any, candidates: set) -> Any:
        if not isinstance(value, str):
            return None
        
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            if isinstance(candidate, str):
                score = self._string_similarity(value, candidate)
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = candidate
        
        return best_match
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


class GNNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x


class DependencyDetector:
    def __init__(self):
        self.fd_threshold = 0.95
        self.id_threshold = 0.8
        
    def discover_dependencies(self, table: pd.DataFrame, graph_data: Data) -> List[Dict]:
        dependencies = []
        
        functional_deps = self._detect_functional_dependencies(table)
        dependencies.extend(functional_deps)
        
        inclusion_deps = self._detect_inclusion_dependencies(table)
        dependencies.extend(inclusion_deps)
        
        return dependencies
    
    def _detect_functional_dependencies(self, table: pd.DataFrame) -> List[Dict]:
        functional_deps = []
        columns = table.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    if self._test_functional_dependency(table, [col1], col2):
                        functional_deps.append({
                            'type': 'functional_dependency',
                            'determinant': [col1],
                            'dependent': col2,
                            'confidence': self._calculate_fd_confidence(table, [col1], col2)
                        })
        
        for col1, col2 in combinations(columns, 2):
            for target_col in columns:
                if target_col not in [col1, col2]:
                    if self._test_functional_dependency(table, [col1, col2], target_col):
                        functional_deps.append({
                            'type': 'functional_dependency',
                            'determinant': [col1, col2],
                            'dependent': target_col,
                            'confidence': self._calculate_fd_confidence(table, [col1, col2], target_col)
                        })
        
        return functional_deps
    
    def _test_functional_dependency(self, table: pd.DataFrame, determinant_cols: List[str], dependent_col: str) -> bool:
        try:
            grouped = table.groupby(determinant_cols)[dependent_col].nunique()
            violations = (grouped > 1).sum()
            total_groups = len(grouped)
            
            if total_groups == 0:
                return False
            
            accuracy = 1 - (violations / total_groups)
            return accuracy >= self.fd_threshold
        except:
            return False
    
    def _calculate_fd_confidence(self, table: pd.DataFrame, determinant_cols: List[str], dependent_col: str) -> float:
        try:
            grouped = table.groupby(determinant_cols)[dependent_col].nunique()
            violations = (grouped > 1).sum()
            total_groups = len(grouped)
            
            if total_groups == 0:
                return 0.0
            
            return 1 - (violations / total_groups)
        except:
            return 0.0
    
    def _detect_inclusion_dependencies(self, table: pd.DataFrame) -> List[Dict]:
        inclusion_deps = []
        columns = table.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    if self._test_inclusion_dependency(table, col1, col2):
                        inclusion_deps.append({
                            'type': 'inclusion_dependency',
                            'source': col1,
                            'target': col2,
                            'confidence': self._calculate_id_confidence(table, col1, col2)
                        })
        
        return inclusion_deps
    
    def _test_inclusion_dependency(self, table: pd.DataFrame, source_col: str, target_col: str) -> bool:
        try:
            source_values = set(table[source_col].dropna().unique())
            target_values = set(table[target_col].dropna().unique())
            
            if len(source_values) == 0:
                return False
            
            included = len(source_values.intersection(target_values))
            inclusion_ratio = included / len(source_values)
            
            return inclusion_ratio >= self.id_threshold
        except:
            return False
    
    def _calculate_id_confidence(self, table: pd.DataFrame, source_col: str, target_col: str) -> float:
        try:
            source_values = set(table[source_col].dropna().unique())
            target_values = set(table[target_col].dropna().unique())
            
            if len(source_values) == 0:
                return 0.0
            
            included = len(source_values.intersection(target_values))
            return included / len(source_values)
        except:
            return 0.0


class GraphBuilder:
    def __init__(self):
        self.label_encoders = {}
        
    def build_graph(self, table: pd.DataFrame) -> Data:
        node_features = self._create_node_features(table)
        edge_index = self._create_edges(table)
        
        return Data(x=node_features, edge_index=edge_index)
    
    def _create_node_features(self, table: pd.DataFrame) -> torch.Tensor:
        features = []
        
        for col in table.columns:
            col_features = self._extract_column_features(table[col])
            features.append(col_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_column_features(self, column: pd.Series) -> List[float]:
        features = []
        
        # Basic statistics
        features.extend([
            len(column),
            column.nunique(),
            column.isna().sum(),
            column.isna().sum() / len(column) if len(column) > 0 else 0
        ])
        
        # Data type features
        dtype_str = str(column.dtype)
        features.extend([
            1 if 'int' in dtype_str else 0,
            1 if 'float' in dtype_str else 0,
            1 if 'object' in dtype_str else 0,
            1 if 'datetime' in dtype_str else 0
        ])
        
        # String-specific features
        if column.dtype == 'object':
            string_lengths = column.astype(str).str.len()
            features.extend([
                string_lengths.mean(),
                string_lengths.std(),
                string_lengths.min(),
                string_lengths.max()
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Numeric-specific features
        if column.dtype in ['int64', 'float64']:
            features.extend([
                column.mean(),
                column.std(),
                column.min(),
                column.max()
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def _create_edges(self, table: pd.DataFrame) -> torch.Tensor:
        edges = []
        columns = table.columns.tolist()
        
        # Create edges based on correlation and co-occurrence
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    # Calculate correlation if both columns are numeric
                    if self._is_numeric_column(table[col1]) and self._is_numeric_column(table[col2]):
                        corr = abs(table[col1].corr(table[col2]))
                        if not pd.isna(corr) and corr > 0.3:
                            edges.append([i, j])
                    
                    # Calculate co-occurrence for categorical columns
                    elif self._is_categorical_column(table[col1]) and self._is_categorical_column(table[col2]):
                        cooccurrence = self._calculate_cooccurrence(table[col1], table[col2])
                        if cooccurrence > 0.5:
                            edges.append([i, j])
        
        if not edges:
            # If no edges found, create a minimal connected graph
            for i in range(len(columns) - 1):
                edges.append([i, i + 1])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _is_numeric_column(self, column: pd.Series) -> bool:
        return column.dtype in ['int64', 'float64']
    
    def _is_categorical_column(self, column: pd.Series) -> bool:
        return column.dtype == 'object' or column.nunique() < len(column) * 0.5
    
    def _calculate_cooccurrence(self, col1: pd.Series, col2: pd.Series) -> float:
        try:
            contingency_table = pd.crosstab(col1, col2)
            total = contingency_table.sum().sum()
            
            if total == 0:
                return 0.0
            
            # Calculate normalized mutual information
            mutual_info = 0
            for i in contingency_table.index:
                for j in contingency_table.columns:
                    if contingency_table.loc[i, j] > 0:
                        p_ij = contingency_table.loc[i, j] / total
                        p_i = contingency_table.loc[i, :].sum() / total
                        p_j = contingency_table.loc[:, j].sum() / total
                        mutual_info += p_ij * np.log(p_ij / (p_i * p_j))
            
            return mutual_info
        except:
            return 0.0