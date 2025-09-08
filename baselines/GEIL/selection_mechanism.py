import pandas as pd
import numpy as np
from typing import Dict, Any
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


class SelectionMechanism:
    def __init__(self):
        self.decision_model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = []
        
    def select_method(self, error_cell):
        from gidcl import CorrectionMethod, ErrorType
        
        if not self.is_trained:
            return self._heuristic_selection(error_cell)
        
        features = self._extract_features(error_cell)
        prediction = self.decision_model.predict([features])[0]
        
        return CorrectionMethod.IMPLICIT if prediction == 0 else CorrectionMethod.EXPLICIT
    
    def _heuristic_selection(self, error_cell):
        from gidcl import CorrectionMethod, ErrorType
        
        if error_cell.error_type in [ErrorType.FORMATTING, ErrorType.PATTERN]:
            return CorrectionMethod.EXPLICIT
        elif error_cell.error_type in [ErrorType.SEMANTIC, ErrorType.CONTEXT_DEPENDENT]:
            return CorrectionMethod.IMPLICIT
        else:
            if isinstance(error_cell.value, str):
                if self._has_pattern_error(error_cell.value):
                    return CorrectionMethod.EXPLICIT
                else:
                    return CorrectionMethod.IMPLICIT
            else:
                return CorrectionMethod.IMPLICIT
    
    def _has_pattern_error(self, value: str) -> bool:
        pattern_indicators = [
            r'\d{3}-?\d{3}-?\d{4}',  # Phone numbers
            r'\w+@\w+\.\w+',        # Email patterns
            r'\$\d+\.?\d*',         # Currency
            r'\d+%',                # Percentages
            r'\d{2}/\d{2}/\d{4}',   # Dates
        ]
        
        for pattern in pattern_indicators:
            if re.search(pattern, value):
                return True
        
        if any(char in value for char in ['_', '-', '.', '@', '%', '$']):
            return True
        
        return False
    
    def train_selection_model(self, training_examples):
        features = []
        labels = []
        
        for error_cell, method in training_examples:
            feature_vector = self._extract_features(error_cell)
            features.append(feature_vector)
            labels.append(0 if method.value == "implicit" else 1)
        
        self.decision_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.decision_model.fit(features, labels)
        self.is_trained = True
        
        return self.decision_model.score(features, labels)
    
    def _extract_features(self, error_cell) -> list:
        features = []
        
        # Error type features
        error_type_encoding = {
            'formatting': [1, 0, 0, 0],
            'semantic': [0, 1, 0, 0],
            'pattern': [0, 0, 1, 0],
            'context_dependent': [0, 0, 0, 1]
        }
        features.extend(error_type_encoding.get(error_cell.error_type.value, [0, 0, 0, 0]))
        
        # Value type features
        value = error_cell.value
        features.extend([
            1 if isinstance(value, str) else 0,
            1 if isinstance(value, (int, float)) else 0,
            1 if pd.isna(value) else 0
        ])
        
        # String-specific features
        if isinstance(value, str):
            features.extend([
                len(value),
                len(value.split()),
                1 if any(char.isdigit() for char in value) else 0,
                1 if any(char.isupper() for char in value) else 0,
                1 if any(char in value for char in '!@#$%^&*()_+-=[]{}|;:,.<>?') else 0,
                1 if re.search(r'\d{2,}', value) else 0,  # Has multi-digit numbers
                1 if '@' in value else 0,
                1 if '%' in value else 0,
                1 if '$' in value else 0,
            ])
        else:
            features.extend([0] * 9)
        
        # Confidence and context features
        features.extend([
            error_cell.confidence,
            len(error_cell.context) if error_cell.context else 0
        ])
        
        if not hasattr(self, 'feature_names') or not self.feature_names:
            self.feature_names = [
                'error_formatting', 'error_semantic', 'error_pattern', 'error_context',
                'is_string', 'is_numeric', 'is_null',
                'str_length', 'word_count', 'has_digits', 'has_upper', 'has_special',
                'has_multi_digit', 'has_at', 'has_percent', 'has_dollar',
                'confidence', 'context_size'
            ]
        
        return features
    
    def save_model(self, filepath: str):
        if self.is_trained:
            model_data = {
                'decision_model': self.decision_model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.decision_model = model_data['decision_model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
    
    def explain_decision(self, error_cell) -> Dict[str, Any]:
        features = self._extract_features(error_cell)
        
        explanation = {
            'selected_method': self.select_method(error_cell).value,
            'features': dict(zip(self.feature_names, features)),
            'reasoning': []
        }
        
        # Add heuristic reasoning
        if error_cell.error_type.value in ['formatting', 'pattern']:
            explanation['reasoning'].append("Pattern/formatting errors favor explicit correction")
        
        if isinstance(error_cell.value, str) and self._has_pattern_error(error_cell.value):
            explanation['reasoning'].append("Value contains pattern indicators")
        
        if error_cell.error_type.value in ['semantic', 'context_dependent']:
            explanation['reasoning'].append("Semantic errors favor implicit correction")
        
        if self.is_trained and self.decision_model:
            feature_importance = dict(zip(
                self.feature_names, 
                self.decision_model.feature_importances_
            ))
            explanation['feature_importance'] = feature_importance
        
        return explanation


class AdaptiveSelector(SelectionMechanism):
    def __init__(self):
        super().__init__()
        self.performance_history = {
            'implicit': {'successes': 0, 'attempts': 0},
            'explicit': {'successes': 0, 'attempts': 0}
        }
        self.adaptation_threshold = 0.1
    
    def update_performance(self, method, success: bool):
        method_name = method.value if hasattr(method, 'value') else str(method)
        
        if method_name in self.performance_history:
            self.performance_history[method_name]['attempts'] += 1
            if success:
                self.performance_history[method_name]['successes'] += 1
    
    def get_method_performance(self, method_name: str) -> float:
        history = self.performance_history.get(method_name, {'successes': 0, 'attempts': 1})
        if history['attempts'] == 0:
            return 0.5
        return history['successes'] / history['attempts']
    
    def select_method(self, error_cell):
        base_method = super().select_method(error_cell)
        
        implicit_perf = self.get_method_performance('implicit')
        explicit_perf = self.get_method_performance('explicit')
        
        performance_diff = abs(implicit_perf - explicit_perf)
        
        if performance_diff > self.adaptation_threshold:
            if implicit_perf > explicit_perf:
                from .gidcl import CorrectionMethod
                return CorrectionMethod.IMPLICIT
            else:
                from .gidcl import CorrectionMethod
                return CorrectionMethod.EXPLICIT
        
        return base_method