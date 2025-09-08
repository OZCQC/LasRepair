import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ErrorType(Enum):
    FORMATTING = "formatting"
    SEMANTIC = "semantic"
    PATTERN = "pattern"
    CONTEXT_DEPENDENT = "context_dependent"


class CorrectionMethod(Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


@dataclass
class ErrorCell:
    row_idx: int
    col_idx: int
    value: Any
    error_type: ErrorType
    confidence: float
    context: Dict[str, Any]


@dataclass
class CorrectionResult:
    original_value: Any
    corrected_value: Any
    method: CorrectionMethod
    confidence: float
    rule: Optional[str] = None


class GIDCL:
    def __init__(self, 
                 llm_model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "auto",
                 config: Optional[Dict] = None):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_model_name = llm_model_name
        self.config = config or {}
        
        # Load configuration file if not provided
        if not config:
            self.config = self._load_default_config()
        
        self.implicit_corrector = None
        self.explicit_corrector = None
        self.graph_refiner = None
        self.selector = None
        
        self._initialize_components()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration from config.json if available."""
        import json
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _initialize_components(self):
        from implicit_correction import ImplicitCorrector
        from explicit_correction import ExplicitCorrector
        from graph_refinement import GraphRefiner
        from selection_mechanism import SelectionMechanism
        
        # Pass configuration to components
        implicit_config = {**self.config, **self.config.get('correction', {}).get('implicit', {})}
        explicit_config = {**self.config, **self.config.get('correction', {}).get('explicit', {})}
        
        self.implicit_corrector = ImplicitCorrector(self.llm_model_name, self.device, implicit_config)
        self.explicit_corrector = ExplicitCorrector(self.llm_model_name, self.device, explicit_config)
        self.graph_refiner = GraphRefiner(self.device)
        self.selector = SelectionMechanism()
    
    def repair_table(self, 
                    dirty_table: pd.DataFrame,
                    error_cells: List[ErrorCell],
                    schema_info: Optional[Dict] = None,
                    auxiliary_context: Optional[Dict] = None) -> Tuple[pd.DataFrame, List[CorrectionResult]]:
        
        repaired_table = dirty_table.copy()
        correction_results = []
        
        for error_cell in error_cells:
            method = self.selector.select_method(error_cell)
            
            if method == CorrectionMethod.IMPLICIT:
                result = self.implicit_corrector.correct_cell(
                    error_cell, dirty_table, schema_info, auxiliary_context
                )
            else:
                result = self.explicit_corrector.correct_cell(
                    error_cell, dirty_table, schema_info, auxiliary_context
                )
            
            repaired_table.iloc[error_cell.row_idx, error_cell.col_idx] = result.corrected_value
            correction_results.append(result)
        
        final_table = self.graph_refiner.refine_table(repaired_table, correction_results)
        
        return final_table, correction_results
    
    def train_implicit_model(self, 
                           training_data: List[Tuple[Any, Any]], 
                           validation_data: Optional[List[Tuple[Any, Any]]] = None,
                           epochs: int = 3,
                           batch_size: int = 8):
        return self.implicit_corrector.train(training_data, validation_data, epochs, batch_size)
    
    def generate_training_data(self, 
                             clean_tables: List[pd.DataFrame],
                             synthetic_error_rate: float = 0.1) -> List[Tuple[Any, Any]]:
        training_pairs = []
        
        for table in clean_tables:
            num_errors = int(len(table) * len(table.columns) * synthetic_error_rate)
            
            for _ in range(num_errors):
                row_idx = np.random.randint(0, len(table))
                col_idx = np.random.randint(0, len(table.columns))
                
                clean_value = table.iloc[row_idx, col_idx]
                dirty_value = self._introduce_synthetic_error(clean_value)
                
                training_pairs.append((dirty_value, clean_value))
        
        return training_pairs
    
    def _introduce_synthetic_error(self, clean_value):
        if pd.isna(clean_value):
            return clean_value
        
        if isinstance(clean_value, str):
            error_types = ['typo', 'case', 'extra_char', 'missing_char']
            error_type = np.random.choice(error_types)
            
            if error_type == 'typo' and len(clean_value) > 1:
                idx = np.random.randint(0, len(clean_value))
                chars = list(clean_value)
                chars[idx] = chr(ord(chars[idx]) + np.random.randint(-5, 6))
                return ''.join(chars)
            elif error_type == 'case':
                return clean_value.swapcase()
            elif error_type == 'extra_char':
                idx = np.random.randint(0, len(clean_value) + 1)
                return clean_value[:idx] + chr(np.random.randint(97, 123)) + clean_value[idx:]
            elif error_type == 'missing_char' and len(clean_value) > 1:
                idx = np.random.randint(0, len(clean_value))
                return clean_value[:idx] + clean_value[idx+1:]
        
        elif isinstance(clean_value, (int, float)):
            noise = np.random.normal(0, abs(clean_value) * 0.1)
            if isinstance(clean_value, int):
                return int(clean_value + noise)
            else:
                return clean_value + noise
        
        return clean_value