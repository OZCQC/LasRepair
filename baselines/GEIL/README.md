# GIDCL: Graph-Enhanced Interpretable Data Cleaning Framework

A comprehensive implementation of the GIDCL (Graph-Enhanced Interpretable Data Cleaning) framework using Large Language Models for automated data repair.

## Overview

GIDCL combines three key components for effective data cleaning:

1. **Implicit Correction**: Fine-tuned LLMs for direct value correction
2. **Explicit Correction**: Interpretable rule/function generation  
3. **Graph-Based Refinement**: GNN-powered global consistency enforcement

## Features

- **Dual Correction Methods**: Automatically selects between implicit LLM-based correction and explicit rule generation
- **Multiple LLM Providers**: Supports OpenAI, Anthropic, Google, Nebius, DeepSeek, and local models
- **API Integration**: Seamlessly switch between cloud APIs and local transformers
- **Graph Neural Networks**: Learns structural dependencies for global consistency
- **Interpretable Rules**: Generates human-readable correction functions
- **Adaptive Selection**: Learns optimal correction method per error type
- **Rate Limiting**: Built-in API rate limiting and error handling
- **Comprehensive Evaluation**: Built-in metrics and benchmarking tools

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- torch>=1.9.0
- transformers>=4.20.0
- torch-geometric>=2.1.0
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- datasets>=2.0.0
- peft>=0.3.0
- accelerate>=0.20.0
- networkx>=2.6.0
- openai>=1.0.0 (for OpenAI API)
- anthropic>=0.18.0 (for Claude API)
- google-generativeai>=0.3.0 (for Gemini API)
- requests>=2.28.0 (for HTTP APIs)

## Quick Start

### Basic Usage

#### With API Providers (Recommended)

```python
from gidcl import GIDCL, ErrorCell, ErrorType
import pandas as pd

# Configuration for API provider
config = {
    "llm_provider": {
        "type": "openai",  # or "anthropic", "google", "nebius", "deepseek"
        "api_keys": {
            "openai": "your-openai-api-key"
        },
        "models": {
            "openai": "gpt-4o-mini"
        }
    },
    "correction": {
        "implicit": {"use_api": True},
        "explicit": {"use_api": True}
    }
}

# Initialize GIDCL with API provider
gidcl = GIDCL(device="cpu", config=config)

# Load your dirty table
dirty_table = pd.read_csv("dirty_data.csv")

# Define error cells (normally detected automatically)
error_cells = [
    ErrorCell(row_idx=0, col_idx=1, value="Jhon", 
             error_type=ErrorType.FORMATTING, confidence=0.9, context={})
]

# Repair the table
repaired_table, corrections = gidcl.repair_table(
    dirty_table, error_cells, schema_info={}, auxiliary_context={}
)
```

#### With Local Models

```python
# Initialize GIDCL with local model (legacy mode)
gidcl = GIDCL(llm_model_name="microsoft/DialoGPT-medium", device="cpu")

# Same usage as above...
```

### Command Line Interface

```bash
# Basic repair
python main.py --input dirty_data.csv --output repaired_data.csv

# With training
python main.py --input dirty_data.csv --output repaired_data.csv --train --training-data clean_data.csv

# With verbose output
python main.py --input dirty_data.csv --output repaired_data.csv --verbose
```

### Run Demo

```bash
# Run complete demonstration
python example.py --demo

# Run benchmarks
python example.py --benchmark

# Run both
python example.py --all

# Test API integration with different providers
python example_api_integration.py --provider openai
python example_api_integration.py --provider anthropic
python example_api_integration.py --provider google
python example_api_integration.py --compare  # Compare all providers
```

## Architecture

### Core Components

1. **GIDCL Main Framework** (`gidcl.py`)
   - Orchestrates the entire repair process
   - Manages component interactions
   - Handles training data generation

2. **Implicit Correction** (`implicit_correction.py`)
   - Fine-tunes LLMs using LoRA for efficient training
   - Incorporates RAG for context-aware correction
   - Handles in-context learning with examples

3. **Explicit Correction** (`explicit_correction.py`)
   - Generates interpretable correction functions
   - Iteratively refines functions based on feedback
   - Supports various correction patterns (regex, string ops, etc.)

4. **Selection Mechanism** (`selection_mechanism.py`)
   - Routes errors to appropriate correction method
   - Supports both heuristic and learned selection
   - Adaptive performance-based routing

5. **Graph Refinement** (`graph_refinement.py`)
   - Constructs table graphs with GNNs
   - Discovers functional and inclusion dependencies
   - Enforces global consistency constraints

### Data Flow

```
Dirty Table → Error Detection → Method Selection → Correction → Graph Refinement → Clean Table
                                     ↓
                              Implicit ← → Explicit
                             (LLM-based)  (Rule-based)
```

## Configuration

The framework uses a JSON configuration file (`config.json`) to customize behavior:

### API Configuration

```json
{
  "llm_provider": {
    "type": "openai",
    "api_keys": {
      "openai": "your-openai-api-key",
      "anthropic": "your-anthropic-api-key",
      "google": "your-google-api-key"
    },
    "models": {
      "openai": "gpt-4o-mini",
      "anthropic": "claude-3-haiku-20240307",
      "google": "gemini-1.5-flash"
    },
    "rate_limit": 60,
    "use_rate_limiting": true
  },
  "correction": {
    "implicit": {
      "use_api": true,
      "temperature": 0.7,
      "max_tokens": 200,
      "few_shot_examples": 5
    },
    "explicit": {
      "use_api": true,
      "temperature": 0.3,
      "max_tokens": 300,
      "max_iterations": 3,
      "confidence_threshold": 0.7
    }
  }
}
```

### Local Model Configuration

```json
{
  "correction": {
    "implicit": {
      "use_api": false,
      "model_name": "microsoft/DialoGPT-medium",
      "training_epochs": 3,
      "batch_size": 8,
      "use_lora": true
    },
    "explicit": {
      "use_api": false,
      "max_iterations": 3,
      "confidence_threshold": 0.7
    }
  },
  "graph_refinement": {
    "fd_threshold": 0.95,
    "gnn_hidden_dim": 64
  }
}
```

## Error Types

The framework handles various error types:

- **FORMATTING**: Whitespace, case, punctuation issues
- **PATTERN**: Regex-based format violations  
- **SEMANTIC**: Context-dependent meaning errors
- **CONTEXT_DEPENDENT**: Errors requiring row/table context

## Training

### Generate Training Data

```python
# From clean tables
training_data = gidcl.generate_training_data(
    clean_tables=[clean_df1, clean_df2], 
    synthetic_error_rate=0.1
)

# Train implicit model
gidcl.train_implicit_model(training_data, epochs=5)
```

### Selection Model Training

```python
# Train selection mechanism
training_examples = [(error_cell, chosen_method), ...]
gidcl.selector.train_selection_model(training_examples)
```

## Evaluation

### Built-in Metrics

```python
from utils import EvaluationMetrics

# Calculate repair accuracy
metrics = EvaluationMetrics.calculate_repair_accuracy(
    original_table, dirty_table, repaired_table, error_positions
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

### Column-wise Analysis

```python
# Analyze performance per column
column_metrics = EvaluationMetrics.calculate_column_wise_metrics(
    original_table, dirty_table, repaired_table
)
```

## API Providers

### Supported Providers

1. **OpenAI** (`openai`)
   - Models: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
   - Best for: General-purpose correction, high accuracy

2. **Anthropic** (`anthropic`)
   - Models: Claude-3-Haiku, Claude-3-Sonnet, Claude-3-Opus
   - Best for: Complex reasoning, safety-critical applications

3. **Google** (`google`)
   - Models: Gemini-1.5-Flash, Gemini-1.5-Pro
   - Best for: Fast inference, cost-effective

4. **Nebius** (`nebius`)
   - Models: Meta-Llama-3.1-70B-Instruct, others
   - Best for: Open-source models, custom deployments

5. **DeepSeek** (`deepseek`)
   - Models: DeepSeek-Chat, DeepSeek-Coder
   - Best for: Code generation, technical corrections

### Provider Selection

```python
# Programmatic selection
config = {
    "llm_provider": {
        "type": "anthropic",  # Change provider here
        "api_keys": {...},
        "models": {...}
    }
}

# Command line selection
python main.py --input data.csv --output repaired.csv --provider anthropic
```

## Advanced Features

### Custom Error Detection

```python
def custom_error_detector(table):
    errors = []
    # Your detection logic
    return errors

# Use in pipeline
error_cells = custom_error_detector(dirty_table)
```

### External Context Integration

```python
auxiliary_context = {
    "external_data": ["valid_value1", "valid_value2"],
    "examples": [("wrong", "correct"), ("bad", "good")]
}

repaired_table, corrections = gidcl.repair_table(
    dirty_table, error_cells, schema_info, auxiliary_context
)
```

### Performance Profiling

```python
from utils import Profiler

profiler = Profiler()

with profiler.time_operation("repair"):
    repaired_table, corrections = gidcl.repair_table(...)
```

## Examples

### Phone Number Correction

```python
# Explicit correction generates function like:
def correct_phone(x):
    import re
    return re.sub(r'(\d{3})(\d{3})(\d{4})', r'\1-\2-\3', x)
```

### Name Formatting

```python
# Implicit correction uses fine-tuned LLM:
# "john DOE" → "John Doe"
```

### Dependency Enforcement

```python
# Graph refinement enforces FD: City → State
# If City="Boston" implies State="MA"
```

## Benchmarking

Run comprehensive benchmarks:

```bash
python example.py --benchmark
```

Results show performance across different:
- Table sizes (50×5 to 200×10)
- Error rates (10% to 20%)
- Error types (formatting, semantic, etc.)

## Limitations

- Requires sufficient training data for optimal performance
- Graph construction can be computationally expensive for large tables
- LLM inference adds latency compared to rule-based methods
- GPU recommended for large-scale deployments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This implementation is for research and educational purposes.

## Citation

If you use this implementation, please cite the original GIDCL paper:

```bibtex
@article{gidcl2024,
  title={GIDCL: A Graph-Enhanced Interpretable Data Cleaning Framework with Large Language Models},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```