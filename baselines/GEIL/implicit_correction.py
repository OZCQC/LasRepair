import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json
import re

from llm_providers import LLMProviderFactory, RateLimitedLLMProvider


class ImplicitCorrector:
    def __init__(self, model_name: str, device: str, config: Optional[Dict] = None):
        self.model_name = model_name
        self.device = device
        self.config = config or {}
        
        # For backward compatibility, default to local model if no config provided
        self.use_api = self.config.get('use_api', False)
        
        if self.use_api:
            self._setup_api_model()
        else:
            self._setup_local_model()
        
        self.few_shot_examples = []
        self.retrieval_index = {}
    
    def _setup_api_model(self):
        """Setup API-based LLM provider."""
        provider_config = self.config.get('llm_provider', {})
        provider_type = provider_config.get('type', 'openai')
        api_keys = provider_config.get('api_keys', {})
        models = provider_config.get('models', {})
        
        if provider_type not in api_keys:
            raise ValueError(f"API key not found for provider: {provider_type}")
        
        api_key = api_keys[provider_type]
        model_name = models.get(provider_type, self.model_name)
        
        self.llm_provider = LLMProviderFactory.create_provider(
            provider_type, api_key, model_name
        )
        
        # Add rate limiting if enabled
        if provider_config.get('use_rate_limiting', True):
            rate_limit = provider_config.get('rate_limit', 60)
            self.llm_provider = RateLimitedLLMProvider(self.llm_provider, rate_limit)
        
        self.tokenizer = None
        self.model = None
        self.trained_model = None
    
    def _setup_local_model(self):
        """Setup local transformers model."""
        self.llm_provider = None
        self.tokenizer = None
        self.model = None
        self.trained_model = None
        self._load_base_model()
    
    def _load_base_model(self):
        """Load local transformers model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
    
    def correct_cell(self, 
                    error_cell, 
                    table: pd.DataFrame, 
                    schema_info: Optional[Dict] = None,
                    auxiliary_context: Optional[Dict] = None):
        """Correct a single error cell using either API or local model."""
        
        if self.use_api:
            return self._correct_cell_api(error_cell, table, schema_info, auxiliary_context)
        else:
            return self._correct_cell_local(error_cell, table, schema_info, auxiliary_context)
    
    def _correct_cell_api(self, error_cell, table, schema_info, auxiliary_context):
        """Correct cell using API-based LLM with few-shot prompting."""
        
        # Build context for the error
        context = self._build_context(error_cell, table, schema_info)
        
        # Create few-shot prompt
        prompt = self._create_few_shot_prompt(error_cell, context, auxiliary_context)
        
        try:
            # Generate correction using API
            response = self.llm_provider.generate(
                prompt,
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 200)
            )
            
            corrected_value = self._extract_correction_from_response(response.text, error_cell.value)
            confidence = self._calculate_confidence_api(response, error_cell)
            
        except Exception as e:
            print(f"API correction failed: {e}")
            # Fallback to simple correction
            corrected_value = self._fallback_correction(error_cell.value)
            confidence = 0.5
        
        from gidcl import CorrectionResult, CorrectionMethod
        return CorrectionResult(
            original_value=error_cell.value,
            corrected_value=corrected_value,
            method=CorrectionMethod.IMPLICIT,
            confidence=confidence
        )
    
    def _correct_cell_local(self, error_cell, table, schema_info, auxiliary_context):
        """Correct cell using local fine-tuned model."""
        
        context = self._build_context(error_cell, table, schema_info)
        
        # Use trained model if available, otherwise use base model
        model_to_use = self.trained_model if self.trained_model else self.model
        
        # Create prompt for local model
        prompt = self._create_local_prompt(error_cell, context)
        
        # Generate using local model
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model_to_use.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=self.config.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_value = self._extract_correction_from_local_response(generated_text, prompt)
        confidence = self._calculate_confidence_local(corrected_value, error_cell)
        
        from gidcl import CorrectionResult, CorrectionMethod
        return CorrectionResult(
            original_value=error_cell.value,
            corrected_value=corrected_value,
            method=CorrectionMethod.IMPLICIT,
            confidence=confidence
        )
    
    def _create_few_shot_prompt(self, error_cell, context, auxiliary_context):
        """Create few-shot prompt for API models."""
        
        prompt_parts = [
            "You are a data cleaning expert. Your task is to correct erroneous values in data tables.",
            "Here are some examples of corrections:",
            ""
        ]
        
        # Add few-shot examples
        examples = self._get_relevant_examples(error_cell, auxiliary_context)
        for i, (dirty, clean) in enumerate(examples[:5], 1):
            prompt_parts.extend([
                f"Example {i}:",
                f"Incorrect: '{dirty}'",
                f"Correct: '{clean}'",
                ""
            ])
        
        # Add current context
        prompt_parts.extend([
            "Current correction task:",
            f"Column: {context.get('column_name', 'unknown')}",
            f"Data type: {context.get('data_type', 'unknown')}",
            f"Row context: {context.get('row_values', {})}",
            ""
        ])
        
        # Add error information
        prompt_parts.extend([
            f"Incorrect value: '{error_cell.value}'",
            f"Error type: {error_cell.error_type.value}",
            f"Error confidence: {error_cell.confidence}",
            "",
            "Please provide only the corrected value, without explanation:",
            "Correct:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _create_local_prompt(self, error_cell, context):
        """Create prompt for local model."""
        return f"Correct this value: '{error_cell.value}' -> "
    
    def _build_context(self, error_cell, table, schema_info):
        """Build context information for the error cell."""
        context = {}
        
        # Column information
        if error_cell.col_idx < len(table.columns):
            column_name = table.columns[error_cell.col_idx]
            context['column_name'] = column_name
            context['data_type'] = str(table[column_name].dtype)
        
        # Row context (other column values in the same row)
        if error_cell.row_idx < len(table):
            row_values = {}
            for col_idx, col_name in enumerate(table.columns):
                if col_idx != error_cell.col_idx:
                    row_values[col_name] = table.iloc[error_cell.row_idx, col_idx]
            context['row_values'] = row_values
        
        # Schema information
        if schema_info:
            context.update(schema_info)
        
        return context
    
    def _get_relevant_examples(self, error_cell, auxiliary_context):
        """Get relevant few-shot examples."""
        examples = []
        
        # From auxiliary context
        if auxiliary_context and 'examples' in auxiliary_context:
            examples.extend(auxiliary_context['examples'])
        
        # From stored few-shot examples
        examples.extend(self.few_shot_examples)
        
        # Default examples based on error type
        if not examples:
            examples = self._get_default_examples(error_cell.error_type)
        
        return examples
    
    def _get_default_examples(self, error_type):
        """Get default examples based on error type."""
        from gidcl import ErrorType
        
        default_examples = {
            ErrorType.FORMATTING: [
                ("  John Doe  ", "John Doe"),
                ("JANE SMITH", "Jane Smith"),
                ("bob_johnson", "Bob Johnson")
            ],
            ErrorType.PATTERN: [
                ("123.456.7890", "123-456-7890"),
                ("john@emailcom", "john@email.com"),
                ("$1,000.00", "1000.00")
            ],
            ErrorType.SEMANTIC: [
                ("Califronia", "California"),
                ("Newyork", "New York"),
                ("23/12/2023", "2023-12-23")
            ],
            ErrorType.CONTEXT_DEPENDENT: [
                ("", "Unknown"),
                ("N/A", ""),
                ("null", "")
            ]
        }
        
        return default_examples.get(error_type, [])
    
    def _extract_correction_from_response(self, response_text, original_value):
        """Extract corrected value from API response."""
        # Clean up the response
        response_text = response_text.strip()
        
        # Look for common patterns
        patterns = [
            r"Correct:\s*['\"]?([^'\"\n]+)['\"]?",
            r"Corrected:\s*['\"]?([^'\"\n]+)['\"]?",
            r"Fixed:\s*['\"]?([^'\"\n]+)['\"]?",
            r"Answer:\s*['\"]?([^'\"\n]+)['\"]?",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, take the first line
        lines = response_text.split('\n')
        if lines:
            return lines[0].strip()
        
        return original_value
    
    def _extract_correction_from_local_response(self, generated_text, prompt):
        """Extract correction from local model response."""
        # Remove the prompt part
        correction = generated_text[len(prompt):].strip()
        
        # Take only the first word/phrase
        if '\n' in correction:
            correction = correction.split('\n')[0]
        
        return correction.strip()
    
    def _calculate_confidence_api(self, response, error_cell):
        """Calculate confidence for API-based correction."""
        base_confidence = 0.8
        
        # Adjust based on response quality
        if response.finish_reason == "stop":
            base_confidence += 0.1
        elif response.finish_reason == "length":
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _calculate_confidence_local(self, corrected_value, error_cell):
        """Calculate confidence for local model correction."""
        if corrected_value == error_cell.value:
            return 0.3  # No change, low confidence
        else:
            return 0.7  # Some change, moderate confidence
    
    def _fallback_correction(self, value):
        """Simple fallback correction."""
        if isinstance(value, str):
            return value.strip()
        return value
    
    def add_few_shot_examples(self, examples: List[Tuple[Any, Any]]):
        """Add few-shot examples for API-based correction."""
        self.few_shot_examples.extend(examples)
    
    def train(self, 
              training_data: List[Tuple[Any, Any]], 
              validation_data: Optional[List[Tuple[Any, Any]]] = None,
              epochs: int = 3,
              batch_size: int = 8) -> Dict[str, Any]:
        """Train the model (only for local models)."""
        
        if self.use_api:
            # For API models, store examples for few-shot prompting
            self.add_few_shot_examples(training_data)
            return {
                "training_loss": 0.0,
                "validation_loss": 0.0,
                "examples_added": len(training_data)
            }
        
        # Original training logic for local models
        train_texts = self._prepare_training_texts(training_data)
        
        train_dataset = Dataset.from_dict({"text": train_texts})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=True, 
                max_length=self.config.get('max_length', 512)
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        
        # Setup LoRA if enabled
        if self.config.get('use_lora', True):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.get('lora_rank', 8),
                lora_alpha=self.config.get('lora_alpha', 32),
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, lora_config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=self.config.get('learning_rate', 5e-5),
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if validation_data else "no",
            load_best_model_at_end=True if validation_data else False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Validation dataset
        val_dataset = None
        if validation_data:
            val_texts = self._prepare_training_texts(validation_data)
            val_dataset = Dataset.from_dict({"text": val_texts})
            val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save trained model
        self.trained_model = self.model
        
        return {
            "training_loss": train_result.training_loss,
            "validation_loss": train_result.metrics.get("eval_loss", 0.0) if validation_data else 0.0
        }
    
    def _prepare_training_texts(self, training_data: List[Tuple[Any, Any]]) -> List[str]:
        """Prepare training texts from dirty-clean pairs."""
        texts = []
        for dirty, clean in training_data:
            text = f"Correct this value: '{dirty}' -> '{clean}'"
            texts.append(text)
        return texts
    
    def build_retrieval_index(self, examples: List[Tuple[Any, Any]]):
        """Build retrieval index for RAG-style correction."""
        # Simple implementation - can be enhanced with embeddings
        for dirty, clean in examples:
            key = str(dirty).lower().strip()
            self.retrieval_index[key] = clean
    
    def retrieve_similar_corrections(self, value: Any, k: int = 3) -> List[Tuple[Any, Any]]:
        """Retrieve similar corrections from the index."""
        value_str = str(value).lower().strip()
        similar = []
        
        for dirty, clean in self.retrieval_index.items():
            if dirty == value_str:
                similar.append((dirty, clean))
            elif len(similar) < k and value_str in dirty:
                similar.append((dirty, clean))
        
        return similar[:k]