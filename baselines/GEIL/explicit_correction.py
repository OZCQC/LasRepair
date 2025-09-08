import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import re
import ast
from typing import List, Tuple, Optional, Dict, Any, Callable
import json

from llm_providers import LLMProviderFactory, RateLimitedLLMProvider


class ExplicitCorrector:
    def __init__(self, model_name: str, device: str, config: Optional[Dict] = None):
        self.model_name = model_name
        self.device = device
        self.config = config or {}
        self.function_cache = {}
        
        # For backward compatibility, default to local model if no config provided
        self.use_api = self.config.get('use_api', False)
        
        if self.use_api:
            self._setup_api_model()
        else:
            self._setup_local_model()
    
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
    
    def _setup_local_model(self):
        """Setup local transformers model."""
        self.llm_provider = None
        self._load_model()
    
    def _load_model(self):
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
        
        context = self._build_context(error_cell, table, schema_info)
        
        if self.use_api:
            correction_function = self._generate_correction_function_api(error_cell, context)
        else:
            correction_function = self._generate_correction_function_local(error_cell, context)
        
        corrected_value, final_function = self._apply_and_refine_function(
            correction_function, error_cell, context
        )
        
        confidence = self._calculate_confidence(corrected_value, error_cell, final_function)
        
        from gidcl import CorrectionResult, CorrectionMethod
        return CorrectionResult(
            original_value=error_cell.value,
            corrected_value=corrected_value,
            method=CorrectionMethod.EXPLICIT,
            confidence=confidence,
            rule=final_function
        )
    
    def _generate_correction_function_api(self, error_cell, context):
        """Generate correction function using API model."""
        prompt = self._create_function_generation_prompt(error_cell, context)
        
        try:
            response = self.llm_provider.generate(
                prompt,
                temperature=self.config.get('temperature', 0.3),
                max_tokens=self.config.get('max_tokens', 300)
            )
            
            function_code = self._extract_function_from_response(response.text)
            return function_code
            
        except Exception as e:
            print(f"API function generation failed: {e}")
            return self._generate_fallback_function(error_cell)
    
    def _generate_correction_function_local(self, error_cell, context):
        """Generate correction function using local model."""
        prompt = self._create_function_generation_prompt(error_cell, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.config.get('max_function_tokens', 200),
                temperature=self.config.get('temperature', 0.3),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        function_code = self._extract_function_from_local_response(generated_text, prompt)
        
        return function_code
    
    def _create_function_generation_prompt(self, error_cell, context):
        """Create prompt for function generation."""
        from gidcl import ErrorType
        
        prompt_parts = [
            "You are a data cleaning expert. Generate a Python function to correct erroneous data values.",
            "",
            "Task: Create a correction function for the following error:",
            f"- Value: '{error_cell.value}'",
            f"- Error Type: {error_cell.error_type.value}",
            f"- Column: {context.get('column_name', 'unknown')}",
            f"- Data Type: {context.get('data_type', 'unknown')}",
            ""
        ]
        
        # Add examples based on error type
        if error_cell.error_type == ErrorType.FORMATTING:
            prompt_parts.extend([
                "Examples of formatting corrections:",
                "def correct_value(x):",
                "    return x.strip().title()  # Remove whitespace and fix case",
                "",
                "def correct_value(x):",
                "    return re.sub(r'\\s+', ' ', x)  # Normalize whitespace",
                ""
            ])
        elif error_cell.error_type == ErrorType.PATTERN:
            prompt_parts.extend([
                "Examples of pattern corrections:",
                "def correct_value(x):",
                "    return re.sub(r'(\\d{3})(\\d{3})(\\d{4})', r'\\1-\\2-\\3', x)  # Phone format",
                "",
                "def correct_value(x):",
                "    return x.replace('.', '-')  # Fix separator",
                ""
            ])
        
        prompt_parts.extend([
            "Generate a correction function for the given value. The function should:",
            "1. Take a single parameter 'x' (the input value)",
            "2. Return the corrected value",
            "3. Handle edge cases (None, empty strings, etc.)",
            "4. Use appropriate imports (re, str methods, etc.)",
            "",
            "Write only the function definition, starting with 'def correct_value(x):':",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_function_from_response(self, response_text):
        """Extract function code from API response."""
        # Look for function definition
        function_pattern = r'def\s+correct_value\s*\([^)]*\):\s*(?:\n|$)((?:\s+.+(?:\n|$))*)'
        match = re.search(function_pattern, response_text, re.MULTILINE)
        
        if match:
            function_body = match.group(1)
            full_function = f"def correct_value(x):\n{function_body}"
            extracted = full_function.strip()
            if self._is_function_complete(extracted):
                return extracted
        
        # Fallback: look for any code block
        code_block_pattern = r'```(?:python)?\s*(def\s+correct_value.*?)```'
        match = re.search(code_block_pattern, response_text, re.DOTALL)
        
        if match:
            extracted = match.group(1).strip()
            if self._is_function_complete(extracted):
                return extracted
        
        # Last resort: try to find any function-like structure
        lines = response_text.split('\n')
        function_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def correct_value'):
                in_function = True
                function_lines.append(line)
            elif in_function and line.strip():
                if line.startswith('    ') or line.startswith('\t'):
                    function_lines.append(line)
                else:
                    break
            elif in_function and not line.strip():
                function_lines.append(line)
        
        if function_lines:
            extracted = '\n'.join(function_lines)
            # Only return if it looks reasonably complete
            if self._is_function_complete(extracted):
                return extracted
            else:
                print(f"Extracted function appears incomplete: {extracted[:100]}...")
        
        return None
    
    def _extract_function_from_local_response(self, generated_text, prompt):
        """Extract function from local model response."""
        # Remove the prompt part
        function_code = generated_text[len(prompt):].strip()
        
        # Clean up the response
        if function_code.startswith('def correct_value'):
            return function_code
        
        return None
    
    def _generate_fallback_function(self, error_cell):
        """Generate a simple fallback function."""
        from gidcl import ErrorType
        
        if error_cell.error_type == ErrorType.FORMATTING:
            return """def correct_value(x):
    if isinstance(x, str):
        return x.strip()
    return x"""
        elif error_cell.error_type == ErrorType.PATTERN:
            return """def correct_value(x):
    import re
    if isinstance(x, str):
        return re.sub(r'[^a-zA-Z0-9\\s\\-\\.]', '', x)
    return x"""
        else:
            return """def correct_value(x):
    return x"""
    
    def _build_context(self, error_cell, table: pd.DataFrame, schema_info: Optional[Dict]) -> Dict[str, Any]:
        """Build context for function generation."""
        context = {}
        
        # Column information
        if error_cell.col_idx < len(table.columns):
            column_name = table.columns[error_cell.col_idx]
            context['column_name'] = column_name
            context['data_type'] = str(table[column_name].dtype)
            
            # Sample values from the column
            sample_values = table[column_name].dropna().head(10).tolist()
            context['sample_values'] = sample_values
        
        # Row context
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
    
    def _apply_and_refine_function(self, 
                                 function_code: str, 
                                 error_cell, 
                                 context: Dict[str, Any]) -> Tuple[Any, str]:
        """Apply the generated function and refine if needed."""
        
        max_iterations = self.config.get('max_iterations', 3)
        current_function = function_code
        
        for iteration in range(max_iterations):
            try:
                # Test the function
                corrected_value = self._execute_function(current_function, error_cell.value)
                
                # Validate the result
                if self._validate_correction(corrected_value, error_cell, context):
                    return corrected_value, current_function
                
                # If validation fails, try to refine the function
                if iteration < max_iterations - 1:
                    current_function = self._refine_function(
                        current_function, error_cell, corrected_value, context
                    )
                else:
                    # Last iteration, return what we have
                    return corrected_value, current_function
                    
            except Exception as e:
                print(f"Function execution failed (iteration {iteration + 1}): {e}")
                
                if iteration < max_iterations - 1:
                    current_function = self._fix_function_syntax(current_function, str(e))
                else:
                    # Fallback to simple correction
                    return self._simple_correction(error_cell.value), current_function
        
        return error_cell.value, current_function
    
    def _execute_function(self, function_code: str, input_value: Any) -> Any:
        """Safely execute the generated function."""
        
        # Check if function is cached
        function_hash = hash(function_code)
        if function_hash in self.function_cache:
            correction_func = self.function_cache[function_hash]
        else:
            # Create a safe execution environment
            import builtins
            import pandas as pd
            import numpy as np
            safe_globals = {
                're': re,
                'pd': pd,
                'pandas': pd,
                'np': np,
                'numpy': np,
                'str': str,
                'int': int,
                'float': float,
                'len': len,
                'isinstance': isinstance,
                'type': type,
                'range': range,
                'enumerate': enumerate,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'abs': abs,
                'max': max,
                'min': min,
                'sum': sum,
                'any': any,
                'all': all,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'zip': zip,
                '__builtins__': {
                    '__import__': __import__,
                    'abs': abs,
                    'any': any,
                    'all': all,
                    'len': len,
                    'max': max,
                    'min': min,
                    'range': range,
                    'sum': sum,
                    'enumerate': enumerate,
                    'isinstance': isinstance,
                    'type': type,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'ValueError': ValueError,
                    'TypeError': TypeError,
                    'AttributeError': AttributeError,
                    'KeyError': KeyError,
                    'IndexError': IndexError,
                    'Exception': Exception,
                    'StopIteration': StopIteration,
                    'RuntimeError': RuntimeError,
                    'NotImplementedError': NotImplementedError,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed,
                    'zip': zip
                }
            }
            
            # First check if function code is complete
            if not self._is_function_complete(function_code):
                print(f"Generated function is incomplete:")
                print(f"Generated code:\n{function_code}")
                print("Using simple fallback function")
                function_code = """def correct_value(x):
    if isinstance(x, str):
        return x.strip()
    return x"""
            
            # Validate syntax before execution
            try:
                compile(function_code, '<string>', 'exec')
            except SyntaxError as e:
                print(f"Syntax error in generated function: {e}")
                print(f"Generated code:\n{function_code}")
                # Try to fix common syntax errors
                function_code = self._fix_function_syntax(function_code, str(e))
                try:
                    compile(function_code, '<string>', 'exec')
                    print("Fixed syntax error successfully")
                    print(f"Fixed code:\n{function_code}")
                except SyntaxError as e2:
                    print(f"Could not fix syntax error: {e2}")
                    print(f"Attempted fix:\n{function_code}")
                    print("Using simple fallback function")
                    # Use a simple fallback function instead of raising an error
                    function_code = """def correct_value(x):
    if isinstance(x, str):
        return x.strip()
    return x"""
            
            # Execute the function definition
            try:
                exec(function_code, safe_globals)
                correction_func = safe_globals.get('correct_value')
                
                if correction_func is None:
                    raise ValueError("Function 'correct_value' not found in generated code")
            except Exception as e:
                print(f"Execution error: {e}")
                print(f"Generated code:\n{function_code}")
                raise ValueError(f"Function execution failed: {e}")
            
            # Cache the function
            self.function_cache[function_hash] = correction_func
        
        # Apply the function
        return correction_func(input_value)
    
    def _validate_correction(self, corrected_value: Any, error_cell, context: Dict[str, Any]) -> bool:
        """Validate the corrected value."""
        
        # Check if the value actually changed (unless it was already correct)
        if corrected_value == error_cell.value and error_cell.confidence > 0.8:
            return False
        
        # Type consistency check
        if 'sample_values' in context and context['sample_values']:
            sample_types = set(type(v) for v in context['sample_values'])
            if type(corrected_value) not in sample_types and corrected_value is not None:
                return False
        
        # Basic sanity checks
        if isinstance(corrected_value, str):
            # Check for reasonable length
            if len(corrected_value) > 1000:
                return False
            # Check for valid characters (no control characters)
            if any(ord(c) < 32 and c not in '\n\t' for c in corrected_value):
                return False
        
        return True
    
    def _refine_function(self, 
                        function_code: str, 
                        error_cell, 
                        failed_result: Any, 
                        context: Dict[str, Any]) -> str:
        """Refine the function based on validation failure."""
        
        if self.use_api:
            return self._refine_function_api(function_code, error_cell, failed_result, context)
        else:
            return self._refine_function_simple(function_code, error_cell, failed_result, context)
    
    def _refine_function_api(self, function_code, error_cell, failed_result, context):
        """Refine function using API model."""
        refinement_prompt = f"""
The following correction function failed validation:

{function_code}

Input: '{error_cell.value}'
Output: '{failed_result}'
Expected behavior: Should correct {error_cell.error_type.value} error

Please generate an improved version of the function that:
1. Handles the input value correctly
2. Produces a valid output
3. Fixes the specific error type

Improved function:
"""
        
        try:
            response = self.llm_provider.generate(
                refinement_prompt,
                temperature=self.config.get('temperature', 0.3),
                max_tokens=self.config.get('max_tokens', 300)
            )
            
            refined_function = self._extract_function_from_response(response.text)
            return refined_function if refined_function else function_code
            
        except Exception as e:
            print(f"Function refinement failed: {e}")
            return function_code
    
    def _refine_function_simple(self, function_code, error_cell, failed_result, context):
        """Simple function refinement for local models."""
        # Add basic error handling
        if 'try:' not in function_code:
            lines = function_code.split('\n')
            if len(lines) > 1:
                indent = '    '
                refined_lines = [lines[0]]  # def line
                refined_lines.append(f'{indent}try:')
                for line in lines[1:]:
                    refined_lines.append(f'{indent}{line}')
                refined_lines.append(f'{indent}except:')
                refined_lines.append(f'{indent}{indent}return x')
                
                return '\n'.join(refined_lines)
        
        return function_code
    
    def _is_function_complete(self, function_code: str) -> bool:
        """Check if the generated function code is complete and valid."""
        if not function_code or not function_code.strip():
            return False
            
        lines = function_code.strip().split('\n')
        
        # Must start with def
        if not any(line.strip().startswith('def correct_value') for line in lines):
            return False
            
        # Must have at least some body content after the def line
        def_line_found = False
        has_body = False
        
        for line in lines:
            if line.strip().startswith('def correct_value'):
                def_line_found = True
                continue
            
            if def_line_found and line.strip():
                # Check if this is a proper indented line (function body)
                if line.startswith('    ') or line.startswith('\t'):
                    has_body = True
                    break
        
        if not has_body:
            return False
            
        # Check for incomplete/truncated code indicators
        last_line = lines[-1].strip()
        
        # Check for incomplete statements
        incomplete_indicators = [
            'try:', 'except:', 'if ', 'elif ', 'else:', 'for ', 'while ',
            'def ', 'class ', 'with ', '# ', '//', '"""', "'''",
        ]
        
        # If last line ends with these patterns, it's likely incomplete
        for indicator in incomplete_indicators:
            if last_line.endswith(indicator.rstrip(':')) and not last_line.endswith(':'):
                return False
                
        # Check for truncated words (incomplete tokens)
        if last_line and not last_line.endswith((':', ')', '}', ']', '"', "'", 'x', 'X')):
            # Last word should be complete
            last_word = last_line.split()[-1] if last_line.split() else ""
            if len(last_word) < 2 and last_word.isalpha():  # Single letters might be truncated
                return False
                
        # Check for proper function structure
        has_return = any('return' in line for line in lines)
        if not has_return:
            # Function should have at least one return statement or be very simple
            return False
            
        return True

    def _fix_function_syntax(self, function_code: str, error_message: str) -> str:
        """Try to fix basic syntax errors in the function."""
        
        # Common fixes
        fixed_code = function_code
        
        # Fix parentheses mismatches
        if "unmatched ')'" in error_message or "unexpected ')'" in error_message:
            lines = fixed_code.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Fix line-by-line parentheses mismatches
                fixed_line = line
                open_parens = line.count('(')
                close_parens = line.count(')')
                
                if close_parens > open_parens:
                    # Remove extra closing parentheses
                    extra_close = close_parens - open_parens
                    # Remove from the end, but be smart about it
                    for _ in range(extra_close):
                        # Find the last ')' that's not essential
                        last_paren_idx = fixed_line.rfind(')')
                        if last_paren_idx != -1:
                            # Check if removing this parenthesis makes sense
                            before_paren = fixed_line[:last_paren_idx]
                            after_paren = fixed_line[last_paren_idx+1:]
                            
                            # Don't remove if it's part of a function call
                            if not (before_paren.rstrip().endswith(')')):
                                fixed_line = before_paren + after_paren
                            else:
                                break
                
                elif open_parens > close_parens:
                    # Add missing closing parentheses at the end of line
                    fixed_line += ')' * (open_parens - close_parens)
                
                fixed_lines.append(fixed_line)
            
            fixed_code = '\n'.join(fixed_lines)
        
        # Fix unmatched quotes
        if "EOL while scanning string literal" in error_message or "unterminated string literal" in error_message:
            # Try to fix string literal issues
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                single_quotes = line.count("'")
                double_quotes = line.count('"')
                
                # Fix unmatched single quotes
                if single_quotes % 2 == 1:
                    line += "'"
                
                # Fix unmatched double quotes  
                if double_quotes % 2 == 1:
                    line += '"'
                
                lines[i] = line
            fixed_code = '\n'.join(lines)
        
        # Fix indentation issues
        if 'IndentationError' in error_message:
            lines = function_code.split('\n')
            fixed_lines = []
            for i, line in enumerate(lines):
                if i == 0:  # def line
                    fixed_lines.append(line)
                elif line.strip():  # non-empty line
                    if not line.startswith('    ') and not line.startswith('\t'):
                        fixed_lines.append('    ' + line.strip())
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            fixed_code = '\n'.join(fixed_lines)
        
        # Fix missing imports
        if 're' in function_code and 'import re' not in function_code:
            fixed_code = 'import re\n' + fixed_code
        
        # Fix missing colons after function definition
        if "invalid syntax" in error_message.lower():
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and not line.strip().endswith(':'):
                    lines[i] = line.rstrip() + ':'
            fixed_code = '\n'.join(lines)
        
        # Fix specific common patterns
        import re
        lines = fixed_code.split('\n')
        for i, line in enumerate(lines):
            original_line = line
            
            # Fix extra closing parentheses in if statements
            # Pattern: "if condition or condition):" or "if condition or condition)"
            if 'if ' in line and (' or ' in line or ' and ' in line):
                # Remove extra closing parentheses at the end of if statements
                # Match patterns like "if ... or ...):" or "if ... or ...)"
                line = re.sub(r'(\s+or\s+[^)]+)\)(\s*:?\s*)$', r'\1\2', line)
                line = re.sub(r'(\s+and\s+[^)]+)\)(\s*:?\s*)$', r'\1\2', line)
                
                # Also handle cases like "if not isinstance(x, str) or x is None):"
                line = re.sub(r'(if\s+[^)]+)\)(\s*:?\s*)$', r'\1\2', line)
                
            # Fix missing closing parentheses in function calls
            # Pattern: "function(" without closing ")"
            elif '(' in line and ':' in line:
                # Look for function calls that are missing closing parentheses
                # Common patterns: .strip(, .lower(, .upper(, etc.
                line = re.sub(r'\.(\w+)\(\s*:', r'.\1():', line)
                line = re.sub(r'(\w+)\(\s*:', r'\1():', line)
            
            # Fix cases where there are extra closing parens anywhere in conditional statements
            if 'if ' in line and ')' in line:
                # Count opening and closing parens
                open_count = line.count('(')
                close_count = line.count(')')
                if close_count > open_count:
                    # Remove the extra closing parentheses from the end
                    extra_closes = close_count - open_count
                    for _ in range(extra_closes):
                        # Remove the last closing parenthesis
                        last_close_idx = line.rfind(')')
                        if last_close_idx != -1:
                            line = line[:last_close_idx] + line[last_close_idx+1:]
            
            lines[i] = line
                
        fixed_code = '\n'.join(lines)
        
        return fixed_code
    
    def _simple_correction(self, value: Any) -> Any:
        """Apply a simple correction as fallback."""
        if isinstance(value, str):
            return value.strip()
        return value
    
    def _calculate_confidence(self, corrected_value: Any, error_cell, function_code: str) -> float:
        """Calculate confidence in the correction."""
        
        base_confidence = self.config.get('confidence_threshold', 0.7)
        
        # Adjust based on function complexity
        if function_code and len(function_code.split('\n')) > 2:
            base_confidence += 0.1
        
        # Adjust based on value change
        if corrected_value != error_cell.value:
            base_confidence += 0.1
        else:
            base_confidence -= 0.2
        
        # Adjust based on error type
        from gidcl import ErrorType
        if error_cell.error_type in [ErrorType.FORMATTING, ErrorType.PATTERN]:
            base_confidence += 0.1  # Explicit methods work well for these
        
        return min(max(base_confidence, 0.0), 1.0)


class FunctionLibrary:
    @staticmethod
    def common_string_fixes():
        return {
            'trim_whitespace': lambda x: x.strip() if isinstance(x, str) else x,
            'title_case': lambda x: x.title() if isinstance(x, str) else x,
            'lower_case': lambda x: x.lower() if isinstance(x, str) else x,
            'remove_special_chars': lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else x,
            'fix_phone_format': lambda x: re.sub(r'(\d{3})(\d{3})(\d{4})', r'\1-\2-\3', x) if isinstance(x, str) else x,
        }
    
    @staticmethod
    def common_numeric_fixes():
        return {
            'remove_currency': lambda x: float(re.sub(r'[$,]', '', str(x))) if '$' in str(x) else x,
            'percentage_to_decimal': lambda x: float(str(x).rstrip('%')) / 100 if '%' in str(x) else x,
            'round_to_int': lambda x: round(float(x)) if isinstance(x, (int, float, str)) else x,
        }