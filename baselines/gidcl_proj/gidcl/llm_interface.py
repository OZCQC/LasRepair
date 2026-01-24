"""
LLM接口模块 - 支持OpenAI API
"""

import os
from typing import List, Dict, Any, Optional
import json
import re


class LLMInterface:
    """LLM接口类，支持OpenAI API"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        初始化LLM接口
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not self.api_key:
            raise ValueError(
                "需要提供OpenAI API密钥。\n"
                "方法1: 传入api_key参数\n"
                "方法2: 设置环境变量 OPENAI_API_KEY\n"
                "方法3: 在.env文件中设置 OPENAI_API_KEY"
            )
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
    
    def generate_detection_pattern(
        self,
        column_name: str,
        dirty_clean_pairs: List[tuple],
        column_values: List[Any],
        previous_attempt: Optional[str] = None,
        wrong_cases: Optional[List[tuple]] = None
    ) -> str:
        """
        生成错误检测模式
        
        Args:
            column_name: 列名
            dirty_clean_pairs: (脏值, 干净值, 是否错误)的列表
            column_values: 列中的所有唯一值
            previous_attempt: 之前的尝试（用于改进）
            wrong_cases: 之前错误的案例
        
        Returns:
            正则表达式模式
        """
        # 构建提示
        if previous_attempt is None:
            prompt = self._build_detection_prompt_initial(
                column_name, dirty_clean_pairs, column_values
            )
        else:
            prompt = self._build_detection_prompt_refine(
                column_name, previous_attempt, wrong_cases
            )
        
        # 调用LLM
        response = self._call_llm(prompt)
        
        # 提取模式
        pattern = self._extract_pattern(response)
        
        return pattern
    
    def generate_corruption_function(
        self,
        column_name: str,
        detection_pattern: str,
        dirty_clean_pairs: List[tuple]
    ) -> str:
        """
        生成数据污染函数描述
        
        Args:
            column_name: 列名
            detection_pattern: 检测模式
            dirty_clean_pairs: (脏值, 干净值)对
        
        Returns:
            污染函数描述
        """
        prompt = f"""You are a data quality expert. Based on the error detection pattern and examples, generate a corruption function that can transform clean values into dirty values with similar errors.

Column: {column_name}
Detection Pattern: {detection_pattern}

Examples of dirty -> clean transformations:
{self._format_pairs(dirty_clean_pairs[:5])}

Generate a Python function description that can corrupt clean values. The function should:
1. Be simple and follow the error pattern
2. Use random operations when appropriate
3. Return the corrupted value

Format your response as:
FUNCTION: [brief description of the corruption logic]

Example response:
FUNCTION: randomly replace a digit with 'x' in the value
"""
        
        response = self._call_llm(prompt)
        
        # 提取函数描述
        match = re.search(r'FUNCTION:\s*(.+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return response.strip()
    
    def generate_correction_function(
        self,
        column_name: str,
        dirty_clean_pairs: List[tuple],
        detection_pattern: str
    ) -> str:
        """
        生成错误修正函数描述
        
        Args:
            column_name: 列名
            dirty_clean_pairs: (脏值, 干净值)对
            detection_pattern: 检测模式
        
        Returns:
            修正函数描述
        """
        prompt = f"""You are a data quality expert. Generate a correction function to fix errors in the column.

Column: {column_name}
Detection Pattern: {detection_pattern}

Examples of corrections needed:
{self._format_pairs(dirty_clean_pairs[:5])}

Generate a Python function description that can correct the dirty values. The function should:
1. Be simple and deterministic
2. Handle the most common error types
3. Return the corrected value

Format your response as:
FUNCTION: [brief description of the correction logic]

Example response:
FUNCTION: remove everything after and including comma, then uppercase
"""
        
        response = self._call_llm(prompt)
        
        # 提取函数描述
        match = re.search(r'FUNCTION:\s*(.+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return response.strip()
    
    def correct_value(
        self,
        dirty_value: Any,
        column_name: str,
        examples: List[tuple],
        context_values: List[Any]
    ) -> Any:
        """
        使用LLM修正单个值
        
        Args:
            dirty_value: 脏值
            column_name: 列名
            examples: (脏值, 干净值)示例对
            context_values: 上下文中的其他值
        
        Returns:
            修正后的值
        """
        prompt = f"""You are a data quality expert. Correct the following dirty value based on the examples and context.

Column: {column_name}
Dirty Value: {dirty_value}

Examples of corrections:
{self._format_pairs(examples[:5])}

Context values from the same column:
{', '.join(str(v) for v in context_values[:10])}

Provide ONLY the corrected value, nothing else. If the value is already correct, return it as is.
"""
        
        response = self._call_llm(prompt)
        
        # 清理响应
        corrected = response.strip().strip('"').strip("'")
        
        # 尝试转换类型
        try:
            if dirty_value is not None and isinstance(dirty_value, (int, float)):
                if '.' in corrected:
                    return float(corrected)
                else:
                    return int(corrected)
        except (ValueError, TypeError):
            pass
        
        return corrected
    
    def _build_detection_prompt_initial(
        self,
        column_name: str,
        dirty_clean_pairs: List[tuple],
        column_values: List[Any]
    ) -> str:
        """构建初始检测提示"""
        clean_examples = [c for d, c, is_err in dirty_clean_pairs if not is_err]
        dirty_examples = [d for d, c, is_err in dirty_clean_pairs if is_err]
        
        prompt = f"""You are a data quality expert. Analyze the clean and dirty values to identify the pattern that distinguishes them.

Column: {column_name}

Clean values (correct):
{', '.join(str(v) for v in clean_examples[:10])}

Dirty values (errors):
{', '.join(str(v) for v in dirty_examples[:10])}

All unique values in column:
{', '.join(str(v) for v in column_values[:20])}

Generate a regular expression pattern that matches ONLY the clean values. The pattern should:
1. Be as specific as possible
2. Match the format/structure of clean values
3. Reject dirty values

Format your response as:
PATTERN: r'your_regex_pattern_here'

Example responses:
PATTERN: r'^[A-Z]{{2}}$'  (for 2-letter state codes)
PATTERN: r'^\d{{5}}$'  (for 5-digit zip codes)
PATTERN: r'^[A-Za-z\s]+$'  (for names with letters and spaces)
"""
        
        return prompt
    
    def _build_detection_prompt_refine(
        self,
        column_name: str,
        previous_pattern: str,
        wrong_cases: List[tuple]
    ) -> str:
        """构建改进检测提示"""
        prompt = f"""You are a data quality expert. The previous pattern failed on some cases. Refine it.

Column: {column_name}
Previous Pattern: {previous_pattern}

Cases where the pattern failed:
{self._format_pairs(wrong_cases[:5])}

Generate an improved regular expression pattern that:
1. Fixes the issues with wrong cases
2. Still matches correct values
3. Is more precise

Format your response as:
PATTERN: r'your_improved_regex_pattern_here'
"""
        
        return prompt
    
    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 提示文本
            temperature: 温度参数
        
        Returns:
            LLM的响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data quality expert specializing in data cleaning and error detection."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"LLM API调用失败: {e}")
            # 返回一个默认模式
            return "PATTERN: r'.*'"
    
    def _extract_pattern(self, response: str) -> str:
        """从响应中提取正则表达式模式"""
        # 尝试提取PATTERN:后的内容
        match = re.search(r'PATTERN:\s*r?["\'](.+?)["\']', response, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # 尝试提取引号中的内容
        match = re.search(r'r?["\'](\^.+?\$)["\']', response)
        if match:
            return match.group(1)
        
        # 默认返回匹配所有
        return r'.*'
    
    def _format_pairs(self, pairs: List[tuple]) -> str:
        """格式化(脏值, 干净值)对"""
        if not pairs:
            return "No examples"
        
        formatted = []
        for item in pairs:
            if len(item) == 2:
                dirty, clean = item
                formatted.append(f"  {dirty} -> {clean}")
            elif len(item) == 3:
                dirty, clean, is_err = item
                if is_err:
                    formatted.append(f"  {dirty} -> {clean} (error)")
                else:
                    formatted.append(f"  {dirty} (clean)")
        
        return '\n'.join(formatted)