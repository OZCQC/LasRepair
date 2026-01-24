# import openai
# import pandas as pd
# import numpy as np
# import json
# import re

# PROMPT_TEMPLATE = """
# You are a data scientist. I will give you:
# 1) A list of column names (Columns).
# 2) A small set of sample rows (Samples) from the table.

# TASK: For every column in Columns, decide which other columns (from Columns excluding the target itself) are REQUIRED to predict that column. 
# - Return ONLY a single JSON object (dictionary) and nothing else.
# - Each key must be a column name (string). Each value must be a JSON array (list) of column names (strings) that are required to predict that key column.
# - If no other column is required to reasonably predict the target, use an empty list `[]`.
# - The sets should be complete, and should not miss any required columns.
# - Do not include confidences, explanations, suggestions, or any extra fields.
# - If there is an obvious data-leak (a column contains future or direct target information), still list it if it is strictly required to reconstruct the target, but do not add comments — only include it in the list.
# - If a timestamp column is present, assume only past information relative to the target event is allowed; do not use future-derived columns unless those columns are clearly historical.
# - Output must be valid JSON compatible with standard parsers.

# INPUT:
# Samples: {samples}

# Now produce the JSON dictionary mapping every column to its required-predictor-columns.

# """.strip()

# API_KEYS = {
#         "openai": "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"
# }
# clean_csv = "/data1/qianc/EMCL/datasets/flight/clean.csv"
# clean_df = pd.read_csv(clean_csv)
# sample = clean_df.iloc[:20]
# input_text = PROMPT_TEMPLATE.format(samples=sample)
# client = openai.OpenAI(api_key=API_KEYS["openai"])
# response = client.chat.completions.create(
#     model="gpt-4o",
#     max_tokens=1000,
#     messages=[
#         {"role": "user", "content": input_text}
#     ]
# )
# print(response.choices[0].message.content)

# # for i in range(len(clean_df.columns)):
# #     for j in range(i, len(clean_df.columns)):
# #         test_1 = clean_df.iloc[:20]
# #         input_text = PROMPT_TEMPLATE.format(columns=str(clean_df.columns), samples=test_1.tolist())
# #         # API_KEY = "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"
# #         client = openai.OpenAI(api_key=API_KEYS["openai"])

# #         response = client.chat.completions.create(
# #             model="gpt-4o",
# #             max_tokens=1000,
# #             messages=[
# #                 {
# #                     "role": "user",
# #                     "content": input_text
# #                 }
# #             ]
# #         )
        
# #         # Extract JSON from the response
# #         response_text = response.choices[0].message.content
        
# #         # Find JSON in the response using regex
# #         json_match = re.search(r'\{[^{}]*\}', response_text)
# #         if json_match:
# #             try:
# #                 result = json.loads(json_match.group())
# #                 strength = result.get('strength', 0.0)
# #                 confidence = result.get('confidence', 0.0)
# #                 casual_matrix[i, j] = strength  # or confidence, depending on what you want to store
# #                 print(f"Column {i}-{j}: strength={strength}, confidence={confidence}")
# #             except json.JSONDecodeError:
# #                 print(f"Failed to parse JSON for columns {i}-{j}")
# #                 casual_matrix[i, j] = 0.0
# #         else:
# #             print(f"No JSON found in response for columns {i}-{j}")
# #             casual_matrix[i, j] = 0.0

# # print("Causal matrix completed!")
# # print(casual_matrix)

# # beers = np.load("/data1/qianc/beers_group.npy")
# # print(beers)

import openai
import pandas as pd
import numpy as np
import json
import re
import os

PROMPT_TEMPLATE = """
You are a careful analytical assistant. I will provide you with a dataset containing {n} rows of data with multiple columns.

TASK: 
- For EVERY pair of columns (A, B) where A and B are different, estimate the causal strength between them. 
- You need to consider each pair of columns carefully and give a high strength if you think there is any causal relationship between them.
- You should also give high strength if there is functional dependency between them.

Return exactly ONE JSON object (only JSON, no extra text) where:
- The key is a string in format "column_i->column_j" (where i and j are column indices, i < j)
- The value is an object with two fields:
  * "strength": a float in [0,1] indicating estimated causal strength (0 = no causal relationship, 1 = very strong causal relationship)
  * "confidence": a float in [0,1] indicating how confident you are in this judgement

Important rules:
1) Analyze ALL possible column pairs where i < j (upper triangle of the causal matrix)
2) Use the provided sample data to estimate relationships
3) Be conservative in your estimates
4) Output must be valid JSON only, no explanations

DATA:
Column names: {columns}
Sample data (first {n} rows):
{data}

Example output format:
{{
  "0->1": {{"strength": 0.3, "confidence": 0.6}},
  "0->2": {{"strength": 0.7, "confidence": 0.8}},
  "1->2": {{"strength": 0.1, "confidence": 0.4}}
}}

Please respond now with the JSON only.
""".strip()

API_KEYS = {
        "openai": os.environ["OPENAI_API_KEY"]
}
clean_csv = "/data1/qianc/EMCL/datasets/rayyan/clean.csv"
clean_df = pd.read_csv(clean_csv)

# 准备数据：取前20行
n_rows = 20
sample_data = clean_df.sample(n_rows, random_state=114)

# 将数据转换为更易读的格式
# data_str = sample_data.to_string()

# 构建prompt
input_text = PROMPT_TEMPLATE.format(
    columns=list(clean_df.columns), 
    n=n_rows,
    data=sample_data
)

print(f"准备调用API，分析 {len(clean_df.columns)} 列之间的所有关系...")
print(f"总共需要分析 {len(clean_df.columns) * (len(clean_df.columns) - 1) // 2} 个列对")

# 只调用一次API
client = openai.OpenAI(api_key=API_KEYS["openai"])
from openai import OpenAI
client = OpenAI(
    base_url="https://lightning.ai/api/v1/",
    api_key=os.environ["LIGHTNING_API_KEY"],
)

response = client.chat.completions.create(
    model="openai/gpt-5",
    messages=[
      {
        "role": "user",
        "content": [{"type": "text", "text": input_text}]
      },
    ],
)

# 提取响应
response_text = response.choices[0].message.content
print("\n=== API响应 ===")
print(response_text)
print("================\n")

# 初始化因果矩阵
casual_matrix = np.zeros((len(clean_df.columns), len(clean_df.columns)))

# 解析JSON响应
try:
    # 尝试找到JSON部分
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        
        # 遍历所有返回的列对关系
        for key, value in result.items():
            # 解析 "i->j" 格式
            match = re.match(r'(\d+)->(\d+)', key)
            if match:
                i = int(match.group(1))
                j = int(match.group(2))
                strength = value.get('strength', 0.0)
                confidence = value.get('confidence', 0.0)
                casual_matrix[i, j] = strength
                print(f"列 {i} ({clean_df.columns[i]}) -> 列 {j} ({clean_df.columns[j]}): strength={strength:.3f}, confidence={confidence:.3f}")
            else:
                print(f"警告: 无法解析键 '{key}'")
        
        print(f"\n成功解析 {len(result)} 个列对关系")
    else:
        print("错误: 在响应中未找到JSON")
        
except json.JSONDecodeError as e:
    print(f"JSON解析错误: {e}")
    print("响应文本:", response_text)

print("\n因果矩阵生成完成!")
print(casual_matrix)
np.save("/data1/qianc/flight_casual_matrix.npy", casual_matrix)
print(f"\n矩阵已保存到: /data1/qianc/flight_casual_matrix.npy")

# beers = np.load("/data1/qianc/hospital_group.npy")
# print(beers)