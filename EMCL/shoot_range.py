import openai
import pandas as pd
import numpy as np
import json
import re

PROMPT_TEMPLATE = """
You are a careful analytical assistant. Given a small sample of paired values for columns A and B, produce exactly ONE JSON object (only JSON, no extra text) with fields:
- strength: a float in [0,1] indicating estimated causal strength (higher means stronger evidence A causes B or B causes A)
- confidence: a float in [0,1] indicating how confident you are in this judgement

Important rules:
1) Use only the provided metrics and sample pairs. Do NOT invent new numerical values.
2) For strength and confidence, give numbers that reflect the combined evidence; be conservative.
3) Output pure JSON.

DATA SUMMARY:
Column A name: {colA}
Column B name: {colB}
n = {n}

Sample pairs (up to 100, truncated):
Sample A: {sample_A}
Sample B: {sample_B}

Please respond now with the JSON only.
""".strip()

API_KEYS = {
        "openai": "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"
}
clean_csv = "/data1/qianc/EMCL/datasets/beers/clean.csv"
clean_df = pd.read_csv(clean_csv)
casual_matrix = np.zeros((len(clean_df.columns), len(clean_df.columns)))
for i in range(len(clean_df.columns)):
    for j in range(i, len(clean_df.columns)):
        test_1 = clean_df.iloc[:100, i]
        test_2 = clean_df.iloc[:100, j]
        input_text = PROMPT_TEMPLATE.format(colA=str(clean_df.columns[i]), colB=str(clean_df.columns[j]), n=len(test_1), sample_A=test_1.tolist(), sample_B=test_2.tolist())
        # API_KEY = "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"
        client = openai.OpenAI(api_key=API_KEYS["openai"])

        response = client.chat.completions.create(
            model="gpt-4",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )
        
        # Extract JSON from the response
        response_text = response.choices[0].message.content
        
        # Find JSON in the response using regex
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                strength = result.get('strength', 0.0)
                confidence = result.get('confidence', 0.0)
                casual_matrix[i, j] = strength  # or confidence, depending on what you want to store
                print(f"Column {i}-{j}: strength={strength}, confidence={confidence}")
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for columns {i}-{j}")
                casual_matrix[i, j] = 0.0
        else:
            print(f"No JSON found in response for columns {i}-{j}")
            casual_matrix[i, j] = 0.0

print("Causal matrix completed!")
print(casual_matrix)

beers = np.load("/data1/qianc/beers_group.npy")
print(beers)
