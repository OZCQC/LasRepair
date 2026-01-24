import os
import dotenv
from openai import OpenAI
dotenv.load_dotenv()

test = os.environ.get("LIGHTNING_API_KEY")
print(test)

client = OpenAI(
        base_url="https://lightning.ai/api/v1/",
        api_key=os.environ.get("LIGHTNING_API_KEY"),
    )
response = client.chat.completions.create(
        model="openai/gpt-5-nano",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}]
            },
        ],
    )
print(response.choices[0].message.content)