from openai import OpenAI
import pandas as pd
import os
import time
from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_API_BASE
)


class base_llm():
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_API_BASE

    def call_with_messages_R1(self, prompt_info, temp=0.1, max_tokens=8192):
        res = self.call_with_messages_V3(prompt_info=prompt_info, temp=temp,
                                         model_name="Pro/deepseek-ai/DeepSeek-R1", max_tokens=max_tokens)
        return res

    def call_with_messages_V3(self, prompt_info, temp=0.1, model_name="Pro/deepseek-ai/DeepSeek-V3.2", max_tokens=8192):
        #SiliconFlow-API
        prompt=self.system_prompt+'\n'+prompt_info
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        print(len(prompt))
        for attempt in range(5):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                response=completion.choices[0].message.content
                result=response.split('</think>')[-1].strip() if response else ""
                reason_response=getattr(completion.choices[0].message, 'reasoning_content', None)
                reason_result=reason_response.split('</think>')[-1].strip() if reason_response else ""
                if len(result)>0:
                    return result
                elif len(reason_result)>0:
                    return reason_result
                else:
                    print(f"warning: output content is empty, retrying ({attempt+1}/5)...")
            except Exception as e:
                print(f"warning: API call failed ({attempt+1}/5): {e}")
        print("error: failed to generate content after 5 trials!")
        return -1

    def call_with_messages_small(self, prompt_info, temp=0, model_name="Qwen/Qwen3-8B", max_tokens=4096):
        prompt=self.system_prompt+'\n'+prompt_info
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content