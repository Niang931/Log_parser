from groq import Groq
from .llm_base import LLMBase
from utils.setting import setting


class GroqParser(LLMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Groq(api_key=setting.GROQ_API_KEY)

    def generate(self, prompt):

        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            max_completion_tokens=4000,
            top_p=1,
            reasoning_effort="medium",
            stream=False,
            stop=None
        )

        return completion.choices[0].message.content

    def extract_message(self, e):
        status_code = e.code
        print(f"HTTP error code: {status_code}")
        error_message = e.status
        print(f"Error message: {error_message}")
        return status_code, error_message
