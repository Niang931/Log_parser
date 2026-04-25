from groq import Groq
from .llm_base import LLMBase
from utils.setting import setting


class GroqParser(LLMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Groq(api_key=setting.GROQ_API_KEY)

    def generate(self, prompt, temperature=1, max_token=512):

        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_token,
            top_p=1,
            reasoning_effort="medium",
            stream=False,
            stop=None
        )

        return completion.choices[0].message.content

    def extract_message(self, e):
        if e.__class__.__name__ == "RateLimitError":
            print("Hit rate limit. Retrying might help.")
            return 429, str(e)

        status_code = getattr(e, "status_code", None)
        error_message = str(e)

        print(f"HTTP error code: {status_code}")
        print(f"Error message: {error_message}")

        return status_code, error_message
