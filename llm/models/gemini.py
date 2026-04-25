from .llm_base import LLMBase
from google import genai
from google.genai import types
from utils.setting import setting


class GeminiParser(LLMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = genai.Client(
            api_key=setting.API_KEY_gemini,
            http_options=types.HttpOptions(timeout=self.timeout)
        )

    def generate(self, prompt):
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text

    def extract_message(self, e):
        status_code = e.code
        print(f"HTTP error code: {status_code}")
        error_message = e.status
        print(f"Error message: {error_message}")
        return status_code, error_message
