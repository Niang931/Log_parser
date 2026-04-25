from llm import LLMParser
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from setting import setting


class GeminiParser(LLMParser):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = genai.Client(api_key=setting.API_KEY_gemini,
                                   http_options=types.HttpOptions(timeout=self.TIMEOUT_THRESHOLD))


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

if __name__ == '__main__':
    test= GeminiParser()
    prompt = "Create the template for log by replacing all the dynamic variables with <*>"
    with open('sample.log', 'r') as f:
        log = f.read()
    prompt = f'{prompt} {log}'.strip()
    completion = test.wrapper(prompt)
    test.load_template(completion)
    completion = test.wrapper(prompt)
    test.load_template(completion)
    completion = test.wrapper(prompt)
    test.load_template(completion)
    completion = test.wrapper(prompt)
    test.load_template(completion)

