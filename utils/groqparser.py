from groq import Groq
from llm import LLMParser
import os
from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent
from setting import setting

# os.environ['OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'] = 'true'
# os.environ['OTEL_INSTRUMENTATION_OPENAI_CAPTURE_MESSAGE_CONTENT'] = 'true'
# logfire.configure()
# logfire.instrument_openai()

class GroqParser(LLMParser):

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

if __name__ == '__main__':
    test = GroqParser()
    prompt = "Create the template for log by replacing all the dynamic variables with <*>"
    with open('sample.log', 'r') as f:
        log = f.read()
    prompt = f'{prompt} {log}'.strip()
    completion = test.wrapper(prompt)
    test.load_template(completion)
    # completion1 = test.wrapper(prompt)
    # test.load_template(completion1)
    # completion2 = test.wrapper(prompt)
    # test.load_template(completion2)
    # completion3 = test.wrapper(prompt)
    # test.load_template(completion3)
    # completion4 = test.wrapper(prompt)
    # test.load_template(completion4)
