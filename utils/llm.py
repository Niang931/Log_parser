import json
import time
from google import genai
from google.genai.errors import ServerError, ClientError
import random
from google.genai import types
from threading import *
import requests
import logging
from token_bucket import TokenBucket

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('test.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class LLMParser:
    SAFE_TO_RETRY = [429, 500, 503]
    MAX_RETRY = 5
    BASE_DELAY = 1
    MAX_PARALLEL_CALL = 5
    TIMEOUT_THRESHOLD = 100000

    def __init__(self):
        self.client = genai.Client(api_key="AIzaSyA2Y-rN2mcjp1JY2jnPpwRn01edd5rYDrU",
                                   http_options=types.HttpOptions(timeout=self.TIMEOUT_THRESHOLD))
        self.count = 0
        self.obj = Semaphore(self.MAX_PARALLEL_CALL)
        self.token_bucket = TokenBucket()

    def call_llm(self, prompt):
        self.count += 1
        try:
            self.token_bucket.consume()
            self.obj.acquire()
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )

            self.obj.release()
            logger.info(f'{self.obj}')
            logger.info(f'Remaining tokens:{self.token_bucket.tokens}')
            logger.info('API call successful')
            return response.text


        except (ServerError, ClientError) as e:
            logger.error('Rate limiting')
            status_code, error_message = self.extract_error(e)
            if status_code in self.SAFE_TO_RETRY:
                """Server side errors & rate limiting, safe to retry"""
                if self.count < self.MAX_RETRY:
                    jitter = random.random() * 10
                    delay_time = self.BASE_DELAY * (2 ** self.count) + jitter
                    time.sleep(delay_time)
                    self.call_llm(prompt)
                    logger.info('LLM Retry')
                else:
                    logger.info('Out of retries')


    def load_template(self, text):
        with open('template.txt','w') as f:
            print(text, file=f)
        logger.info('Load templates')

    def extract_error(self, e):
        status_code = e.code
        print(f"HTTP error code: {status_code}")
        error_message =e.status
        print(f"Error message {error_message}")
        return status_code, error_message



if __name__ == '__main__':
    parser = LLMParser()
    prompt = ("Given this log file: Create the template for it by replacing all the dynamic variables with <*> "
              "while keeping the fixed values"
              "Answer ONLY the template with no extra word ")
    with open('sample.log', 'r') as f:
        log = f.read()
    response= parser.call_llm(f'{prompt} {log}')
    parser.load_template(response)
    # response1 = parser.call_llm(f'{prompt} {log}')
    # parser.load_template(response1)
    # response2 = parser.call_llm(f'{prompt} {log}')
    # parser.load_template(response2)
    # response3 = parser.call_llm(f'{prompt} {log}')
    # parser.load_template(response3)
    # response4 = parser.call_llm(f'{prompt} {log}')
    # parser.load_template(response4)

