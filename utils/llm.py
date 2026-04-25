import json
import time
import random
from threading import *
import logging
from token_bucket import TokenBucket
import logfire
import os
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('test.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



# os.environ['OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'] = 'true'
# logfire.configure()
# logfire.instrument_google_genai()
# logfire.instrument_openai()

class LLMParser(ABC):
    SAFE_TO_RETRY = [429, 500, 503, 502]
    MAX_RETRY = 5
    BASE_DELAY = 1
    MAX_PARALLEL_CALL = 5
    TIMEOUT_THRESHOLD = 100000

    def __init__(self):
        self.count = 0
        self.obj = Semaphore(self.MAX_PARALLEL_CALL)
        self.token_bucket = TokenBucket()

    def wrapper(self, prompt):
        self.count += 1
        self.token_bucket.consume()
        self.obj.acquire()
        try:
            response = self.generate(prompt)
            self.obj.release()
            logger.info(f'{self.obj}')
            logger.info(f'Remaining tokens:{self.token_bucket.tokens}')
            logger.info('API call successful')
            return response

        except Exception as e:
            status_code, error_message = self.extract_message(e)
            logger.error(f'{error_message}')
            if status_code in self.SAFE_TO_RETRY:
                """Server side errors & rate limiting, safe to retry"""
                if self.count < self.MAX_RETRY:
                    jitter = random.random() * 10
                    delay_time = self.BASE_DELAY * (2 ** self.count) + jitter
                    time.sleep(delay_time)
                    logger.info('LLM Retry')
                    self.wrapper(prompt)
                else:
                    logger.info('Out of retries')

    @abstractmethod
    def generate(self, prompt):
        pass

    def load_template(self, text):
        if text:
            with open('template.txt','w') as in_file:
                print(text, file=in_file)
            logger.info('Load templates')

    @abstractmethod
    def extract_message(self, e):
        pass

if __name__ == '__main__':
    parser = LLMParser()
    prompt = "Create the template for log by replacing all the dynamic variables with <*>"
    with open('sample.log', 'r') as f:
        log = f.read()
    prompt = f'{prompt} {log}'.strip()
    print(prompt)


